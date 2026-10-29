
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix
)

# Load dataset
df = pd.read_csv("Dataset .csv")
df = df.dropna(subset=["Cuisines"]).copy()

# Use the first cuisine as the target
df["Primary Cuisine"] = df["Cuisines"].apply(
    lambda x: str(x).split(",")[0].strip()
)

# Keep the 15 most common cuisines
top_cuisines = df["Primary Cuisine"].value_counts().head(15).index
df = df[df["Primary Cuisine"].isin(top_cuisines)].copy()

# Features and target
features = [
    "Restaurant Name", "City", "Average Cost for two",
    "Has Table booking", "Has Online delivery",
    "Price range", "Votes"
]
X = df[features].copy()
y = df["Primary Cuisine"]

text_col = "Restaurant Name"
cat_cols = ["City", "Has Table booking", "Has Online delivery"]
num_cols = [
    "Average Cost for two", "Price range", "Votes"
]

X[text_col] = X[text_col].fillna("Unknown").astype(str)

# Preprocessing
preprocessor = ColumnTransformer([
    ("name", TfidfVectorizer(
        max_features=5000, ngram_range=(1, 2)
    ), text_col),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_cols),
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ]), num_cols)
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train LinearSVC
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LinearSVC(
        C=1.0, class_weight="balanced",
        random_state=42, max_iter=5000
    ))
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, predictions)
macro_f1 = f1_score(y_test, predictions, average="macro")

print("\nFINAL MODEL: LinearSVC")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1: {macro_f1:.4f}")
print("\nClassification Report:")
print(classification_report(
    y_test, predictions, zero_division=0
))

# Save trained model
joblib.dump(model, "cuisine_model.joblib")
print("Model saved as cuisine_model.joblib")

# Save predictions
results = pd.DataFrame({
    "Actual Cuisine": y_test.to_numpy(),
    "Predicted Cuisine": predictions
})
results.to_csv("cuisine_predictions.csv", index=False)
print("Predictions saved as cuisine_predictions.csv")

# Save confusion matrix
labels = model.named_steps["classifier"].classes_
cm = confusion_matrix(y_test, predictions, labels=labels)

plt.figure(figsize=(13, 10))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=labels, yticklabels=labels
)
plt.title("LinearSVC Cuisine Classification")
plt.xlabel("Predicted Cuisine")
plt.ylabel("Actual Cuisine")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.close()

print("Confusion matrix saved as confusion_matrix.png")
