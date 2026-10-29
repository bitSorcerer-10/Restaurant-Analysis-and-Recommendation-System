
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("Dataset .csv")

# Select features and target
features = [
    "Country Code",
    "City",
    "Average Cost for two",
    "Has Table booking",
    "Has Online delivery",
    "Price range",
    "Votes"
]
target = "Cuisines"

df = df[features + [target]].dropna(subset=[target]).copy()
df[target] = df[target].astype(str).str.strip()
df = df[df[target] != ""]

# Use the first cuisine listed as the main cuisine
df[target] = df[target].str.split(",").str[0].str.strip()

# Keep the 15 most common cuisine classes
top_cuisines = df[target].value_counts().head(15).index
df = df[df[target].isin(top_cuisines)].copy()

X = df[features]
y = df[target]

print("Records used:", len(df))
print("Cuisine classes:", y.nunique())

# Identify column types
categorical = [
    "City",
    "Has Table booking",
    "Has Online delivery"
]
numerical = [
    "Country Code",
    "Average Cost for two",
    "Price range",
    "Votes"
]

# Preprocess data
preprocessor = ColumnTransformer([
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical),
    ("num", SimpleImputer(strategy="median"), numerical)
])

# Build model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    ))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train model
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
print("\nCUISINE CLASSIFICATION RESULTS")
print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))
