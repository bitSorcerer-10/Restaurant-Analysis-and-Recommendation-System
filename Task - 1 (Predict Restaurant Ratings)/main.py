
# PROJECT: Predict Restaurant Ratings
# Step 1: Load libraries

import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import numpy as np

# Step 2: Load dataset
df = pd.read_csv("Dataset .csv")

print("=" * 55)
print("RESTAURANT RATING PREDICTION - MACHINE LEARNING")
print("=" * 55)

# Step 3: Select input features and target
features = [
    "Country Code",
    "City",
    "Cuisines",
    "Average Cost for two",
    "Has Table booking",
    "Has Online delivery",
    "Price range",
    "Votes"
]

target = "Aggregate rating"

data = df[features + [target]].copy()
data = data.dropna(subset=[target])

X = data[features]
y = data[target]

# Step 4: Identify column types
categorical_features = X.select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()

numerical_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

# Step 5: Preprocess data
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numerical_features),
    ("cat", categorical_transformer, categorical_features)
])

# Step 6: Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTotal records:", len(data))
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# Step 7: Create models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
}

results = []
best_model = None
best_r2 = float("-inf")
best_predictions = None

# Step 8: Train and evaluate each model
for name, model in models.items():
    print(f"\nTraining {name}...")

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    if r2 > best_r2:
        best_r2 = r2
        best_model = pipeline
        best_predictions = predictions

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

    print("MAE:", round(mae, 3))
    print("RMSE:", round(rmse, 3))
    print("R2 Score:", round(r2, 3))

# Step 9: Compare results
results_df = pd.DataFrame(results)

print("\n" + "=" * 55)
print("MODEL COMPARISON")
print("=" * 55)
print(results_df.to_string(index=False))

# Step 10: Save results
results_df.to_csv("model_comparison.csv", index=False)

print("\nResults saved to model_comparison.csv")
print("\nMachine learning training completed!")


# Save the best trained model
joblib.dump(best_model, "best_model.joblib")
print("Best model saved to best_model.joblib")

# Create actual vs predicted rating graph
plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    best_predictions,
    alpha=0.5
)

plt.plot(
    [0, 5],
    [0, 5],
    linestyle="--"
)

plt.xlabel("Actual Restaurant Rating")
plt.ylabel("Predicted Restaurant Rating")
plt.title("Actual vs Predicted Restaurant Ratings")
plt.xlim(0, 5)
plt.ylim(0, 5)
plt.grid(True)
plt.tight_layout()

plt.savefig("prediction_graph.png")
plt.show()

print("Graph saved to prediction_graph.png")

