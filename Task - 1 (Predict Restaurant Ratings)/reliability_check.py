
# TASK 1: CHECK MODEL RELIABILITY
# Compare Random Forest with and without Votes
# using 5-fold cross-validation.

import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

# 1. Load the same dataset
df = pd.read_csv("Dataset .csv")

target = "Aggregate rating"

base_features = [
    "Country Code",
    "City",
    "Cuisines",
    "Average Cost for two",
    "Has Table booking",
    "Has Online delivery",
    "Price range"
]

# 2. Prepare two feature sets
feature_sets = {
    "With Votes": base_features + ["Votes"],
    "Without Votes": base_features
}

# 3. Use the same 5 folds for a fair comparison
cv = KFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    "MAE": "neg_mean_absolute_error",
    "RMSE": "neg_root_mean_squared_error",
    "R2": "r2"
}

results = []

# 4. Train and evaluate each feature set
for name, features in feature_sets.items():

    print("\nChecking:", name)

    data = df[features + [target]].dropna(
        subset=[target]
    )

    X = data[features]
    y = data[target]

    categorical = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    numerical = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numerical),
        ("cat", categorical_pipeline, categorical)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ))
    ])

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=1
    )

    mae = -scores["test_MAE"]
    rmse = -scores["test_RMSE"]
    r2 = scores["test_R2"]

    results.append({
        "Feature Set": name,
        "Mean MAE": mae.mean(),
        "MAE Std": mae.std(),
        "Mean RMSE": rmse.mean(),
        "RMSE Std": rmse.std(),
        "Mean R2": r2.mean(),
        "R2 Std": r2.std()
    })

    print("Mean MAE:", round(mae.mean(), 3))
    print("Mean RMSE:", round(rmse.mean(), 3))
    print("Mean R2:", round(r2.mean(), 3))

# 5. Display and save the comparison
results_df = pd.DataFrame(results)

print("\n" + "=" * 65)
print("5-FOLD CROSS-VALIDATION COMPARISON")
print("=" * 65)
print(results_df.round(3).to_string(index=False))

results_df.to_csv(
    "reliability_results.csv",
    index=False
)

print("\nSaved results to reliability_results.csv")
print("Reliability check completed!")
