
import joblib
import pandas as pd

# Load the saved model
model = joblib.load("best_model.joblib")

# Enter sample restaurant details
restaurant = pd.DataFrame([{
    "Country Code": 1,
    "City": "New Delhi",
    "Cuisines": "North Indian, Chinese",
    "Average Cost for two": 800,
    "Has Table booking": "Yes",
    "Has Online delivery": "Yes",
    "Price range": 2,
    "Votes": 150
}])

# Predict the restaurant rating
prediction = model.predict(restaurant)[0]

# Keep the displayed estimate within the dataset's rating scale
prediction = max(0, min(5, prediction))

print("=" * 45)
print("RESTAURANT RATING PREDICTION")
print("=" * 45)
print("\nRestaurant Details:")
print(restaurant.to_string(index=False))
print(f"\nPredicted Rating: {prediction:.2f} / 5")
