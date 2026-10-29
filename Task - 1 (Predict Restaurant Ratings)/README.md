# Restaurant Rating Prediction Using Machine Learning

## 1. Project Overview

This project predicts restaurant ratings using Machine Learning techniques. It uses restaurant information such as location, cuisine, average cost, table booking availability, online delivery availability, price range, and votes to predict the restaurant's aggregate rating.

The project compares Linear Regression and Random Forest Regression to identify the better-performing model.

## 2. Project Objective

The main objectives are:

- To analyze restaurant data.
- To preprocess the dataset for Machine Learning.
- To train and compare different regression models.
- To evaluate model performance using standard metrics.
- To predict restaurant ratings for new restaurant details.

## 3. Technologies Used

- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Joblib
- **IDE:** Visual Studio Code
- **Dataset Format:** CSV

## 4. Dataset Description

The project uses a restaurant dataset containing 9,551 records and 21 original columns.

Selected input features include:

- Country Code
- City
- Cuisines
- Average Cost for Two
- Has Table Booking
- Has Online Delivery
- Price Range
- Votes

**Target Variable:** Aggregate Rating

The target variable represents the restaurant's overall rating.

## 5. Project Workflow

1. Load the restaurant dataset.
2. Explore and preprocess the data.
3. Select relevant input features and the target variable.
4. Handle missing values and prepare categorical features.
5. Split the dataset into training and testing sets.
6. Train Linear Regression and Random Forest Regression models.
7. Evaluate both models using MAE, RMSE, and R² score.
8. Compare the results and select the better-performing model.
9. Save the trained model and prediction results.
10. Predict ratings for new restaurant details.

## 6. Machine Learning Algorithms

### Linear Regression

Linear Regression predicts a numerical value by learning relationships between input features and the target variable.

### Random Forest Regression

Random Forest Regression combines predictions from multiple decision trees to produce a final numerical prediction. It can learn complex relationships between restaurant features and ratings.

## 7. Model Evaluation

The models were evaluated using three metrics:

- **MAE (Mean Absolute Error):** Measures the average absolute difference between actual and predicted ratings. Lower is better.
- **RMSE (Root Mean Squared Error):** Measures prediction error while penalizing larger errors more heavily. Lower is better.
- **R² Score:** Measures how well the model explains variation in the target variable. A score closer to 1 indicates a stronger fit.

### Results

| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| Linear Regression | 0.988 | 1.203 | 0.364 |
| Random Forest Regression | 0.204 | 0.319 | 0.955 |

Random Forest Regression performed better on the test set across all three metrics.

## 8. Sample Prediction

Example input:

- Country Code: 1
- City: New Delhi
- Cuisines: North Indian, Chinese
- Average Cost for Two: 800
- Table Booking: Yes
- Online Delivery: Yes
- Price Range: 2
- Votes: 150

**Predicted Restaurant Rating: 3.31 / 5**

This is an example prediction produced by the trained model.

## 9. Project Files

- `Dataset .csv` — Restaurant dataset
- `main.py` — Data preprocessing, model training, evaluation, and comparison
- `predict.py` — Predicts ratings for new restaurant details
- `model_comparison.csv` — Model performance comparison
- `best_model.joblib` — Saved trained model
- `prediction_graph.png` — Prediction visualization
- `README.md` — Project documentation

## 10. How to Run the Project

### Step 1: Install Python

Install Python and ensure it is available through the Windows `py` command.

### Step 2: Install Required Libraries

```bash
py -m pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

### Step 3: Run Model Training

```bash
py main.py
```

### Step 4: Generate a Prediction

```bash
py predict.py
```

Ensure that the dataset and Python files are in the same project folder.

## 11. Key Learnings

- Data preprocessing and feature selection
- Regression model training
- Training and testing dataset separation
- Model evaluation and comparison
- Saving and loading trained models
- Making predictions on new data

## 12. Conclusion

This project demonstrates how Machine Learning can be used to predict restaurant ratings from restaurant-related features. Two regression algorithms were compared, and Random Forest Regression achieved better performance on the test dataset.

The project provided practical experience in data preprocessing, model development, evaluation, and prediction.

## 13. Future Improvements

- Improve model reliability through cross-validation.
- Analyze feature importance.
- Tune model hyperparameters.
- Test performance on unseen restaurant data.
- Develop a simple web interface for rating predictions.
