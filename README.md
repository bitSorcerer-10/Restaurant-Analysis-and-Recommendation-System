# Restaurant Data Analysis and Machine Learning

## Project Overview

This project focuses on analyzing restaurant data and applying Machine Learning techniques to solve different problems related to the restaurant industry.

Using a restaurant dataset, the project explores restaurant ratings, generates personalized restaurant recommendations, classifies cuisines and performs exploratory data analysis to identify useful patterns and insights.

The project is divided into four major tasks.

## Project Structure

```text
Cognifyz/
│
├── Task - 1 (Predict Restaurant Ratings)/
├── Task - 2 (Restaurant Recommendations)/
├── Task - 3 (Cuisine Classification)/
├── Task - 4 (Restaurant Data Analysis)/
│
└── README.md
```

## Tasks and Implementation

### 1. Restaurant Rating Prediction

**Objective:** Predict the aggregate rating of a restaurant based on its features.

**Implementation:**
- Cleaned and preprocessed the restaurant dataset.
- Selected relevant features such as city, cuisine, average cost, price range, votes and service availability.
- Applied Linear Regression and Random Forest Regressor.
- Compared the models using MAE, RMSE and R² score.

**Result:** Random Forest Regressor performed better than Linear Regression, achieving an R² score of approximately 0.955.

### 2. Restaurant Recommendation System

**Objective:** Recommend restaurants based on similarities between restaurant features.

**Implementation:**
- Preprocessed restaurant information.
- Used TF-IDF to convert textual features into numerical representations.
- Applied Cosine Similarity to calculate similarities between restaurants.
- Generated recommendations based on features such as cuisine, city and price range.

**Result:** The system provides a list of restaurants that are similar to the selected restaurant.

### 3. Cuisine Classification

**Objective:** Predict the cuisine category of a restaurant using Machine Learning.

**Implementation:**
- Cleaned and prepared the dataset.
- Processed restaurant-related features.
- Applied a classification algorithm.
- Evaluated the model using classification metrics.

**Result:** Developed a classification model to predict restaurant cuisine categories based on input features.

### 4. Restaurant Data Analysis

**Objective:** Explore restaurant data and identify meaningful patterns and trends.

**Implementation:**
- Performed Exploratory Data Analysis (EDA).
- Analyzed restaurant distribution across countries and cities.
- Studied popular cuisines and restaurant ratings.
- Explored price ranges, votes and other restaurant attributes.
- Created visualizations to understand the dataset.

**Result:** Identified useful insights about restaurant distributions, cuisines, ratings and pricing patterns.

## Dataset Description

The project uses a restaurant dataset containing information about restaurants across different locations.

Important attributes include:

- Restaurant Name
- Country Code
- City
- Address
- Cuisines
- Average Cost for Two
- Has Table Booking
- Has Online Delivery
- Price Range
- Aggregate Rating
- Votes

## Technologies and Libraries Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine Learning algorithms |
| Joblib | Saving trained models |
| Git & GitHub | Version control |

## Machine Learning Techniques

- Linear Regression
- Random Forest Regressor
- TF-IDF Vectorization
- Cosine Similarity
- Classification Algorithms
- Model Evaluation

## Project Outcomes

Through this project, different Machine Learning and Data Analysis techniques are applied to restaurant-related problems.

The project demonstrates how restaurant data can be used to:
- Predict restaurant ratings.
- Recommend similar restaurants.
- Classify restaurant cuisines.
- Discover patterns and insights through data analysis.

## Conclusion

This project demonstrates the practical application of Python, Data Analysis and Machine Learning in the restaurant domain. It combines predictive modelling, recommendation systems, classification and exploratory data analysis to extract useful information from restaurant data.
