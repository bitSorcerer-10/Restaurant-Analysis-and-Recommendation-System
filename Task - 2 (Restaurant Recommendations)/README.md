# Restaurant Recommendation System

## Project Overview
This project recommends restaurants based on user preferences such as city, cuisine, price range, and minimum rating. It uses Python, Pandas, TF-IDF, and cosine similarity.

## Objective
To develop a content-based restaurant recommendation system that helps users find restaurants matching their preferences.

## Technologies Used
- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- Cosine Similarity

## Dataset
The project uses a restaurant dataset containing restaurant names, cities, cuisines, price ranges, and aggregate ratings.

## Methodology
1. Load the restaurant dataset.
2. Select the required columns.
3. Handle missing values.
4. Remove duplicate restaurant entries.
5. Convert cuisine descriptions into numerical features using TF-IDF.
6. Calculate cosine similarity between the cuisine query and restaurant descriptions.
7. Filter restaurants by city, price range, and minimum rating.
8. Rank the matching restaurants by similarity.
9. Save the recommendations to `recommendations.csv`.

## Sample Test
- City: New Delhi
- Cuisine: Chinese
- Maximum Price Range: 2
- Minimum Rating: 4.0

## Output
The system generates up to 10 restaurant recommendations and saves the results in `recommendations.csv`.

## Conclusion
The project demonstrates a basic content-based restaurant recommendation system. It uses cuisine similarity and user-defined constraints to provide relevant restaurant suggestions.
