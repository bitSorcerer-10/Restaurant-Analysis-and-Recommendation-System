
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("Dataset .csv")

# Select useful columns
df = df[[
    "Restaurant Name",
    "City",
    "Cuisines",
    "Price range",
    "Aggregate rating"
]].copy()

# Handle missing values
df = df.dropna(subset=["Restaurant Name", "Cuisines"])
df["City"] = df["City"].fillna("")
df["Price range"] = df["Price range"].fillna(0)
df["Aggregate rating"] = df["Aggregate rating"].fillna(0)

# Remove duplicate restaurants
df = df.sort_values(
    "Aggregate rating", ascending=False
).drop_duplicates(
    subset=["Restaurant Name", "City"]
).reset_index(drop=True)

# Convert cuisine text into numerical features
tfidf = TfidfVectorizer()
cuisine_matrix = tfidf.fit_transform(df["Cuisines"])

# Calculate restaurant similarity
similarity = cosine_similarity(cuisine_matrix)

# Recommendation function
def recommend_restaurants(city, cuisine, price, min_rating):
    matches = df[
        (df["City"].str.contains(city, case=False, na=False)) &
        (df["Cuisines"].str.contains(cuisine, case=False, na=False)) &
        (df["Price range"] <= price) &
        (df["Aggregate rating"] >= min_rating)
    ].copy()

    if matches.empty:
        print("No restaurants match your preferences.")
        return matches

    # Rank matching restaurants by cuisine similarity
    cuisine_vector = tfidf.transform([cuisine])
    scores = cosine_similarity(
        cuisine_vector, cuisine_matrix
    )[0]

    matches["Similarity"] = matches.index.map(
        lambda i: scores[i]
    )

    matches = matches.sort_values(
        ["Similarity", "Aggregate rating"],
        ascending=False
    ).head(10)

    return matches

# Sample preferences
result = recommend_restaurants(
    city="New Delhi",
    cuisine="Chinese",
    price=2,
    min_rating=4.0
)

print("\nRECOMMENDED RESTAURANTS")
print(result.to_string(index=False))

result.to_csv("recommendations.csv", index=False)
print("\nSaved to recommendations.csv")
