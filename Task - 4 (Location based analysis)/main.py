
import pandas as pd
import folium
from folium.plugins import MarkerCluster, HeatMap
from pathlib import Path

# Project folder
BASE_DIR = Path(__file__).resolve().parent
CSV_FILE = BASE_DIR / "Dataset .csv"

# Load dataset
df = pd.read_csv(CSV_FILE, encoding="utf-8-sig")
df.columns = df.columns.str.strip()

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# Check required columns
required = [
    "Latitude", "Longitude", "City",
    "Aggregate rating", "Cuisines", "Price range"
]

missing = [col for col in required if col not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Clean coordinates and numerical columns
for col in ["Latitude", "Longitude", "Aggregate rating", "Price range"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Valid geographical coordinates
geo = df.dropna(subset=["Latitude", "Longitude"]).copy()
geo = geo[
    geo["Latitude"].between(-90, 90)
    & geo["Longitude"].between(-180, 180)
    & (geo["Latitude"] != 0)
    & (geo["Longitude"] != 0)
]

if geo.empty:
    raise ValueError("No valid restaurant coordinates found.")

print("\nTotal restaurants:", len(df))
print("Restaurants with valid coordinates:", len(geo))

# Create interactive map
center = [
    geo["Latitude"].median(),
    geo["Longitude"].median()
]

restaurant_map = folium.Map(
    location=center,
    zoom_start=3,
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
    attr="Tiles © Esri"
)
cluster = MarkerCluster().add_to(restaurant_map)

for _, row in geo.iterrows():
    name = row.get("Restaurant Name", "Restaurant")
    popup = (
        f"Restaurant: {name}<br>"
        f"City: {row['City']}<br>"
        f"Rating: {row['Aggregate rating']}<br>"
        f"Cuisines: {row['Cuisines']}"
    )

    folium.Marker(
        [row["Latitude"], row["Longitude"]],
        popup=folium.Popup(popup, max_width=300),
        tooltip=str(name)
    ).add_to(cluster)

restaurant_map.fit_bounds([
    [geo["Latitude"].min(), geo["Longitude"].min()],
    [geo["Latitude"].max(), geo["Longitude"].max()]
])

restaurant_map.save(BASE_DIR / "restaurant_location_map.html")

# Create geographical heatmap
heatmap = folium.Map(
    location=center,
    zoom_start=3,
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
    attr="Tiles © Esri"
)
HeatMap(
    geo[["Latitude", "Longitude"]].values.tolist(),
    radius=12,
    blur=15
).add_to(heatmap)

heatmap.fit_bounds([
    [geo["Latitude"].min(), geo["Longitude"].min()],
    [geo["Latitude"].max(), geo["Longitude"].max()]
])

heatmap.save(BASE_DIR / "restaurant_heatmap.html")

# City-wise statistics
city_stats = (
    df.dropna(subset=["City"])
    .groupby("City")
    .agg(
        Restaurant_Count=("City", "size"),
        Average_Rating=("Aggregate rating", "mean"),
        Average_Price_Range=("Price range", "mean")
    )
    .sort_values("Restaurant_Count", ascending=False)
)

city_stats = city_stats.round(2)
city_stats.to_csv(BASE_DIR / "city_analysis.csv")

print("\nTop 10 cities by restaurant count:")
print(city_stats.head(10).to_string())

# Locality-wise statistics
if "Locality" in df.columns:
    locality_stats = (
        df.dropna(subset=["City", "Locality"])
        .groupby(["City", "Locality"])
        .agg(
            Restaurant_Count=("Locality", "size"),
            Average_Rating=("Aggregate rating", "mean"),
            Average_Price_Range=("Price range", "mean")
        )
        .sort_values("Restaurant_Count", ascending=False)
    )

    locality_stats.to_csv(BASE_DIR / "locality_analysis.csv")

# Most common cuisines
cuisines = (
    df["Cuisines"].dropna()
    .str.split(",").explode().str.strip()
)

top_cuisines = cuisines.value_counts().head(10)
top_cuisines.to_csv(
    BASE_DIR / "top_cuisines.csv",
    header=["Restaurant_Count"]
)

print("\nTop 10 cuisines:")
print(top_cuisines.to_string())

print("\nTask 4 analysis completed successfully!")
print("Open restaurant_location_map.html and restaurant_heatmap.html")
print("to view your maps.")
