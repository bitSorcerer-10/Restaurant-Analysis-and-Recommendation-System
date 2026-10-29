# Task 4: Location-Based Restaurant Analysis

## Project Overview

This project performs location-based analysis of restaurants using geographical coordinates and restaurant-related information. It identifies restaurant distribution patterns, highlights areas with high restaurant concentrations, and analyzes restaurant ratings, price ranges, and cuisine popularity across different cities and localities.

The project was completed as part of the Cognifyz Technologies internship tasks.

## Objectives

- Visualize restaurant locations using latitude and longitude.
- Identify areas with a high concentration of restaurants using a heatmap.
- Analyze restaurant distribution across different cities and localities.
- Compare average restaurant ratings and price ranges.
- Identify the most frequently listed cuisines.

## Technologies Used

- **Python** — programming language
- **Pandas** — data cleaning and analysis
- **Folium** — interactive map visualization
- **MarkerCluster** — grouping nearby restaurant markers
- **HeatMap** — visualizing restaurant concentration

## Dataset

The project uses `Dataset .csv`, which contains restaurant information, including:

- Restaurant Name
- City
- Locality
- Latitude and Longitude
- Aggregate Rating
- Cuisines
- Price Range

## Methodology

1. **Data loading:** Imported the restaurant dataset using Pandas.
2. **Data cleaning:** Processed geographical coordinates and removed invalid location data where necessary.
3. **Location mapping:** Created an interactive map displaying restaurant locations with clustered markers.
4. **Heatmap generation:** Visualized areas with a high concentration of restaurants.
5. **City-wise analysis:** Calculated restaurant counts, average ratings, and average price-range values for each city.
6. **Locality-wise analysis:** Analyzed restaurant distribution and ratings across localities.
7. **Cuisine analysis:** Identified the most frequently listed cuisines and exported the results to CSV files.

## Key Findings

- New Delhi had the highest number of restaurants in the analyzed dataset, with 5,473 records.
- Gurgaon had 1,118 restaurant records, while Noida had 1,080.
- Gurgaon had an average aggregate rating of approximately 2.65, compared with 2.44 for New Delhi and 2.04 for Noida.
- North Indian cuisine was the most frequently listed cuisine, followed by Chinese and Fast Food.
- The heatmap and clustered map help identify restaurant hotspots and understand geographical distribution patterns.

*Note: Cuisine counts represent cuisine listings, not necessarily unique restaurants. City averages and ratings reflect the available dataset and should be interpreted in that context.*

## Output Files

| File | Description |
|---|---|
| `restaurant_location_map.html` | Interactive map showing restaurant locations |
| `restaurant_heatmap.html` | Heatmap showing restaurant concentration |
| `city_analysis.csv` | City-wise restaurant statistics |
| `locality_analysis.csv` | Locality-wise restaurant statistics |
| `top_cuisines.csv` | Most frequently listed cuisines |

## How to Run the Project

### 1. Install the required libraries

```bash
pip install pandas folium
```

### 2. Open the project folder

Ensure that `main.py` and `Dataset .csv` are in the same folder.

### 3. Run the program

```bash
py main.py
```

### 4. View the results

Open `restaurant_location_map.html` and `restaurant_heatmap.html` in a web browser. Open the CSV files in Excel or another spreadsheet application.

**Note:** The interactive maps require an internet connection to load the external map tiles.

## Conclusion

This project demonstrates how Python and geospatial visualization can be used to analyze restaurant locations, identify high-density areas, compare city-wise restaurant statistics, and explore cuisine popularity. The generated interactive maps and CSV reports provide useful insights into the geographical distribution of restaurants.

## Author

Completed as part of the Cognifyz Technologies internship project.
