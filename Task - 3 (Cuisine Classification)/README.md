# Cuisine Classification Using Machine Learning

## Project Overview
This project uses machine learning to predict a restaurant's primary cuisine based on restaurant-related information. It was developed as part of my Cognifyz internship tasks.

## Objective
To build a classification model that predicts one of the 15 most common cuisine categories using restaurant name, city, average cost, table booking availability, online delivery availability, price range, and votes.

## Dataset
- Dataset: `Dataset .csv`
- Records used: 8,023
- Cuisine classes: 15
- Training and testing split: 80% training, 20% testing

## Technologies Used
- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- LinearSVC
- Matplotlib
- Seaborn
- Joblib

## Methodology
1. Loaded and cleaned the restaurant dataset.
2. Extracted the primary cuisine from the cuisine column.
3. Selected the 15 most common cuisine categories.
4. Processed restaurant names using TF-IDF.
5. Encoded categorical features and handled missing numerical values.
6. Split the dataset into training and testing sets.
7. Trained and evaluated machine learning models.
8. Selected LinearSVC based on its evaluation results.

## Final Model Results

| Metric | Result |
|---|---:|
| Model | LinearSVC |
| Accuracy | 61.99% |
| Macro F1-score | 0.5680 |
| Test records | 1,605 |

## Output Files
- `main.py` — Main project code
- `Dataset .csv` — Dataset
- `cuisine_model.joblib` — Trained model pipeline
- `cuisine_predictions.csv` — Actual and predicted cuisine labels
- `confusion_matrix.png` — Confusion matrix visualization

## How to Run

Install the required libraries:

```bash
pip install pandas scikit-learn matplotlib seaborn joblib
```

Run the project:

```bash
python main.py
```

## Conclusion
The LinearSVC model achieved 61.99% accuracy and a macro F1-score of 0.5680 across 15 cuisine classes. The results demonstrate how machine learning and text features can be used for cuisine classification. Performance varies across cuisine categories, leaving opportunities for further improvement.

## Author
AIML Student
