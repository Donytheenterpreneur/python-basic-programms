# Week 1 - Machine Learning Basics

**Project:** House Price Prediction using Linear Regression  
**Topics:** Supervised vs. Unsupervised Learning, Linear Regression, and Data Preprocessing

This beginner-friendly project demonstrates the four Week 1 practice questions using Python, pandas, and scikit-learn.

> **Dataset note:** `data/housing.csv` is a small, synthetic educational dataset. Its values are illustrative and must not be treated as actual property-market prices or a reliable real-estate valuation tool.

## Learning objectives

### 1. Supervised vs. Unsupervised Learning

| Supervised learning | Unsupervised learning |
|---|---|
| Learns from labelled examples, where the expected output is provided. | Learns from data without supplied target labels. |
| Used for regression and classification. | Commonly used for clustering and dimensionality reduction. |
| Example: predicting a house price from its area. | Example: grouping houses into clusters based on their features. |

This project uses **supervised learning** because the dataset contains both the input (`area_sqft`) and the known output (`price_lakhs`).

### 2. Linear Regression

Linear regression estimates the relationship between an input feature and a continuous target value. In this project, the input is house area and the target is house price.

The model learns a best-fit straight line from the training data and uses it to estimate prices for unseen areas.

### 3. Data Preprocessing

The program prepares the CSV before training by:
- standardizing column names;
- converting the area and price columns to numeric values;
- removing duplicate rows;
- removing rows without a target price;
- filling missing area values with the median;
- excluding non-positive area or price values.

## Practice questions covered

1. **Load a CSV using pandas and print first 10 rows.** `load_dataset()` uses `pd.read_csv()` and `head(10)`.
2. **Split dataset into train/test using sklearn.** `train_test_split()` keeps 80% for training and 20% for testing.
3. **Train a Linear Regression model and check accuracy.** `LinearRegression` is fitted on training data. Since this is a regression task, classification-style accuracy is not appropriate. The program reports R-squared (R²), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE) on the test set.
4. **Predict house price based on area input.** The program asks for an area in square feet and prints the estimated price in lakhs.

## Project structure

```text
week1-ml-basics/
├── data/
│   └── housing.csv
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Requirements

- Python 3.9 or newer
- pandas
- scikit-learn

## How to run

Open a terminal in the project folder and run:

### 1. Create and activate a virtual environment (recommended)

**Windows:**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Run the project

```bash
python main.py
```

When prompted, enter an area such as `1200`. The model will show an estimated price. Enter `q` to quit.

## Understanding the evaluation metrics

- **R²:** Indicates how much of the variation in test-set prices is explained by the model. A value closer to 1 generally indicates a better fit; it is not a percentage accuracy score.
- **MAE:** The average absolute prediction error, in lakhs.
- **RMSE:** The square root of the average squared prediction error, in lakhs. It gives larger errors more influence than MAE.

Metrics are calculated on the held-out test data, not the training data.

## Limitations

This is a simple one-feature demonstration. Real house prices also depend on location, age, construction quality, number of rooms, amenities, and market conditions. The synthetic dataset is small and cannot produce real-world valuation estimates.

## Submission

Prepared as a Week 1 Machine Learning Basics practice submission.
