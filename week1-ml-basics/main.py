"""Week 1 ML Basics: house-price prediction with simple linear regression.

Run from this folder with: python main.py
The included housing dataset is synthetic and for educational practice only.
"""

from pathlib import Path
import math

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data" / "housing.csv"
REQUIRED_COLUMNS = {"area_sqft", "price_lakhs"}


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Load the CSV and display the first 10 records (Practice Question 1)."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}\n"
            "Make sure data/housing.csv exists inside the project folder."
        )

    df = pd.read_csv(file_path)
    print("\n1) FIRST 10 ROWS OF THE DATASET")
    print(df.head(10).to_string(index=False))
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names, remove duplicates, convert columns and handle missing data."""
    cleaned = df.copy()
    cleaned.columns = cleaned.columns.str.strip().str.lower()

    missing_columns = REQUIRED_COLUMNS - set(cleaned.columns)
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {sorted(missing_columns)}")

    for column in REQUIRED_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned = cleaned.drop_duplicates()
    cleaned = cleaned.dropna(subset=["price_lakhs"])

    if cleaned["area_sqft"].isna().any():
        median_area = cleaned["area_sqft"].median()
        if pd.isna(median_area):
            raise ValueError("The dataset has no valid area_sqft values.")
        cleaned["area_sqft"] = cleaned["area_sqft"].fillna(median_area)

    cleaned = cleaned[
        (cleaned["area_sqft"] > 0) & (cleaned["price_lakhs"] > 0)
    ].copy()

    if len(cleaned) < 5:
        raise ValueError("At least 5 valid rows are required to train and test the model.")

    print("\n2) DATA PREPROCESSING SUMMARY")
    print(f"Rows after cleaning: {len(cleaned)}")
    print(f"Missing values after cleaning:\n{cleaned[list(REQUIRED_COLUMNS)].isna().sum()}")
    return cleaned


def train_and_evaluate(df: pd.DataFrame):
    """Split data, train Linear Regression, and evaluate test-set predictions."""
    X = df[["area_sqft"]]
    y = df["price_lakhs"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    print("\n3) TRAIN-TEST SPLIT")
    print(f"Training records: {len(X_train)}")
    print(f"Testing records:  {len(X_test)}")

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = math.sqrt(mean_squared_error(y_test, predictions))

    print("\n4) LINEAR REGRESSION MODEL EVALUATION")
    print(f"R-squared (R²): {r2:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f} lakhs")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} lakhs")
    print(
        f"Model equation: predicted price = {model.intercept_:.2f} + "
        f"({model.coef_[0]:.4f} × area in sq. ft.)"
    )

    print("\nActual vs predicted prices on the test set:")
    comparison = pd.DataFrame(
        {
            "area_sqft": X_test["area_sqft"].to_numpy(),
            "actual_price_lakhs": y_test.to_numpy(),
            "predicted_price_lakhs": predictions,
        }
    ).sort_values("area_sqft")
    print(comparison.round(2).to_string(index=False))

    return model


def predict_house_price(model: LinearRegression) -> None:
    """Ask for an area and predict its estimated price (Practice Question 4)."""
    print("\n5) PREDICT A HOUSE PRICE")
    while True:
        raw_area = input("Enter house area in square feet (or type q to quit): ").strip()
        if raw_area.lower() in {"q", "quit", "exit"}:
            print("Prediction session ended.")
            return

        try:
            area = float(raw_area)
            if not math.isfinite(area) or area <= 0:
                raise ValueError
        except ValueError:
            print("Please enter a valid positive number, such as 1200, or type q to quit.")
            continue

        estimate = model.predict(pd.DataFrame({"area_sqft": [area]}))[0]
        print(f"Estimated house price for {area:,.0f} sq. ft.: ₹{estimate:.2f} lakhs")
        return


def main() -> None:
    print("=" * 62)
    print("WEEK 1 - MACHINE LEARNING BASICS")
    print("House Price Prediction using Linear Regression")
    print("=" * 62)

    try:
        raw_df = load_dataset(DATA_FILE)
        clean_df = preprocess_data(raw_df)
        model = train_and_evaluate(clean_df)
        predict_house_price(model)
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        print(f"\nError: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
