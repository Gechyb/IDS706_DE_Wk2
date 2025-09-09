"""
Author: Ogechukwu Ezenwa
Data: September 9, 2025
Course: IDS 706 - Data Engineering Systems
Assignment: Week 2 Mini-Assignment - Data Analysis and Exploring Machine Learning Algorithms

Purpose: This script analyzes heart disease dataset and identify whether a patient is likely to have heart disease based on diagnostic measurements.
Dataset source: https://www.kaggle.com/datasets/navjotkaushal/heart-disease-uci-dataset

Usage:
    python data_analysis.py

"""

# Primary packages
import pandas as pd
import numpy as np
from scipy.stats import zscore
import operator

# Encoding Categorical variables
from sklearn.preprocessing import LabelEncoder

# Visualization packages
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning model development packages
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load the Heart Disease dataset.

    Parameters:
        file_path (str): Path to the CSV dataset file.

    Returns:
        pd.DataFrame: Loaded dataset as a pandas DataFrame.
    """
    try:
        data = pd.read_csv(file_path)
        print("Dataset loaded successfully!")
        print("First 5 rows:\n", data.head())
        print(
            f"Number of rows: {data.shape[0]}, Number of columns: {data.shape[1]}"
        )  # Show number of rows and columns
        print("Data type for each column:\n", data.dtypes)  # Show data types

        return data
    except FileNotFoundError:
        print("File not found. Please check the file path.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return pd.DataFrame()


def clean_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the dataset:
    - Check for missing values
    - Drop duplicates
    - Ensure correct data types

    Parameters:
        data (pd.DataFrame): Raw dataset

    Returns:
        pd.DataFrame: Cleaned dataset
    """
    print("Data Cleaning & Preprocessing...")

    # Check missing values
    missing_values = data.isnull().sum()
    total_missing = missing_values.sum()
    if total_missing > 0:
        print(
            "Missing values detected per column: \n", missing_values[missing_values > 0]
        )
    else:
        print("No missing values found")

    # Drop duplicates
    before = data.shape[0]
    data = data.drop_duplicates()
    after = data.shape[0]
    if before == after:
        print("No duplicate rows found.")
    else:
        print(f"Removed {before - after} duplicate rows.")

    # Show data types
    print("Data types:\n", data.dtypes)

    # Summary statistics
    print("Summary statistics:\n", data.describe())

    return data


def remove_outliers(
    data: pd.DataFrame, threshold: float = 3.0, columns: list = None
) -> pd.DataFrame:
    """
    Remove outliers from numeric columns using the z-score method.

    Parameters:
        data (pd.DataFrame): Input dataset
        threshold (float): z-score threshold for defining outliers (default=3.0)
        columns (list, optional): List of numeric columns to check for outliers.
                                  If None, all numeric columns are used.

    Returns:
        pd.DataFrame: Dataset with outliers removed
    """
    print("Outlier Detection & Removal...")

    # Determine which numeric columns to use
    if columns is None:
        numeric_cols = data.select_dtypes(include=[np.number])
    else:
        # Ensure only numeric columns from the list are used
        numeric_cols = data[columns].select_dtypes(include=[np.number])

    # Calculate z-scores
    z_scores = numeric_cols.apply(zscore)

    # Flag rows with outliers in ANY numeric column
    outliers = (z_scores.abs() > threshold).any(axis=1)

    before = data.shape[0]
    data = data[~outliers]
    after = data.shape[0]

    if before == after:
        print("No outliers removed (all rows within threshold).")
    else:
        print(f"Removed {before - after} outlier rows.")

    print("Summary of cleaned data:\n", data.describe())
    return data


def filter_data(data: pd.DataFrame, conditions: dict) -> pd.DataFrame:
    """
    Filter rows based on multiple conditions with flexible operators.

    Parameters:
        data (pd.DataFrame): Input DataFrame
        conditions (dict): Dictionary where keys are column names and values are tuples of (operator, value).
                           Supported operators: '>', '<', '>=', '<=', '==', '!='
                           Example: {'age': ('>', 50), 'chol': ('>=', 240)}

    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    ops = {
        ">": operator.gt,
        "<": operator.lt,
        ">=": operator.ge,
        "<=": operator.le,
        "==": operator.eq,
        "!=": operator.ne,
    }

    filtered = data.copy()  # to avoid modifying original dataset
    for col, (op_str, value) in conditions.items():
        if col in filtered.columns:
            if op_str in ops:
                filtered = filtered[ops[op_str](filtered[col], value)]
            else:
                print(f"Operator '{op_str}' not supported. Skipping filter on '{col}'.")
        else:
            print(f"Column '{col}' not found in DataFrame. Skipping filter.")

    print(f"Filtered data based on conditions: {conditions}")
    print(filtered.head())
    print(filtered.shape)
    return filtered


def group_and_summarize(data: pd.DataFrame, group_cols: list, agg_dict: dict):
    """
    Perform flexible groupby operations and summarize data.

    Parameters:
        data (pd.DataFrame): Input DataFrame
        group_cols (list): List of column names to group by
        agg_dict (dict): Dictionary of {column: [aggregations]} to apply.
                         Example: {"chol": ["mean"], "age": ["mean", "max"]}

    Returns:
        pd.DataFrame: Grouped summary DataFrame
    """
    if not all(col in data.columns for col in group_cols):
        missing = [col for col in group_cols if col not in data.columns]
        print(f"Missing group columns: {missing}. Skipping them.")

    if not all(col in data.columns for col in agg_dict.keys()):
        missing = [col for col in agg_dict.keys() if col not in data.columns]
        print(f"Missing aggregation columns: {missing}. Skipping them.")

    grouped = data.groupby([col for col in group_cols if col in data.columns]).agg(
        {col: funcs for col, funcs in agg_dict.items() if col in data.columns}
    )

    print(f"Grouped by {group_cols} with aggregations {agg_dict}:")
    print(grouped.head())
    return grouped


def apply_label_encoding(data: pd.DataFrame, categorical_cols: list) -> pd.DataFrame:
    """
    Apply Label Encoding to categorical columns (good for binary or ordinal categories).

    Parameters:
        data (pd.DataFrame): The dataset
        categorical_cols (list): List of column names to encode

    Returns:
        pd.DataFrame: Dataset with encoded categorical columns
    """
    data = data.copy()
    le = LabelEncoder()

    for col in categorical_cols:
        if col in data.columns:
            data[col] = le.fit_transform(
                data[col].astype(str)
            )  # convert to string to be safe

    return data


def apply_one_hot_encoding(
    data: pd.DataFrame, categorical_cols: list, drop_first: bool = True
) -> pd.DataFrame:
    """
    Apply One-Hot Encoding to categorical columns (good for nominal categories).

    Parameters:
        data (pd.DataFrame): The dataset
        categorical_cols (list): List of column names to encode
        drop_first (bool): Whether to drop the first category to avoid multicollinearity

    Returns:
        pd.DataFrame: Dataset with one-hot encoded columns
    """
    data = data.copy()
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=drop_first)

    return data


# def run_linear_regression(data: pd.DataFrame, target_col: str = "num"):
#     """
#     Train and evaluate a simple Linear Regression model.

#     Parameters:
#         data (pd.DataFrame): Dataset including features + target
#         target_col (str): Column to predict (numeric)

#     Returns:
#         model: Trained Linear Regression model
#     """
#     print(f"Running Linear Regression to predict '{target_col}'...")

#     # Step 1: Features and target
#     X = data.drop(columns=[target_col])
#     y = data[target_col]

#     # Step 2: Train-test split
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )

#     # Step 3: Train model
#     model = LinearRegression()
#     model.fit(X_train, y_train)

#     # Step 4: Predictions
#     y_pred = model.predict(X_test)

#     # Step 5: Evaluation
#     mse = mean_squared_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)

#     print(f"Mean Squared Error: {mse:.2f}")
#     print(f"R² Score: {r2:.2f}")

#     return model


if __name__ == "__main__":
    file_path = "data/heart_disease_UCI_dataset.csv"
    heart_disease = load_dataset(file_path)

    if heart_disease.empty:
        print("Dataset is empty. Exiting script...")
    else:
        heart_disease = clean_dataset(heart_disease)
        heart_disease = remove_outliers(heart_disease)

        filters = {"age": (">", 50), "chol": (">=", 240)}
        filtered_data = filter_data(heart_disease, filters)

        # Average cholesterol by sex
        group_and_summarize(
            heart_disease, group_cols=["sex"], agg_dict={"chol": ["mean"]}
        )

        # Count of patients by chest pain type
        group_and_summarize(
            heart_disease, group_cols=["cp"], agg_dict={"num": ["count"]}
        )

        # Multiple aggregations
        group_and_summarize(
            heart_disease,
            group_cols=["sex", "cp"],
            agg_dict={"age": ["mean", "max"], "chol": ["mean", "max"]},
        )

        categorical_cols = ["sex", "cp", "fbs", "restecg", "exang"]
        # Label Encoding
        label_encoded_data = apply_label_encoding(heart_disease, categorical_cols)
        print("Label encoding:\n", label_encoded_data.head())

        # One-Hot Encoding
        one_hot_encoded_data = apply_one_hot_encoding(heart_disease, categorical_cols)
        print("One hot encoded data:\n", one_hot_encoded_data.head())
