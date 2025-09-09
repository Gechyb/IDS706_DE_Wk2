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

import pandas as pd
import numpy as np
from scipy.stats import zscore
import operator
import matplotlib.pyplot as plt
import seaborn as sns


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

    filtered = data.copy()
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
