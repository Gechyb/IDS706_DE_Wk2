"""
Author: Ogechukwu Ezenwa
Date: September 9, 2025
Course: IDS 706 - Data Engineering Systems
Assignment: Week 2 Mini-Assignment - Data Analysis and Exploring Machine Learning Algorithms

Purpose: This script analyzes heart disease dataset and identify whether a patient is likely to have heart disease based on diagnostic measurements.
Dataset source: https://www.kaggle.com/datasets/navjotkaushal/heart-disease-uci-dataset

Usage:
    make run
    python data_analysis.py

"""

# Primary packages
import pandas as pd
import numpy as np
from scipy.stats import zscore
import operator


# Visualization packages
import matplotlib.pyplot as plt
import seaborn as sns

# Encoding and Machine learning model packages
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
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

        if data.empty:
            print("Warning: Loaded DataFrame is empty.")
            return data

        print("Dataset loaded successfully!")
        print("First 5 rows:\n", data.head())
        print(f"Number of rows: {data.shape[0]}, Number of columns: {data.shape[1]}")
        print("Data type for each column:\n", data.dtypes)

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
    if data.empty:
        print("DataFrame is empty. Skipping cleaning.")
        return data

    print("Data Cleaning & Preprocessing...")

    # Check missing values
    check_missing_values(data)

    # Drop duplicates
    return drop_duplicates(data)


def drop_duplicates(data):
    before = data.shape[0]
    data = data.drop_duplicates()
    after = data.shape[0]
    if before == after:
        print("No duplicate rows found.")
    else:
        print(f"Removed {before - after} duplicate rows.")

    print("Data types:\n", data.dtypes)
    print("Summary statistics:\n", data.describe())
    return data


def check_missing_values(data):
    missing_values = data.isnull().sum()
    total_missing = missing_values.sum()
    if total_missing > 0:
        print(
            "Missing values detected per column: \n", missing_values[missing_values > 0]
        )
    else:
        print("No missing values found")


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

    data = clean_dataset(data)

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
    operators = {
        ">": operator.gt,
        "<": operator.lt,
        ">=": operator.ge,
        "<=": operator.le,
        "==": operator.eq,
        "!=": operator.ne,
    }

    filtered_data = data.copy()  # to avoid modifying original dataset
    for col, (operator_str, value) in conditions.items():
        if col in filtered_data.columns:
            if operator_str in operators:
                filtered_data = filtered_data[
                    operators[operator_str](filtered_data[col], value)
                ]
            else:
                print(
                    f"Operator '{operator_str}' not supported. Skipping filter on '{col}'."
                )
        else:
            print(f"Column '{col}' not found in DataFrame. Skipping filter.")

    print(f"Filtered data based on conditions: {conditions}")
    print(filtered_data.head())
    print(filtered_data.shape)
    return filtered_data


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
    labelencoder = LabelEncoder()

    for col in categorical_cols:
        if col in data.columns:
            data[col] = labelencoder.fit_transform(
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


def run_model(
    data: pd.DataFrame,
    target_col: str = "num",
    categorical_cols: list = None,
    encoding: str = "onehot",
    model_type: str = "linear",  # "linear" or "random_forest"
    n_estimators: int = 100,
    max_depth: int = None,
    random_state: int = 42,
):
    """
    Train and evaluate either Linear Regression or Random Forest Regression.

    Parameters:
        data (pd.DataFrame): Dataset including features + target
        target_col (str): Column to predict (numeric)
        categorical_cols (list): List of categorical columns to encode
        encoding (str): "label" or "onehot" encoding
        model_type (str): "linear" or "random_forest"
        n_estimators (int): Number of trees (for random forest only)
        max_depth (int): Max tree depth (for random forest only)
        random_state (int): Random seed for reproducibility

    Returns:
        model: Trained model
    """
    print(f"Running {model_type.title()} Regression to predict '{target_col}'...")

    # Handle categorical variables
    if categorical_cols:
        if encoding == "label":
            data = apply_label_encoding(data, categorical_cols)
            print("Applied Label Encoding")
        elif encoding == "onehot":
            data = apply_one_hot_encoding(data, categorical_cols)
            print("Applied One-Hot Encoding")
        else:
            raise ValueError("encoding must be 'label' or 'onehot'")

    # Features and target
    X = data.drop(columns=[target_col])
    y = data[target_col]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    # Choose model
    if model_type == "linear":
        model = LinearRegression()
    elif model_type == "random_forest":
        model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
        )
    else:
        raise ValueError("model_type must be 'linear' or 'random_forest'")

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.2f}")

    return model


def plot_data(
    data: pd.DataFrame, x: str, y: str = None, plot_type: str = "hist", palette="Set3"
):
    """
    Create a visualization for the dataset.

    Parameters:
        data (pd.DataFrame): The dataset
        x (str): Column for the x-axis
        y (str, optional): Column for the y-axis (if needed)
        plot_type (str): Type of plot ("hist", "box", "scatter")
        palette (str): Can input any palette of choice from seaborn

    Returns:
        None
    """
    plt.figure(figsize=(8, 5))

    if plot_type == "hist":
        sns.histplot(data=data, x=x, bins=20, kde=True, color=palette)
        plt.title(f"Histogram of {x}")
        plt.xlabel(x)
        plt.ylabel("Count")

    elif plot_type == "box" and y is not None:
        sns.boxplot(data=data, x=x, y=y, hue=x, palette=palette, legend=False)
        plt.title(f"Boxplot of {y} by {x}")
        plt.xlabel(x)
        plt.ylabel(y)

    elif plot_type == "scatter" and y is not None:
        sns.scatterplot(data=data, x=x, y=y, hue=y, palette=palette)
        plt.title(f"Scatter Plot of {y} vs {x}")
        plt.xlabel(x)
        plt.ylabel(y)

    else:
        raise ValueError("Invalid plot_type or missing y for box/scatter plot")

    plt.show()
