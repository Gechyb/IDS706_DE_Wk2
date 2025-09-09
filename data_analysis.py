"""
Author: Ogechukwu Ezenwa
Data: September 9, 2025
Course: IDS 706 - Data Engineering Systems
Assignment: Week 2 Mini-Assignment - Data Analysis and Exploring Machine Learning Algorithms


Question:
Purpose: This script analyzes heart disease dataset to identify the factors that results in heart disease
Dataset source: https://www.kaggle.com/datasets/navjotkaushal/heart-disease-uci-dataset
Usage:
    python data_analysis.py
"""

import pandas as pd
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd


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
        print("✅ Dataset loaded successfully!")
        print("First 5 rows:\n", data.head())
        print(
            f"Number of rows: {data.shape[0]}, Number of columns: {data.shape[1]}"
        )  # Show number of rows and columns
        print("\nData type for each column:\n", data.dtypes)  # Show data types

        return data
    except FileNotFoundError:
        print("File not found. Please check the file path.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    file_path = "data/heart_disease_UCI_dataset.txt"
    heart_disease = load_dataset(file_path)
