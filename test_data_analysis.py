"""
Author: Ogechukwu Ezenwa
Date: September 13, 2025
Course: IDS 706 - Data Engineering Systems
Assignment: Week 3 Mini-Assignment - Testing and Environment Setup

Purpose: This script tests the data analysis functions to ensure the project is reproducible and its results are verifiable.

Usage:
    make test
    python -m pytest -vv --cov=data_analysis test_data_analysis.py

"""

from data_analysis import (
    load_dataset,
    clean_dataset,
    remove_outliers,
    filter_data,
    group_and_summarize,
    apply_label_encoding,
    apply_one_hot_encoding,
    run_linear_regression,
    plot_data
)
import unittest
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


class TestDataAnalysis(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up a sample dataset to use across tests"""
        cls.sample_data = pd.DataFrame({
            "age": [45, 60, 30, 50],
            "chol": [200, 250, 180, 300],
            "sex": ["M", "F", "M", "F"],
            "cp": ["typical", "asymptomatic", "non-anginal", "typical"],
            "num": [0, 1, 0, 1]
        })

    def test_load_dataset_success(self):
        """Test loading an existing CSV file"""
        df = load_dataset("data/heart_disease_UCI_dataset.csv")
        self.assertFalse(df.empty)
        self.assertIn("age", df.columns)

    def test_clean_dataset(self):
        """Test cleaning removes duplicates and checks missing values"""
        data = self.sample_data.copy()
        cleaned = clean_dataset(data)
        # no duplicates in sample
        self.assertEqual(cleaned.shape[0], len(data))
        self.assertFalse(cleaned.isnull().any().any())

    def test_remove_outliers(self):
        """Test outlier removal"""
        data = self.sample_data.copy()
        cleaned = remove_outliers(data, threshold=2)
        self.assertTrue(cleaned.shape[0] <= data.shape[0])

    def test_filter_data(self):
        """Test filtering functionality"""
        data = self.sample_data.copy()
        conditions = {"age": (">", 40)}
        filtered = filter_data(data, conditions)
        self.assertTrue((filtered["age"] > 40).all())

    def test_group_and_summarize(self):
        """Test groupby and aggregation"""
        data = self.sample_data.copy()
        grouped = group_and_summarize(
            data, group_cols=["sex"], agg_dict={"age": ["mean"]})
        self.assertIn("age", grouped.columns.get_level_values(0))

    def test_apply_label_encoding(self):
        """Test label encoding works"""
        data = self.sample_data.copy()
        encoded = apply_label_encoding(data, ["sex", "cp"])
        self.assertTrue(encoded["sex"].dtype == int)
        self.assertTrue(encoded["cp"].dtype == int)

    def test_apply_one_hot_encoding(self):
        """Test one-hot encoding works"""
        data = self.sample_data.copy()
        encoded = apply_one_hot_encoding(data, ["sex", "cp"])
        self.assertTrue(all(col.startswith("sex_") or col.startswith(
            "cp_") or col in ["age", "chol", "num"] for col in encoded.columns))

    def test_run_linear_regression(self):
        """Test linear regression runs without error"""
        data = self.sample_data.copy()
        model = run_linear_regression(data, target_col="num", categorical_cols=[
                                      "sex", "cp"], encoding="onehot")
        self.assertIsNotNone(model)

    def test_plot_data(self):
        """Smoke test for plotting function (checks it runs without error)"""
        data = self.sample_data.copy()
        try:
            # Histogram
            plot_data(data, x="age", plot_type="hist", palette="skyblue")
            # Boxplot
            plot_data(data, x="num", y="chol", plot_type="box")
            # Scatter
            plot_data(data, x="age", y="chol",
                      plot_type="scatter", palette="icefire")
        except Exception as e:
            self.fail(f"plot_data raised an exception: {e}")


# Creating Systems Test for data analysis project
class TestDataAnalysisSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Set up a small sample dataset to run system tests.
        """
        cls.sample_data = pd.DataFrame({
            "age": [45, 55, 60, 35],
            "sex": ["M", "F", "M", "F"],
            "cp": [1, 2, 3, 4],
            "chol": [200, 250, 180, 300],
            "fbs": [0, 1, 0, 1],
            "restecg": [0, 1, 1, 0],
            "exang": [0, 1, 0, 1],
            "num": [0, 1, 1, 0],
            "thalch": [150, 160, 170, 140],
            "trestbps": [120, 140, 130, 125],
            "oldpeak": [2.3, 3.1, 1.0, 0.5]
        })

    def test_full_pipeline(self):
        """
        Test the entire data analysis pipeline on sample data.
        """
        # Step 1: Clean dataset
        cleaned = clean_dataset(self.sample_data)
        self.assertFalse(cleaned.isnull().any().any(),
                         "Dataset should have no missing values")

        # Step 2: Remove outliers
        no_outliers = remove_outliers(cleaned, threshold=3)
        self.assertLessEqual(no_outliers.shape[0], cleaned.shape[0])

        # Step 3: Filter data
        filtered = filter_data(no_outliers, {"age": (">", 40)})
        self.assertTrue((filtered["age"] > 40).all())

        # Step 4: Group and summarize
        grouped = group_and_summarize(
            filtered, group_cols=["sex"], agg_dict={"chol": ["mean"]})
        self.assertIn("chol", grouped.columns.get_level_values(0))

        # Step 5: Encode categorical data
        label_encoded = apply_label_encoding(
            filtered, ["sex", "cp", "fbs", "restecg", "exang"])
        onehot_encoded = apply_one_hot_encoding(
            filtered, ["sex", "cp", "fbs", "restecg", "exang"])
        self.assertIsInstance(label_encoded, pd.DataFrame)
        self.assertIsInstance(onehot_encoded, pd.DataFrame)

        # Step 6: Run ML model
        model = run_linear_regression(label_encoded, target_col="num", categorical_cols=[
                                      "sex", "cp", "fbs", "restecg", "exang"], encoding="label")
        self.assertIsNotNone(model)

        # Step 7: Test plot function (basic smoke test)
        try:
            plot_data(filtered, x="age", plot_type="hist", palette="skyblue")
            plot_data(filtered, x="num", y="chol", plot_type="box")
            plot_data(filtered, x="age", y="chol",
                      plot_type="scatter", palette="coolwarm")
        except Exception as e:
            self.fail(f"Plotting failed: {e}")

    def test_edge_cases(self):
        """Test edge cases such as empty data, single row, or missing columns"""
        # Empty dataset
        empty_df = pd.DataFrame()
        cleaned_empty = clean_dataset(empty_df)
        self.assertTrue(cleaned_empty.empty)

        # Single row dataset
        single_row = self.sample_data.iloc[[0]]
        cleaned_single = clean_dataset(single_row)
        self.assertEqual(cleaned_single.shape[0], 1)

        # Dataset with missing columns for filtering
        missing_col_df = self.sample_data.drop(columns=["age"])
        filtered = filter_data(missing_col_df, {"age": (">", 40)})
        # filtering should skip missing column
        self.assertTrue(filtered.equals(missing_col_df))

        # Dataset with identical values
        identical_df = pd.DataFrame({
            "age": [50]*4,
            "sex": ["M"]*4,
            "cp": [1]*4,
            "chol": [200]*4,
            "num": [0]*4
        })
        no_outliers = remove_outliers(identical_df, threshold=3)
        self.assertEqual(no_outliers.shape[0], 4)  # nothing removed


if __name__ == "__main__":
    unittest.main()
