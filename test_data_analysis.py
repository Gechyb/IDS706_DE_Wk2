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

import unittest
import pandas as pd
from data_analysis import (
    load_dataset,
    clean_dataset,
    remove_outliers,
    filter_data,
    group_and_summarize,
    apply_label_encoding,
    apply_one_hot_encoding,
    run_linear_regression
)


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


if __name__ == "__main__":
    unittest.main()
