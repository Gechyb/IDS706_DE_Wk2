"""
Author: Ogechukwu Ezenwa
Date: September 29, 2025
Purpose: Usage script demonstrating data_analysis.py with heart disease dataset
"""

from data_analysis import (
    load_dataset,
    clean_dataset,
    remove_outliers,
    filter_data,
    group_and_summarize,
    apply_label_encoding,
    apply_one_hot_encoding,
    run_model,
    plot_data,
)


def main():
    # Step 1: Load the dataset
    file_path = "data/heart_disease_UCI_dataset.csv"
    heart_disease = load_dataset(file_path)

    if heart_disease.empty:
        print("Dataset is empty. Exiting script...")
        return

    # Step 2: Clean the dataset
    heart_disease = clean_dataset(heart_disease)

    # Step 3: Remove outliers
    numeric_columns = ["age", "trestbps", "chol", "thalch", "oldpeak"]
    heart_disease = remove_outliers(heart_disease, columns=numeric_columns)

    # Step 4: Optional filtering
    filters = {"age": (">", 50), "chol": (">=", 240)}
    filtered_data = filter_data(heart_disease, filters)

    # Step 5: Grouping and summarizing
    group_and_summarize(heart_disease, group_cols=["sex"], agg_dict={"chol": ["mean"]})
    group_and_summarize(heart_disease, group_cols=["cp"], agg_dict={"num": ["count"]})
    group_and_summarize(
        heart_disease,
        group_cols=["sex", "cp"],
        agg_dict={"age": ["mean", "max"], "chol": ["mean", "max"]},
    )

    # Step 6: Encode categorical variables
    categorical_cols = ["sex", "cp", "fbs", "restecg", "exang"]
    label_encoded_data = apply_label_encoding(heart_disease, categorical_cols)
    print("Label encoded data:\n", label_encoded_data.head())
    one_hot_encoded_data = apply_one_hot_encoding(heart_disease, categorical_cols)
    print("One-hot encoded data:\n", one_hot_encoded_data.head())

    # Step 7: Train models
    print("\n--- Running Linear Regression ---")
    linear_model = run_model(
        heart_disease,
        target_col="num",
        categorical_cols=categorical_cols,
        encoding="onehot",
        model_type="linear",
    )

    print("\n--- Running Random Forest Regression ---")
    rf_model = run_model(
        heart_disease,
        target_col="num",
        categorical_cols=categorical_cols,
        encoding="onehot",
        model_type="random_forest",
        n_estimators=200,
        max_depth=10,
    )

    # Step 8: Data Visualization
    plot_data(heart_disease, x="age", plot_type="hist", palette="y")
    plot_data(heart_disease, x="num", y="chol", plot_type="box")
    plot_data(heart_disease, x="age", y="chol", plot_type="scatter", palette="coolwarm")


if __name__ == "__main__":
    main()
