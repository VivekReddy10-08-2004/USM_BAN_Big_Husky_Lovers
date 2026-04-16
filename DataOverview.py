import os
import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

warnings.filterwarnings("ignore")


def load_data(file_path):
    """Load dataset from CSV."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find file: {file_path}")
    return pd.read_csv(file_path)


def basic_report(df):
    """Print quick dataset overview."""
    print("\n" + "=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")

    print("Column names:")
    for col in df.columns:
        print(f"- {col}")

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    missing = df.isnull().sum().sort_values(ascending=False)
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("No missing values found.")

    print("\nPreview:")
    print(df.head())


def suggest_target_column(df):
    """Try to guess a likely target column."""
    common_targets = [
        "target", "label", "y", "outcome", "class",
        "sales", "revenue", "profit", "price", "churn",
        "default", "score", "demand", "survived"
    ]

    lower_map = {col.lower(): col for col in df.columns}

    for target_name in common_targets:
        if target_name in lower_map:
            return lower_map[target_name]

    return df.columns[-1]


def detect_problem_type(series):
    """Detect classification vs regression."""
    if series.dtype == "object" or str(series.dtype).startswith("category"):
        return "classification"

    unique_count = series.nunique(dropna=True)

    if unique_count <= 15:
        return "classification"

    return "regression"


def split_features_target(df, target_col):
    """Split dataframe into X and y."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def build_preprocessor(X):
    """Build preprocessing pipeline for numeric and categorical columns."""
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    return preprocessor, numeric_features, categorical_features


def build_model(problem_type):
    """Build a baseline model."""
    if problem_type == "classification":
        return RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )
    return RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )


def evaluate_model(problem_type, y_test, preds):
    """Print evaluation metrics."""
    print("\n" + "=" * 60)
    print("MODEL RESULTS")
    print("=" * 60)

    if problem_type == "classification":
        acc = accuracy_score(y_test, preds)
        print(f"Accuracy: {acc:.4f}\n")
        print("Classification Report:")
        print(classification_report(y_test, preds))
    else:
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        print(f"MAE:  {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2:   {r2:.4f}")


def save_missing_plot(df):
    """Save a missing-values bar chart."""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if len(missing) == 0:
        return

    plt.figure(figsize=(10, 5))
    missing.plot(kind="bar")
    plt.title("Missing Values by Column")
    plt.xlabel("Columns")
    plt.ylabel("Missing Count")
    plt.tight_layout()
    plt.savefig("missing_values.png")
    plt.close()


def save_numeric_histograms(df):
    """Save histograms for up to 8 numeric columns."""
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if len(numeric_cols) == 0:
        return

    for col in numeric_cols[:8]:
        plt.figure(figsize=(6, 4))
        df[col].dropna().hist(bins=30)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()

        safe_name = col.replace("/", "_").replace("\\", "_").replace(" ", "_")
        plt.savefig(f"hist_{safe_name}.png")
        plt.close()


def get_feature_names_from_preprocessor(preprocessor, numeric_features, categorical_features):
    """Get feature names after preprocessing."""
    feature_names = []
    feature_names.extend(numeric_features)

    if categorical_features:
        encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
        encoded_names = encoder.get_feature_names_out(categorical_features)
        feature_names.extend(encoded_names.tolist())

    return feature_names


def save_feature_importance(pipeline, preprocessor, numeric_features, categorical_features):
    """Save feature importance if the model supports it."""
    model = pipeline.named_steps["model"]

    if not hasattr(model, "feature_importances_"):
        return

    feature_names = get_feature_names_from_preprocessor(
        preprocessor,
        numeric_features,
        categorical_features
    )

    importances = model.feature_importances_

    if len(feature_names) != len(importances):
        print("\nCould not align feature names with importances.")
        return

    fi = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    fi.to_csv("feature_importance.csv", index=False)

    top_n = fi.head(15).sort_values("importance")
    plt.figure(figsize=(10, 6))
    plt.barh(top_n["feature"], top_n["importance"])
    plt.title("Top Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.close()

    print("\nTop 10 important features:")
    print(fi.head(10))


def main():
    if len(sys.argv) < 2:
        print("Usage: python DataOverview.py your_file.csv [target_column]")
        sys.exit(1)

    file_path = sys.argv[1]
    df = load_data(file_path)

    basic_report(df)
    save_missing_plot(df)
    save_numeric_histograms(df)

    if len(sys.argv) >= 3:
        target_col = sys.argv[2]
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset.")
    else:
        target_col = suggest_target_column(df)

    print("\n" + "=" * 60)
    print("TARGET SELECTION")
    print("=" * 60)
    print(f"Using target column: {target_col}")

    X, y = split_features_target(df, target_col)

    # Drop rows where target is missing
    valid_rows = y.notna()
    X = X[valid_rows]
    y = y[valid_rows]

    print(f"Missing target values removed: {(~valid_rows).sum()}")
    print(f"Remaining rows after target cleanup: {len(y)}")

    if len(y) == 0:
        raise ValueError("No rows remain after dropping missing target values.")

    problem_type = detect_problem_type(y)
    print(f"Detected problem type: {problem_type}")

    preprocessor, numeric_features, categorical_features = build_preprocessor(X)
    model = build_model(problem_type)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    stratify_value = None
    if problem_type == "classification":
        class_counts = y.value_counts()
        min_class_count = class_counts.min()

        if min_class_count >= 2:
            stratify_value = y
        else:
            print("\nWarning: At least one class has fewer than 2 rows, so stratify was skipped.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_value
    )

    print("\nTraining model...")
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    evaluate_model(problem_type, y_test, preds)

    fitted_preprocessor = pipeline.named_steps["preprocessor"]
    save_feature_importance(
        pipeline,
        fitted_preprocessor,
        numeric_features,
        categorical_features
    )

    print("\nFiles saved:")
    print("- missing_values.png")
    print("- feature_importance.csv")
    print("- feature_importance.png")
    print("- hist_*.png for numeric columns")

    print("\nDone.")


if __name__ == "__main__":
    main()