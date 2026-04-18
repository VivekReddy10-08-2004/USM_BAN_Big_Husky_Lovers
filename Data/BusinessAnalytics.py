import os
import re
import sys
import warnings
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def load_data(file_path):
    """Load a CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find file: {file_path}")
    return pd.read_csv(file_path)


def clean_column_names(df):
    """Strip spaces from column names."""
    df.columns = [col.strip() for col in df.columns]
    return df


def convert_possible_numeric(df):
    """
    Try converting object columns to numeric when it makes sense.
    Keeps original values if conversion would destroy too much data.
    """
    for col in df.columns:
        if df[col].dtype == "object":
            cleaned = df[col].astype(str).str.replace(",", "", regex=False).str.strip()
            numeric_version = pd.to_numeric(cleaned, errors="coerce")

            # convert only if a strong portion can become numeric
            success_rate = numeric_version.notna().mean()
            if success_rate >= 0.8:
                df[col] = numeric_version

    return df


def classify_columns(df):
    """Split columns into numeric, categorical, and text-like."""
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = []
    text_cols = []

    for col in df.select_dtypes(exclude=["number"]).columns:
        non_null = df[col].dropna().astype(str)
        avg_len = non_null.str.len().mean() if len(non_null) > 0 else 0
        unique_count = non_null.nunique()

        if avg_len > 40:
            text_cols.append(col)
        elif unique_count <= 50:
            categorical_cols.append(col)
        else:
            text_cols.append(col)

    return numeric_cols, categorical_cols, text_cols


def dataset_overview(df, numeric_cols, categorical_cols, text_cols):
    print("\n" + "=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")

    print("Columns:")
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

    print("\nColumn groups:")
    print(f"Numeric: {numeric_cols}")
    print(f"Categorical: {categorical_cols}")
    print(f"Text: {text_cols}")

    print("\nPreview:")
    print(df.head())


def save_missing_values_chart(df):
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if len(missing) == 0:
        return

    plt.figure(figsize=(10, 5))
    missing.plot(kind="bar")
    plt.title("Missing Values by Column")
    plt.xlabel("Column")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("missing_values.png")
    plt.close()


def analyze_numeric_columns(df, numeric_cols):
    if not numeric_cols:
        print("\nNo numeric columns found.")
        return

    print("\n" + "=" * 60)
    print("NUMERIC ANALYSIS")
    print("=" * 60)
    print(df[numeric_cols].describe().T)

    for col in numeric_cols[:8]:
        plt.figure(figsize=(6, 4))
        df[col].dropna().hist(bins=30)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        safe_name = make_safe_filename(col)
        plt.savefig(f"hist_{safe_name}.png")
        plt.close()


def analyze_categorical_columns(df, categorical_cols):
    if not categorical_cols:
        print("\nNo categorical columns found.")
        return

    print("\n" + "=" * 60)
    print("CATEGORICAL ANALYSIS")
    print("=" * 60)

    for col in categorical_cols[:8]:
        print(f"\nTop values for {col}:")
        print(df[col].value_counts(dropna=False).head(10))

        plt.figure(figsize=(8, 5))
        df[col].value_counts(dropna=False).head(10).plot(kind="bar")
        plt.title(f"Top 10 Values: {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.tight_layout()
        safe_name = make_safe_filename(col)
        plt.savefig(f"bar_{safe_name}.png")
        plt.close()


def analyze_correlations(df, numeric_cols):
    if len(numeric_cols) < 2:
        return

    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS")
    print("=" * 60)

    corr = df[numeric_cols].corr(numeric_only=True)
    print(corr)

    plt.figure(figsize=(8, 6))
    plt.imshow(corr, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig("correlation_matrix.png")
    plt.close()


def analyze_text_columns(df, text_cols):
    if not text_cols:
        print("\nNo text columns found.")
        return

    print("\n" + "=" * 60)
    print("TEXT ANALYSIS")
    print("=" * 60)

    stop_words = {
        "the", "and", "for", "with", "that", "this", "from", "you", "your", "are",
        "our", "will", "all", "job", "work", "role", "team", "their", "have", "has",
        "who", "what", "about", "into", "than", "they", "them", "its", "able", "including",
        "within", "across", "such", "through", "while", "where", "when", "also", "may",
        "any", "one", "two", "new", "open", "position", "support", "performs", "variety"
    }

    for col in text_cols[:3]:
        text_series = df[col].dropna().astype(str)

        if len(text_series) == 0:
            continue

        avg_length = text_series.str.len().mean()
        print(f"\nColumn: {col}")
        print(f"Non-null rows: {len(text_series)}")
        print(f"Average text length: {avg_length:.2f}")

        all_words = " ".join(text_series).lower()
        words = re.findall(r"\b[a-zA-Z]{3,}\b", all_words)
        words = [word for word in words if word not in stop_words]

        top_words = Counter(words).most_common(20)
        print("Top words:")
        print(top_words)

        if top_words:
            word_df = pd.DataFrame(top_words, columns=["word", "count"]).sort_values("count")
            plt.figure(figsize=(8, 6))
            plt.barh(word_df["word"], word_df["count"])
            plt.title(f"Top Words in {col}")
            plt.xlabel("Count")
            plt.tight_layout()
            safe_name = make_safe_filename(col)
            plt.savefig(f"words_{safe_name}.png")
            plt.close()


def analyze_business_metrics(df):
    print("\n" + "=" * 60)
    print("BUSINESS ANALYTICS")
    print("=" * 60)

    columns = set(df.columns)

    # top openings by category
    if "NUMBER_OF_OPENINGS_AVAILABLE" in columns:
        print("\nTotal openings available:")
        print(df["NUMBER_OF_OPENINGS_AVAILABLE"].sum())

    if "PRIMARY_LOCATION" in columns:
        print("\nTop locations:")
        print(df["PRIMARY_LOCATION"].value_counts().head(10))

        plt.figure(figsize=(8, 5))
        df["PRIMARY_LOCATION"].value_counts().head(10).plot(kind="bar")
        plt.title("Top 10 Locations by Posting Count")
        plt.xlabel("Location")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig("top_locations.png")
        plt.close()

    if "WORKER_SUB_TYPE" in columns:
        print("\nTop worker sub types:")
        print(df["WORKER_SUB_TYPE"].value_counts().head(10))

        plt.figure(figsize=(8, 5))
        df["WORKER_SUB_TYPE"].value_counts().head(10).plot(kind="bar")
        plt.title("Top 10 Worker Sub Types")
        plt.xlabel("Worker Sub Type")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig("worker_sub_type.png")
        plt.close()

    if "JOB_PROFILE" in columns:
        print("\nTop job profiles:")
        print(df["JOB_PROFILE"].value_counts().head(10))

        plt.figure(figsize=(8, 5))
        df["JOB_PROFILE"].value_counts().head(10).plot(kind="bar")
        plt.title("Top 10 Job Profiles")
        plt.xlabel("Job Profile")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig("job_profiles.png")
        plt.close()

    if "DAYS_OPEN" in columns:
        days_open_numeric = pd.to_numeric(df["DAYS_OPEN"], errors="coerce")
        valid_days = days_open_numeric.dropna()

        if len(valid_days) > 0:
            print("\nDays open summary:")
            print(valid_days.describe())

            plt.figure(figsize=(6, 4))
            valid_days.hist(bins=25)
            plt.title("Distribution of Days Open")
            plt.xlabel("Days Open")
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.savefig("days_open_distribution.png")
            plt.close()

            if "PRIMARY_LOCATION" in columns:
                temp = df.copy()
                temp["DAYS_OPEN_NUMERIC"] = pd.to_numeric(temp["DAYS_OPEN"], errors="coerce")
                by_location = (
                    temp.groupby("PRIMARY_LOCATION")["DAYS_OPEN_NUMERIC"]
                    .mean()
                    .dropna()
                    .sort_values(ascending=False)
                    .head(10)
                )

                if len(by_location) > 0:
                    print("\nAverage days open by location:")
                    print(by_location)

                    plt.figure(figsize=(8, 5))
                    by_location.sort_values().plot(kind="barh")
                    plt.title("Top 10 Locations by Average Days Open")
                    plt.xlabel("Average Days Open")
                    plt.tight_layout()
                    plt.savefig("avg_days_open_by_location.png")
                    plt.close()

            if "JOB_PROFILE" in columns:
                temp = df.copy()
                temp["DAYS_OPEN_NUMERIC"] = pd.to_numeric(temp["DAYS_OPEN"], errors="coerce")
                by_profile = (
                    temp.groupby("JOB_PROFILE")["DAYS_OPEN_NUMERIC"]
                    .mean()
                    .dropna()
                    .sort_values(ascending=False)
                    .head(10)
                )

                if len(by_profile) > 0:
                    print("\nAverage days open by job profile:")
                    print(by_profile)

                    plt.figure(figsize=(8, 5))
                    by_profile.sort_values().plot(kind="barh")
                    plt.title("Top 10 Job Profiles by Average Days Open")
                    plt.xlabel("Average Days Open")
                    plt.tight_layout()
                    plt.savefig("avg_days_open_by_job_profile.png")
                    plt.close()

    if "NUMBER_OF_OPENINGS_AVAILABLE" in columns and "PRIMARY_LOCATION" in columns:
        openings_by_location = (
            df.groupby("PRIMARY_LOCATION")["NUMBER_OF_OPENINGS_AVAILABLE"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )

        print("\nOpenings by location:")
        print(openings_by_location)

        plt.figure(figsize=(8, 5))
        openings_by_location.sort_values().plot(kind="barh")
        plt.title("Top 10 Locations by Total Openings")
        plt.xlabel("Total Openings")
        plt.tight_layout()
        plt.savefig("openings_by_location.png")
        plt.close()

    if "NUMBER_OF_OPENINGS_AVAILABLE" in columns and "JOB_PROFILE" in columns:
        openings_by_profile = (
            df.groupby("JOB_PROFILE")["NUMBER_OF_OPENINGS_AVAILABLE"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )

        print("\nOpenings by job profile:")
        print(openings_by_profile)

        plt.figure(figsize=(8, 5))
        openings_by_profile.sort_values().plot(kind="barh")
        plt.title("Top 10 Job Profiles by Total Openings")
        plt.xlabel("Total Openings")
        plt.tight_layout()
        plt.savefig("openings_by_job_profile.png")
        plt.close()


def generate_insight_text(df):
    print("\n" + "=" * 60)
    print("QUICK BUSINESS TAKEAWAYS")
    print("=" * 60)

    columns = set(df.columns)

    if "PRIMARY_LOCATION" in columns:
        top_location = df["PRIMARY_LOCATION"].value_counts().idxmax()
        top_location_count = df["PRIMARY_LOCATION"].value_counts().max()
        print(f"- The location with the most postings is {top_location} with {top_location_count} postings.")

    if "WORKER_SUB_TYPE" in columns:
        top_subtype = df["WORKER_SUB_TYPE"].value_counts().idxmax()
        top_subtype_count = df["WORKER_SUB_TYPE"].value_counts().max()
        print(f"- The most common worker subtype is {top_subtype} with {top_subtype_count} postings.")

    if "JOB_PROFILE" in columns:
        top_profile = df["JOB_PROFILE"].value_counts().idxmax()
        top_profile_count = df["JOB_PROFILE"].value_counts().max()
        print(f"- The most common job profile is {top_profile} with {top_profile_count} postings.")

    if "DAYS_OPEN" in columns:
        days_open_numeric = pd.to_numeric(df["DAYS_OPEN"], errors="coerce")
        if days_open_numeric.notna().sum() > 0:
            print(f"- The average posting stays open for {days_open_numeric.mean():.2f} days.")
            print(f"- The longest open posting is {days_open_numeric.max():.0f} days.")

    if "NUMBER_OF_OPENINGS_AVAILABLE" in columns:
        print(f"- Total openings in the dataset: {df['NUMBER_OF_OPENINGS_AVAILABLE'].sum()}.")


def make_safe_filename(name):
    return re.sub(r"[^A-Za-z0-9_]+", "_", name.strip())


def save_summary_files(df, numeric_cols, categorical_cols, text_cols):
    pd.DataFrame({
        "column": df.columns,
        "dtype": [str(df[col].dtype) for col in df.columns],
        "missing_count": [df[col].isna().sum() for col in df.columns],
        "unique_values": [df[col].nunique(dropna=True) for col in df.columns],
        "group": [
            "numeric" if col in numeric_cols else
            "categorical" if col in categorical_cols else
            "text" if col in text_cols else
            "other"
            for col in df.columns
        ]
    }).to_csv("column_summary.csv", index=False)

    df.head(25).to_csv("data_preview.csv", index=False)


def main():
    if len(sys.argv) < 2:
        print("Usage: python BusinessAnalytics.py your_file.csv")
        sys.exit(1)

    file_path = sys.argv[1]

    df = load_data(file_path)
    df = clean_column_names(df)
    df = convert_possible_numeric(df)

    numeric_cols, categorical_cols, text_cols = classify_columns(df)

    dataset_overview(df, numeric_cols, categorical_cols, text_cols)
    save_missing_values_chart(df)
    analyze_numeric_columns(df, numeric_cols)
    analyze_categorical_columns(df, categorical_cols)
    analyze_correlations(df, numeric_cols)
    analyze_text_columns(df, text_cols)
    analyze_business_metrics(df)
    generate_insight_text(df)
    save_summary_files(df, numeric_cols, categorical_cols, text_cols)

    print("\nFiles saved:")
    print("- missing_values.png")
    print("- column_summary.csv")
    print("- data_preview.csv")
    print("- hist_*.png")
    print("- bar_*.png")
    print("- words_*.png")
    print("- correlation_matrix.png if numeric columns exist")
    print("- business-specific charts like top_locations.png, openings_by_location.png, etc.")
    print("\nDone.")


if __name__ == "__main__":
    main()
