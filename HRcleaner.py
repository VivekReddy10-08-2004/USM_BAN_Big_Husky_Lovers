import argparse
import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import pandas as pd


@dataclass
class HRDataCleaner:
    """
    Cleans the Open Reqs dataset into an ML-ready file.

    Outputs:
    - Cleaned_ML_Ready.csv
    - category_mappings.json
    - cleaning_report.json
    """

    input_path: str
    output_dir: str = "."
    null_strategy: str = "impute"  # 'impute' keeps more rows, 'drop' keeps stricter data quality

    # These columns are treated as important business categories that should be
    # converted into numeric codes for ML models.
    mapping_columns: List[str] = field(default_factory=lambda: [
        "PRIMARY_LOCATION",
        "WORKER_SUB_TYPE",
        "MANAGEMENT_LEVEL_JOB_REQUISITION",
    ])

    # These text fields are cleaned into model-friendly text while still preserving
    # the original raw columns in the dataset.
    text_columns: List[str] = field(default_factory=lambda: [
        "JOB_DESCRIPTION",
        "JOB_PROFILE_SUMMARY",
    ])

    def load_data(self) -> pd.DataFrame:
        """Load the input CSV and fail early if the file path is wrong."""
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Input file not found: {self.input_path}")
        return pd.read_csv(self.input_path)

    @staticmethod
    def clean_days_open(value) -> pd._libs.missing.NAType | int:
        """
        Extract the numeric part from values like '236 days ago'.

        This makes the field usable as a real numeric feature in analysis or ML.
        """
        if pd.isna(value):
            return pd.NA
        text = str(value).strip().lower()
        match = re.search(r"(\d+)", text)
        if not match:
            return pd.NA
        return int(match.group(1))

    @staticmethod
    def clean_text(value) -> str:
        """
        Standardize text for downstream NLP or feature engineering.

        Lowercasing and removing punctuation helps reduce meaningless variations
        like capitalization and formatting differences.
        """
        if pd.isna(value):
            return ""
        text = str(value).lower()
        text = text.replace("\n", " ").replace("\r", " ")
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"_+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def handle_management_level_nulls(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Handle missing management-level values based on the selected strategy.

        We track before/after stats so the cleaning report can explain how much
        data was changed or removed.
        """
        before_rows = len(df)
        before_nulls = int(df["MANAGEMENT_LEVEL_JOB_REQUISITION"].isna().sum())

        if self.null_strategy == "drop":
            # Dropping nulls gives cleaner category data but may remove useful rows.
            df = df.dropna(subset=["MANAGEMENT_LEVEL_JOB_REQUISITION"]).copy()
        elif self.null_strategy == "impute":
            # Imputing preserves row count and makes the missing category explicit.
            df["MANAGEMENT_LEVEL_JOB_REQUISITION"] = df[
                "MANAGEMENT_LEVEL_JOB_REQUISITION"
            ].fillna("Uncategorized")
        else:
            raise ValueError("null_strategy must be 'drop' or 'impute'")

        after_rows = len(df)
        after_nulls = int(df["MANAGEMENT_LEVEL_JOB_REQUISITION"].isna().sum())
        stats = {
            "rows_before": before_rows,
            "rows_after": after_rows,
            "nulls_before": before_nulls,
            "nulls_after": after_nulls,
            "rows_removed": before_rows - after_rows,
        }
        return df, stats

    @staticmethod
    def build_code_map(series: pd.Series) -> Tuple[pd.Series, Dict[str, int], Dict[str, str]]:
        """
        Encode a categorical column into integers.

        We also return both directions of the mapping so the encoded values remain
        interpretable outside the model.
        """
        # Sorting makes the encoding stable across runs, which helps reproducibility.
        unique_values = sorted(series.dropna().astype(str).unique().tolist())
        forward_map = {value: idx for idx, value in enumerate(unique_values)}
        reverse_map = {str(idx): value for value, idx in forward_map.items()}
        encoded = series.astype(str).map(forward_map).astype("int64")
        return encoded, forward_map, reverse_map

    def encode_categories(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Dict[str, str]]]]:
        """
        Encode all configured business category columns and store their mappings.
        """
        mappings: Dict[str, Dict[str, Dict[str, str]]] = {}

        for col in self.mapping_columns:
            if col not in df.columns:
                raise KeyError(f"Required mapping column not found: {col}")

            encoded_col_name = f"{col}_CODE"
            encoded, forward_map, reverse_map = self.build_code_map(df[col])
            df[encoded_col_name] = encoded
            mappings[col] = {
                "string_to_code": {k: int(v) for k, v in forward_map.items()},
                "code_to_string": reverse_map,
            }

        return df, mappings

    def transform(self) -> Tuple[pd.DataFrame, Dict[str, object]]:
        """
        Run the full cleaning pipeline and return both cleaned data and metadata.
        """
        df = self.load_data()
        report: Dict[str, object] = {
            "input_file": os.path.basename(self.input_path),
            "rows_initial": int(len(df)),
            "columns_initial": list(df.columns),
            "null_strategy": self.null_strategy,
        }

        # Task 1: clean DAYS_OPEN so it becomes usable as a numeric feature.
        if "DAYS_OPEN" not in df.columns:
            raise KeyError("DAYS_OPEN column is missing from the source file.")
        df["DAYS_OPEN"] = df["DAYS_OPEN"].apply(self.clean_days_open).astype("Int64")
        report["days_open_nulls_after_parse"] = int(df["DAYS_OPEN"].isna().sum())

        # Task 2: handle missing management-level values before encoding categories.
        if "MANAGEMENT_LEVEL_JOB_REQUISITION" not in df.columns:
            raise KeyError("MANAGEMENT_LEVEL_JOB_REQUISITION column is missing from the source file.")
        df, null_stats = self.handle_management_level_nulls(df)
        report["management_level_null_handling"] = null_stats

        # Convert DAYS_OPEN to plain int64 after row filtering is done.
        if df["DAYS_OPEN"].isna().any():
            # Use -1 as a sentinel value so the column stays numeric for ML while
            # still signaling "missing/unparsable".
            df["DAYS_OPEN"] = df["DAYS_OPEN"].fillna(-1)
            report["days_open_fill_value"] = -1
        df["DAYS_OPEN"] = df["DAYS_OPEN"].astype("int64")

        # Task 3: create cleaned versions of text fields for downstream modeling.
        cleaned_columns = []
        for col in self.text_columns:
            if col not in df.columns:
                raise KeyError(f"Required text column not found: {col}")
            new_col = f"CLEANED_{col}"

            # Keep both raw and cleaned text so analysts can inspect originals while
            # models use the normalized version.
            df[new_col] = df[col].apply(self.clean_text)
            cleaned_columns.append(new_col)
        report["cleaned_text_columns"] = cleaned_columns

        # Task 4: convert core business dimensions into numeric codes.
        df, mappings = self.encode_categories(df)
        report["encoded_columns"] = [f"{col}_CODE" for col in self.mapping_columns]

        # Save schema details so we can verify the final dataset is ML-ready.
        report["final_dtypes"] = {col: str(dtype) for col, dtype in df.dtypes.items()}
        report["rows_final"] = int(len(df))
        report["output_columns"] = list(df.columns)

        return df, {"report": report, "mappings": mappings}

    def save_outputs(self) -> Tuple[str, str, str]:
        """
        Save the cleaned dataset plus metadata files that explain the transformation.
        """
        os.makedirs(self.output_dir, exist_ok=True)

        cleaned_df, metadata = self.transform()

        csv_path = os.path.join(self.output_dir, "Cleaned_ML_Ready.csv")
        mapping_path = os.path.join(self.output_dir, "category_mappings.json")
        report_path = os.path.join(self.output_dir, "cleaning_report.json")

        cleaned_df.to_csv(csv_path, index=False)

        # Mapping JSON keeps encoded category values understandable outside Python.
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(metadata["mappings"], f, indent=2, ensure_ascii=False)

        # Report JSON acts as a lightweight audit trail of the preprocessing.
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(metadata["report"], f, indent=2, ensure_ascii=False)

        return csv_path, mapping_path, report_path


def clean_hr_data(filepath: str, output_dir: str = ".", null_strategy: str = "impute") -> Tuple[str, str, str]:
    """Convenience wrapper so the cleaner can be reused from other scripts."""
    cleaner = HRDataCleaner(input_path=filepath, output_dir=output_dir, null_strategy=null_strategy)
    return cleaner.save_outputs()


def main() -> None:
    """CLI entry point for running the cleaner from the terminal."""
    parser = argparse.ArgumentParser(description="Clean HR open requisition data into an ML-ready dataset.")
    parser.add_argument("input_path", help="Path to Open Reqs Mar 2026_2026-04-01.csv")
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory where Cleaned_ML_Ready.csv and metadata files will be saved.",
    )
    parser.add_argument(
        "--null-strategy",
        choices=["impute", "drop"],
        default="impute",
        help="How to handle null MANAGEMENT_LEVEL_JOB_REQUISITION values.",
    )
    args = parser.parse_args()

    csv_path, mapping_path, report_path = clean_hr_data(
        filepath=args.input_path,
        output_dir=args.output_dir,
        null_strategy=args.null_strategy,
    )

    print("Created files:")
    print(f"- {csv_path}")
    print(f"- {mapping_path}")
    print(f"- {report_path}")


if __name__ == "__main__":
    main()
