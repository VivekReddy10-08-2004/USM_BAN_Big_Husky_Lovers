import argparse
import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any
from collections import Counter

import pandas as pd
import numpy as np


# Common English stopwords for keyword extraction
STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need", "must",
    "it", "its", "this", "that", "these", "those", "i", "we", "you", "he",
    "she", "they", "me", "him", "her", "us", "them", "my", "your", "his",
    "our", "their", "what", "which", "who", "whom", "how", "when", "where",
    "why", "not", "no", "nor", "so", "if", "then", "than", "too", "very",
    "just", "about", "above", "after", "again", "all", "also", "am", "any",
    "as", "because", "before", "between", "both", "each", "few", "more",
    "most", "other", "out", "over", "own", "same", "some", "such", "under",
    "until", "up", "while", "into", "through", "during", "only", "here",
    "there", "once", "further", "s", "t", "don", "didn", "doesn", "won",
})


def _safe_int(value: Any) -> int:
    """Safely convert a value to int, returning 0 for NaN/None."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0
    return int(value)


def _safe_float(value: Any, decimals: int = 4) -> float:
    """Safely convert a value to a rounded float, returning 0.0 for NaN/None."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    return round(float(value), decimals)


@dataclass
class HRDataCleaner:
    """
    Cleans the Open Reqs dataset into an ML-ready file with comprehensive EDA.

    Outputs:
    - Cleaned_ML_Ready.csv
    - category_mappings.json
    - cleaning_report.json
    - data_profile.json  (detailed EDA stats)
    - quality_metrics.json (data quality assessment)
    """

    input_path: str
    output_dir: str = "."
    null_strategy: str = "impute"
    mapping_columns: List[str] = field(default_factory=lambda: [
        "PRIMARY_LOCATION",
        "WORKER_SUB_TYPE",
        "MANAGEMENT_LEVEL_JOB_REQUISITION",
    ])
    text_columns: List[str] = field(default_factory=lambda: [
        "JOB_DESCRIPTION",
        "JOB_PROFILE_SUMMARY",
    ])
    custom_stopwords: frozenset = field(default_factory=lambda: STOPWORDS)
    top_n_keywords: int = 20
    ngram_range: Tuple[int, int] = (1, 2)

    def load_data(self) -> pd.DataFrame:
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Input file not found: {self.input_path}")
        return pd.read_csv(self.input_path)

    @staticmethod
    def clean_days_open(value) -> pd._libs.missing.NAType | int:
        """Convert values like '236 days ago' into 236."""
        if pd.isna(value):
            return pd.NA
        text = str(value).strip().lower()
        match = re.search(r"(\d+)", text)
        if not match:
            return pd.NA
        return int(match.group(1))

    @staticmethod
    def clean_text(value) -> str:
        """Lowercase text, remove punctuation, remove newlines, collapse whitespace."""
        if pd.isna(value):
            return ""
        text = str(value).lower()
        text = text.replace("\n", " ").replace("\r", " ")
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"_+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def handle_management_level_nulls(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        before_rows = len(df)
        before_nulls = int(df["MANAGEMENT_LEVEL_JOB_REQUISITION"].isna().sum())

        if self.null_strategy == "drop":
            df = df.dropna(subset=["MANAGEMENT_LEVEL_JOB_REQUISITION"]).copy()
        elif self.null_strategy == "impute":
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
    def build_code_map(
        series: pd.Series,
    ) -> Tuple[pd.Series, Dict[str, int], Dict[str, str]]:
        """Encode a categorical series and return forward and reverse mappings."""
        unique_values = sorted(series.dropna().astype(str).unique().tolist())
        forward_map = {value: idx for idx, value in enumerate(unique_values)}
        reverse_map = {str(idx): value for value, idx in forward_map.items()}
        encoded = series.astype(str).map(forward_map).astype("int64")
        return encoded, forward_map, reverse_map

    def encode_categories(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Dict[str, str]]]]:
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

    # ------------------------------------------------------------------
    # Keyword extraction
    # ------------------------------------------------------------------

    def _extract_ngrams(self, tokens: List[str]) -> List[str]:
        """Generate n-grams from a token list within self.ngram_range."""
        results = []
        lo, hi = self.ngram_range
        for n in range(lo, hi + 1):
            for i in range(len(tokens) - n + 1):
                gram = " ".join(tokens[i : i + n])
                results.append(gram)
        return results

    def extract_keywords(self, texts: pd.Series) -> Dict[str, int]:
        """
        Extract top keywords from a text series with stopword filtering
        and n-gram support.
        """
        all_tokens: List[str] = []
        for text in texts.fillna("").astype(str):
            words = text.lower().split()
            filtered = [w for w in words if w not in self.custom_stopwords and len(w) > 1]
            all_tokens.extend(self._extract_ngrams(filtered))

        freq = Counter(all_tokens).most_common(self.top_n_keywords)
        return {term: count for term, count in freq}

    # ------------------------------------------------------------------
    # Data profiling
    # ------------------------------------------------------------------

    def generate_data_profile(
        self,
        df_raw: pd.DataFrame,
        df_clean: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive EDA statistics on both raw and cleaned data:
        distributions, cardinality, text statistics, correlations, and
        format validation.
        """
        profile: Dict[str, Any] = {}

        # --- Numeric columns ---
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        profile["numeric_columns"] = {}
        for col in numeric_cols:
            series = df_clean[col].dropna()
            if len(series) > 0:
                profile["numeric_columns"][col] = {
                    "dtype": str(df_clean[col].dtype),
                    "count": int(series.count()),
                    "nulls": int(df_clean[col].isna().sum()),
                    "mean": _safe_float(series.mean()),
                    "median": _safe_float(series.median()),
                    "std": _safe_float(series.std()),
                    "min": _safe_float(series.min()),
                    "max": _safe_float(series.max()),
                    "q25": _safe_float(series.quantile(0.25)),
                    "q75": _safe_float(series.quantile(0.75)),
                    "skewness": _safe_float(series.skew()),
                    "kurtosis": _safe_float(series.kurtosis()),
                }

        # --- Correlation matrix ---
        if len(numeric_cols) >= 2:
            corr = df_clean[numeric_cols].corr()
            profile["correlation_matrix"] = {
                col: {row: _safe_float(corr.loc[row, col]) for row in corr.index}
                for col in corr.columns
            }
        else:
            profile["correlation_matrix"] = {}

        # --- Categorical columns (JSON-safe int casts) ---
        categorical_cols = df_clean.select_dtypes(include=["object"]).columns.tolist()
        profile["categorical_columns"] = {}
        for col in categorical_cols:
            value_counts = df_clean[col].value_counts()
            profile["categorical_columns"][col] = {
                "dtype": str(df_clean[col].dtype),
                "cardinality": int(df_clean[col].nunique()),
                "nulls": int(df_clean[col].isna().sum()),
                "top_5_values": {
                    str(k): int(v) for k, v in value_counts.head(5).items()
                },
                "null_percentage": round(
                    float(df_clean[col].isna().sum() / len(df_clean) * 100), 2
                ),
            }

        # --- Text columns (with stopword-filtered keywords + n-grams) ---
        profile["text_analysis"] = {}
        for col in self.text_columns:
            if col not in df_clean.columns:
                continue

            non_null = df_clean[col].dropna()
            if len(non_null) == 0:
                profile["text_analysis"][col] = {
                    "avg_char_length": 0.0,
                    "max_char_length": 0,
                    "min_char_length": 0,
                    "avg_word_count": 0.0,
                    "max_word_count": 0,
                    "nulls": int(df_clean[col].isna().sum()),
                    "top_keywords": {},
                }
                continue

            text_lengths = non_null.str.len()
            word_counts = non_null.str.split().str.len()
            profile["text_analysis"][col] = {
                "avg_char_length": _safe_float(text_lengths.mean(), 1),
                "max_char_length": _safe_int(text_lengths.max()),
                "min_char_length": _safe_int(text_lengths.min()),
                "avg_word_count": _safe_float(word_counts.mean(), 1),
                "max_word_count": _safe_int(word_counts.max()),
                "nulls": int(df_clean[col].isna().sum()),
                "top_keywords": self.extract_keywords(non_null),
            }

        # --- Before / after snapshot ---
        profile["raw_vs_clean"] = {
            "rows_raw": len(df_raw),
            "rows_clean": len(df_clean),
            "columns_raw": len(df_raw.columns),
            "columns_clean": len(df_clean.columns),
            "nulls_raw": int(df_raw.isna().sum().sum()),
            "nulls_clean": int(df_clean.isna().sum().sum()),
        }

        return profile

    # ------------------------------------------------------------------
    # Quality metrics
    # ------------------------------------------------------------------

    def generate_quality_metrics(
        self,
        df_raw: pd.DataFrame,
        df_clean: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Assess data quality: completeness, consistency, uniqueness,
        and column-level format validity.
        """
        metrics: Dict[str, Any] = {}

        # Completeness
        clean_size = df_clean.size if df_clean.size > 0 else 1
        metrics["completeness"] = {
            "total_values_raw": int(df_raw.size),
            "total_values_clean": int(df_clean.size),
            "null_count_raw": int(df_raw.isna().sum().sum()),
            "null_count_clean": int(df_clean.isna().sum().sum()),
            "completeness_ratio": round(
                float((1 - df_clean.isna().sum().sum() / clean_size) * 100), 2
            ),
        }

        # Consistency
        metrics["consistency"] = {
            "duplicate_rows_raw": int(df_raw.duplicated().sum()),
            "duplicate_rows_clean": int(df_clean.duplicated().sum()),
        }

        # Coverage
        raw_len = len(df_raw) if len(df_raw) > 0 else 1
        metrics["coverage"] = {
            "rows_retained_pct": round(float(len(df_clean) / raw_len * 100), 2),
            "columns_produced": len(df_clean.columns),
            "encoded_columns": len(
                [c for c in df_clean.columns if c.endswith("_CODE")]
            ),
        }

        # Format validity per column
        validity: Dict[str, Dict[str, Any]] = {}

        if "DAYS_OPEN" in df_clean.columns:
            days_series = df_clean["DAYS_OPEN"]
            negative = int((days_series < 0).sum())
            validity["DAYS_OPEN"] = {
                "valid_positive_int": int(len(days_series) - negative),
                "sentinel_negative_ones": negative,
                "valid_pct": round(
                    float((len(days_series) - negative) / max(len(days_series), 1) * 100), 2
                ),
            }

        for col in self.mapping_columns:
            code_col = f"{col}_CODE"
            if code_col in df_clean.columns:
                series = df_clean[code_col]
                validity[code_col] = {
                    "all_non_negative": bool((series >= 0).all()),
                    "unique_codes": int(series.nunique()),
                    "dtype": str(series.dtype),
                }

        for col in self.text_columns:
            cleaned_col = f"CLEANED_{col}"
            if cleaned_col in df_clean.columns:
                series = df_clean[cleaned_col]
                empty_count = int((series == "").sum())
                validity[cleaned_col] = {
                    "empty_strings": empty_count,
                    "non_empty": int(len(series) - empty_count),
                    "non_empty_pct": round(
                        float((len(series) - empty_count) / max(len(series), 1) * 100), 2
                    ),
                }

        metrics["format_validity"] = validity

        return metrics

    # ------------------------------------------------------------------
    # Transform pipeline
    # ------------------------------------------------------------------

    def transform(
        self,
    ) -> Tuple[pd.DataFrame, Dict[str, object], Dict[str, Any], Dict[str, Any]]:
        df_raw = self.load_data()
        df = df_raw.copy()

        report: Dict[str, object] = {
            "input_file": os.path.basename(self.input_path),
            "rows_initial": int(len(df)),
            "columns_initial": list(df.columns),
            "null_strategy": self.null_strategy,
        }

        # Task 1: DAYS_OPEN cleanup
        if "DAYS_OPEN" not in df.columns:
            raise KeyError("DAYS_OPEN column is missing from the source file.")
        df["DAYS_OPEN"] = df["DAYS_OPEN"].apply(self.clean_days_open).astype("Int64")
        report["days_open_nulls_after_parse"] = int(df["DAYS_OPEN"].isna().sum())

        # Task 2: management level null handling
        if "MANAGEMENT_LEVEL_JOB_REQUISITION" not in df.columns:
            raise KeyError(
                "MANAGEMENT_LEVEL_JOB_REQUISITION column is missing from the source file."
            )
        df, null_stats = self.handle_management_level_nulls(df)
        report["management_level_null_handling"] = null_stats

        # Cast DAYS_OPEN to int64 after any row removal
        if df["DAYS_OPEN"].isna().any():
            df["DAYS_OPEN"] = df["DAYS_OPEN"].fillna(-1)
            report["days_open_fill_value"] = -1
        df["DAYS_OPEN"] = df["DAYS_OPEN"].astype("int64")

        # Task 3: text cleanup
        cleaned_columns = []
        for col in self.text_columns:
            if col not in df.columns:
                raise KeyError(f"Required text column not found: {col}")
            new_col = f"CLEANED_{col}"
            df[new_col] = df[col].apply(self.clean_text)
            cleaned_columns.append(new_col)
        report["cleaned_text_columns"] = cleaned_columns

        # Task 4: categorical encoding + mapping JSON
        df, mappings = self.encode_categories(df)
        report["encoded_columns"] = [f"{col}_CODE" for col in self.mapping_columns]

        # Final dtypes and shape
        report["final_dtypes"] = {
            col: str(dtype) for col, dtype in df.dtypes.items()
        }
        report["rows_final"] = int(len(df))
        report["output_columns"] = list(df.columns)

        # EDA profile (raw vs clean) and quality metrics
        data_profile = self.generate_data_profile(df_raw, df)
        quality_metrics = self.generate_quality_metrics(df_raw, df)

        return df, {"report": report, "mappings": mappings}, data_profile, quality_metrics

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_outputs(self) -> Tuple[str, str, str, str, str]:
        os.makedirs(self.output_dir, exist_ok=True)

        cleaned_df, metadata, data_profile, quality_metrics = self.transform()

        csv_path = os.path.join(self.output_dir, "Cleaned_ML_Ready.csv")
        mapping_path = os.path.join(self.output_dir, "category_mappings.json")
        report_path = os.path.join(self.output_dir, "cleaning_report.json")
        profile_path = os.path.join(self.output_dir, "data_profile.json")
        quality_path = os.path.join(self.output_dir, "quality_metrics.json")

        cleaned_df.to_csv(csv_path, index=False)

        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(metadata["mappings"], f, indent=2, ensure_ascii=False)

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(metadata["report"], f, indent=2, ensure_ascii=False)

        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(data_profile, f, indent=2, ensure_ascii=False)

        with open(quality_path, "w", encoding="utf-8") as f:
            json.dump(quality_metrics, f, indent=2, ensure_ascii=False)

        return csv_path, mapping_path, report_path, profile_path, quality_path


def clean_hr_data(
    filepath: str, output_dir: str = ".", null_strategy: str = "impute"
) -> Tuple[str, str, str, str, str]:
    cleaner = HRDataCleaner(
        input_path=filepath, output_dir=output_dir, null_strategy=null_strategy
    )
    return cleaner.save_outputs()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean HR open requisition data into an ML-ready dataset with EDA."
    )
    parser.add_argument("input_path", help="Path to Open Reqs CSV file")
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory where output files will be saved.",
    )
    parser.add_argument(
        "--null-strategy",
        choices=["impute", "drop"],
        default="impute",
        help="How to handle null MANAGEMENT_LEVEL_JOB_REQUISITION values.",
    )
    args = parser.parse_args()

    csv_path, mapping_path, report_path, profile_path, quality_path = clean_hr_data(
        filepath=args.input_path,
        output_dir=args.output_dir,
        null_strategy=args.null_strategy,
    )

    print("Created files:")
    print(f"  - {csv_path}")
    print(f"  - {mapping_path}")
    print(f"  - {report_path}")
    print(f"  - {profile_path}")
    print(f"  - {quality_path}")


if __name__ == "__main__":
    main()