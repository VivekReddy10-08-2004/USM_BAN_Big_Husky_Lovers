# Project Overview

This repo contains three Python programs:

* `HRcleaner.py`
* `HRDeepClean.py`
* `businessanaylics.py`

They are meant to be used together, but they solve different problems.

## What Each Program Does

### `HRcleaner.py`

`HRcleaner.py` is the basic data-cleaning step.
Its purpose is to take raw HR or open requisition data and turn it into a cleaner, more structured dataset that is easier to use for machine learning and analysis.

It can:

* clean fields like `DAYS_OPEN`
* handle missing values in important HR columns
* clean text fields such as job descriptions
* encode category columns into numeric codes
* save a cleaned CSV
* save mapping files so encoded values can still be understood later
* save a report showing what was cleaned

Use this first when your raw HR dataset is messy and needs preprocessing.

### `HRDeepClean.py`

`HRDeepClean.py` is the enhanced data-cleaning and EDA step.
It does everything `HRcleaner.py` does, plus adds data profiling, quality assessment, keyword extraction, and correlation analysis.

It can:

* do all the cleaning that `HRcleaner.py` does
* generate a full data profile with numeric stats (mean, median, std, quartiles, skewness, kurtosis)
* produce a correlation matrix across numeric columns
* extract top keywords from text columns with stopword filtering and bigram support
* compare raw vs cleaned data side by side (row counts, null counts, column counts)
* assess data quality including completeness ratio, duplicate detection, and format validity
* validate output formats (sentinel values in DAYS_OPEN, code column integrity, empty string rates)

It produces five output files instead of three:

* `Cleaned_ML_Ready.csv`
* `category_mappings.json`
* `cleaning_report.json`
* `data_profile.json` — detailed EDA statistics
* `quality_metrics.json` — data quality assessment

Use this instead of `HRcleaner.py` when you want deeper insight into the data during the cleaning step.

### `businessanaylics.py`

`businessanaylics.py` is the analysis step.
Its purpose is to explore the cleaned dataset and generate useful business insights, summaries, and charts.

It can:

* show dataset shape, columns, and data types
* show missing values
* separate columns into numeric, categorical, and text groups
* create charts for numeric and categorical columns
* analyze text-heavy columns
* show business trends such as locations, job profiles, worker types, and openings
* produce summary files for reporting

Use this after cleaning when you want to understand the data and present insights.

## Recommended Workflow

You have two workflow options depending on how much detail you need.

### Option A: Basic cleaning then analysis

1. Run `HRcleaner.py` on the raw HR dataset.
2. Review the cleaned output files.
3. Run `businessanaylics.py` on the cleaned CSV.
4. Review the charts and summary files.

### Option B: Enhanced cleaning with EDA then analysis

1. Run `HRDeepClean.py` on the raw HR dataset.
2. Review the cleaned CSV, data profile, and quality metrics.
3. Run `businessanaylics.py` on the cleaned CSV.
4. Review the charts and summary files.

Option B gives you profiling and quality data that Option A does not.

## How To Run

Install the needed packages first:

```
pip install pandas numpy matplotlib scikit-learn
```

### Run `HRcleaner.py`

```
python HRcleaner.py "Open Reqs Mar 2026_2026-04-01.csv"
```

Expected outputs:

* `Cleaned_ML_Ready.csv`
* `category_mappings.json`
* `cleaning_report.json`

### Run `HRDeepClean.py`

```
python HRDeepClean.py "Open Reqs Mar 2026_2026-04-01.csv" --output-dir ./output
```

You can also choose how to handle null management levels:

```
python HRDeepClean.py "Open Reqs Mar 2026_2026-04-01.csv" --output-dir ./output --null-strategy drop
```

Expected outputs (in the `./output` directory):

* `Cleaned_ML_Ready.csv`
* `category_mappings.json`
* `cleaning_report.json`
* `data_profile.json`
* `quality_metrics.json`

### Run `businessanaylics.py`

```
python businessanaylics.py Cleaned_ML_Ready.csv
```

Expected outputs may include:

* `missing_values.png`
* `column_summary.csv`
* `data_preview.csv`
* `hist_*.png`
* `bar_*.png`
* `words_*.png`
* business charts such as location, openings, and job profile summaries

## Simple Difference

| Program | Purpose |
|---|---|
| `HRcleaner.py` | Prepares the data (basic) |
| `HRDeepClean.py` | Prepares the data + profiles it + checks quality |
| `businessanaylics.py` | Explains the data with charts and summaries |

## Notes

* Both programs expect CSV input.
* Output files are saved in the current working directory unless `--output-dir` is specified.
* Some outputs depend on which columns exist in the dataset.
* `HRDeepClean.py` and `HRcleaner.py` produce the same `Cleaned_ML_Ready.csv`, so either can feed into `businessanaylics.py`.
