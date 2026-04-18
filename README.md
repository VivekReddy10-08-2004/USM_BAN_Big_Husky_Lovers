# Project Overview

This repo contains two Python programs:

- `businessanaylics.py`
- `HRcleaner.py`

They are meant to be used together, but they solve different problems.

## What Each Program Does

### `HRcleaner.py`

`HRcleaner.py` is the data-cleaning step.

Its purpose is to take raw HR or open requisition data and turn it into a cleaner, more structured dataset that is easier to use for machine learning and analysis.

It can:

- clean fields like `DAYS_OPEN`
- handle missing values in important HR columns
- clean text fields such as job descriptions
- encode category columns into numeric codes
- save a cleaned CSV
- save mapping files so encoded values can still be understood later
- save a report showing what was cleaned

Use this first when your raw HR dataset is messy and needs preprocessing.

### `businessanaylics.py`

`businessanaylics.py` is the analysis step.

Its purpose is to explore the cleaned dataset and generate useful business insights, summaries, and charts.

It can:

- show dataset shape, columns, and data types
- show missing values
- separate columns into numeric, categorical, and text groups
- create charts for numeric and categorical columns
- analyze text-heavy columns
- show business trends such as locations, job profiles, worker types, and openings
- produce summary files for reporting

Use this after cleaning when you want to understand the data and present insights.

## Recommended Workflow

Use the programs in this order:

1. Run `HRcleaner.py` on the raw HR dataset.
2. Review the cleaned output files.
3. Run `businessanaylics.py` on the cleaned CSV.
4. Review the charts and summary files.

## How To Run

Install the needed packages first:

```bash
pip install pandas numpy matplotlib scikit-learn
```

### Run `HRcleaner.py`

```bash
python HRcleaner.py your_file.csv
```

Example:

```bash
python HRcleaner.py "Open Reqs Mar 2026_2026-04-01.csv"
```

Expected outputs:

- `Cleaned_ML_Ready.csv`
- `category_mappings.json`
- `cleaning_report.json`

### Run `businessanaylics.py`

```bash
python businessanaylics.py Cleaned_ML_Ready.csv
```

Example:

```bash
python businessanaylics.py Cleaned_ML_Ready.csv
```

Expected outputs may include:

- `missing_values.png`
- `column_summary.csv`
- `data_preview.csv`
- `hist_*.png`
- `bar_*.png`
- `words_*.png`
- business charts such as location, openings, and job profile summaries

## Simple Difference

- `HRcleaner.py` prepares the data.
- `businessanaylics.py` explains the data.

## Notes

- Both programs expect CSV input.
- Output files are saved in the current working directory unless the script says otherwise.
- Some outputs depend on which columns exist in the dataset.

