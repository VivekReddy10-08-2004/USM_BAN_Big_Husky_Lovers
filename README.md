# DataOverview

This folder contains two Python programs that help you understand a dataset from two different angles:

- `BusinessAnalytics.py` explores the data and produces business-style summaries and charts.
- `DataOverview.py` builds a quick baseline machine learning model so you can test prediction ideas.

Together, they give you a simple workflow:

1. Use `BusinessAnalytics.py` to understand the dataset, missing values, major categories, text patterns, and business trends.
2. Use `DataOverview.py` to try a baseline prediction model and see which features matter most.

## Purpose Of Each Program

### `BusinessAnalytics.py`

This script is for exploratory data analysis and business insight generation.

It helps answer questions like:

- How many rows and columns are in the dataset?
- Which columns are numeric, categorical, or text-heavy?
- Where are the missing values?
- Which locations, job profiles, or worker types appear most often?
- How many openings are available?
- Which locations or job profiles stay open the longest?
- What words appear most often in text fields?

It is useful when you want charts and summary files that are easy to discuss in a business report or presentation.

### `DataOverview.py`

This script is for quick machine learning experimentation.

It helps answer questions like:

- What column should be used as the prediction target?
- Is this a classification problem or a regression problem?
- How well does a basic model perform?
- Which input features seem most important?

It is useful when you want a fast baseline model before doing deeper ML work.

## Requirements

Install Python packages before running the scripts:

```bash
pip install pandas numpy matplotlib scikit-learn
```

## How To Run

Open a terminal in this folder:

```bash
cd DataOverview
```

### Run `BusinessAnalytics.py`

Use:

```bash
python BusinessAnalytics.py your_file.csv
```

Example:

```bash
python BusinessAnalytics.py hackathon.csv
```

What it does:

- loads the CSV file
- trims column names
- tries to convert columns to numeric when it is safe
- groups columns into numeric, categorical, and text
- prints a dataset overview
- creates charts for missing values, numeric columns, categorical columns, correlations, and text terms
- creates business-specific summaries for hiring-related columns when they exist
- saves a preview file and a column summary file

Main output files:

- `missing_values.png`
- `column_summary.csv`
- `data_preview.csv`
- `hist_*.png`
- `bar_*.png`
- `words_*.png`
- `correlation_matrix.png`
- `top_locations.png`
- `worker_sub_type.png`
- `job_profiles.png`
- `openings_by_location.png`
- `openings_by_job_profile.png`
- `avg_days_open_by_location.png`
- `avg_days_open_by_job_profile.png`

### Run `DataOverview.py`

Use:

```bash
python DataOverview.py your_file.csv
```

Or specify the target column manually:

```bash
python DataOverview.py your_file.csv target_column_name
```

Example:

```bash
python DataOverview.py titanic.csv Survived
```

What it does:

- loads the CSV file
- prints a quick overview of the dataset
- saves missing-value and numeric distribution charts
- picks a target column automatically if you do not provide one
- detects whether the target looks like classification or regression
- preprocesses numeric and categorical features
- trains a baseline random forest model
- prints evaluation metrics
- saves feature importance outputs when available

Main output files:

- `missing_values.png`
- `hist_*.png`
- `feature_importance.csv`
- `feature_importance.png`

## Recommended Workflow

If you are starting with a new dataset, use this order:

1. Run `BusinessAnalytics.py` first.
2. Review the generated charts and summary CSV files.
3. Decide which column would make a good prediction target.
4. Run `DataOverview.py` with that target column.
5. Review the model metrics and feature importance results.

## Example Use Cases

For hiring or HR data:

- `BusinessAnalytics.py` helps you understand demand by location, job profile, worker subtype, and time open.
- `DataOverview.py` helps you test whether a column like `DAYS_OPEN`, `MANAGEMENT_LEVEL_JOB_REQUISITION`, or another field can be predicted from the rest of the data.

For general datasets:

- `BusinessAnalytics.py` gives you quick exploratory analysis.
- `DataOverview.py` gives you a simple first-pass ML model.

## Notes

- Both scripts expect a CSV file as input.
- Output files are saved in the current working directory.
- Some charts only appear when the needed columns exist in the dataset.
- `DataOverview.py` uses a basic model for speed and simplicity, not final production accuracy.

## Troubleshooting

If you get `Could not find file`, check that:

- the CSV file path is correct
- you are running the command from the correct folder

If the model results look weak:

- try choosing a better target column manually
- clean the dataset more before training
- check for too many missing values or low-quality categories

If business charts are missing:

- your dataset may not contain columns like `PRIMARY_LOCATION`, `JOB_PROFILE`, `WORKER_SUB_TYPE`, `DAYS_OPEN`, or `NUMBER_OF_OPENINGS_AVAILABLE`

