# 2024-Paramount-Iterations-analysis

This repository contains:

1) Scripts to analyze the Paramount Iteration data collected from several applications running on the cloud.
2) Output files produced by these scripts.

## Scripts

- `00-extract-logs.py`: parse the log files and generate a single pickle file with the data for analysis.
- `01-sanity-checking.py`: analyze the data in the pickle file.

### Executing the scripts

```
python 00-extract-logs.py -i prediction_data2 -o prediction_data2.pkl -v 2
python 01-sanity-checking.py -i prediction_data2.pkl --analysis_per_instance > prediction_data2-analysis_per_instance.csv
mkdir charts
python 01-sanity-checking.py -i prediction_data2.pkl --analysis_per_application --application_charts_dir charts > prediction_data2-analysis_per_application.csv
```

## Output files

- `prediction_data2.pkl`: Pickle file with data to be analyzed (extracted from CSV files)
- `prediction_data2-analysis_per_instance.csv`: Statistics about each instance, i.e., the execution of an application on a given cloud instance.
- `prediction_data2-analysis_per_application.csv`: Statistics about the applications.
- `charts/*.pdf`: set of PDF files with charts produced for each one of the applications analyzed.
