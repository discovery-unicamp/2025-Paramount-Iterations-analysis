# 2024-Paramount-Iterations-analysis

This repository contains:

1) Scripts to analyze the Paramount Iteration data collected from several applications running on the cloud.
2) Key output files produced by these scripts.

## Scripts

- `00_process_raw_csv.py`: Process Paramount Iterations CSVs of multiple executions, producing the charts of multiple executions and selecting the median result.
- `01-plot-pis.py`: Compare the iterations of an application in multiple cloud executions.
- `02-extract-logs.py`: Parse the median CSVs files and generate a single pickle file with the data for analysis.
- `03-sanity-checking.py`: Analyze the data in the pickle file, producing `analysis_per_application` and `analysis_per_instance`.
- `04-post-process.py`: Generate the histogram charts and LaTex table based on the `analysis_per_application.csv`.

### Executing the scripts

```sh
./00_process_raw_csv.py --input_dir <path/to/csv_all_data-dir> --verbosity 3 --csv_data_dir csv_selected_data --charts_dir charts_mult-exec
./01-plot-pis.py --input_dir csv_selected_data --output charts_pi --verbosity 3
./02-extract-logs.py --input_dir csv_selected_data --output_file prediction_data.pkl --verbosity 3
./03-sanity-checking.py --input_file prediction_data.pkl --analysis_per_instance > prediction_data-analysis_per_instance.csv
./03-sanity-checking.py --input_file prediction_data.pkl --analysis_per_application --application_charts_dir charts_analysis > prediction_data-analysis_per_application.csv
./04-post-process.py --input_file prediction_data-analysis_per_application.csv --verbosity 3 --output_sufix prediction_data --generate_histogram --generate_latex
```

## Outputs

### Files
- `prediction_data.pkl`: Pickle file with data to be analyzed (extracted from CSV files)
- `prediction_data-analysis_per_instance.csv`: Statistics about each instance, i.e., the execution of an application on a given cloud instance.
- `prediction_data-analysis_per_application.csv`: Statistics about the applications.
- `latex_table-prediction_data.tex`: LaTeX version of input table.
- `histogram_[time|cost]-prediction_data.pdf`: Histograms of Pearson Correlations for time and cost of metrics.

### Directories
- `csv_selected_data/**/*.csv`: Filtered median CSVs of experiments.
- `charts_mult-exec/*.pdf`: Charts of multiple executions on an experiment at the same cloud configuration.
- `charts_pi/*.pdf`: Plots of Paramount Iterations in all cloud configurations.
- `charts_analysis/[costs/]*.pdf`: Set of PDF files with charts produced for each one of the applications analyzed.
