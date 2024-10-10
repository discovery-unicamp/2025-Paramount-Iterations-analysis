#!/usr/bin/env python3

# Description: This script reads the processed CSV files and generates the histogram chart and a LaTeX output table

import argparse
import re
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.experim_aliases import EXPERIM_ALIASES


# ====================================================
# Utility functions
def error(msg):
    print('[ERROR] ', msg)
    exit(1)


def warning(msg, csv_filename=None):
    print('[WARNING] ', msg)


verbosity_level = 0


def verbose(msg, level=0):
    if level <= verbosity_level:
        print(msg)


# ====================================================


def get_app_behavior(group, app):
    if app == 'TC-1':
        return 'Unmatch'
    if 'amg' in app or app == '3Dadvect' or group == 'NAMD':
        return 'Phases'
    if group == 'NPB-EP' or 'ember' in app:
        return 'Fuzzy'
    return 'Expected'


def load_and_clean_result(input_file):
    # Clean-up dataframe file
    df = pd.read_csv(input_file)

    remain_columns = [
        'app_alias',
        'app_behavior',
        '# instances',
        'min time',
        'max time',
        'Second PI vs All PIs - R2',
        'Second PI-Cost vs All PIs - R2',
        'From 2 to 5 vs All PIs - R2',
        'From 2 to 5-Cost vs All PIs - R2',
        'From 2 to 10 vs All PIs - R2',
        'From 2 to 10-Cost vs All PIs - R2',
    ]
    # remain_columns += [column for column in df.keys() if column.endswith(' - R2')]

    # df[df['min wallclock_time'].isnull()]
    df['min time'] = df['min wallclock_time'].fillna(df['min PIs sum'])
    df['max time'] = df['max wallclock_time'].fillna(df['max PIs sum'])

    # Filter experiments with lass then 3 configurations
    df = df[df['# instances'] > 2]

    df = df[df['Rank 0 min/max PI sample ratio'] > 0.7]

    df['app_alias'] = df['app'].apply(lambda app: f'\\app{{{EXPERIM_ALIASES[app].replace("_", "\\_")}}}')
    df['app_behavior'] = df.apply(lambda x: get_app_behavior(x.group, x.app), axis=1)
    print(f'Total lines: {len(df.index)}')
    print(df['app_behavior'].value_counts())
    return df[[*remain_columns]]


def correlation_formatter(x):
    if not isinstance(x, float):  # Ensure the value is a float
        return x
    if x < 0.75:
        return '\\textbf{{{:0.2f}}}'.format(x)
    return '{:0.2f}'.format(x)


def cleanup_latex(dataframe):
    latex_table = dataframe.to_latex(index=False, float_format=correlation_formatter)

    lines = latex_table.splitlines()
    # headers = lines[2]
    headers = r'Application & behavior &  cfgs &  min time(s) &  max time(s) & \multicolumn{2}{c|}{Second PI} &  & \multicolumn{2}{c|}{From 2 to 5} &  & \multicolumn{2}{c|}{From 2 to 10} \\&'
    content_lines = lines[4:-2]
    content_lines.sort()

    # Split the lines into cells
    split_lines = [line.split('&') for line in [headers] + content_lines]
    # split_lines = [line.split('&') for line in content_lines]
    # Trim whitespace from cells
    split_lines = [[cell.strip() for cell in line] for line in split_lines]
    # Determine the maximum width for each column
    max_widths = [max(len(cell) for cell in col) for col in zip(*split_lines)]
    # Align the cells by adding spaces
    aligned_lines = []
    for line in split_lines:
        aligned_line = ' & '.join(cell.ljust(max_width) for cell, max_width in zip(line, max_widths)) + ' \\hline'
        aligned_lines.append(aligned_line)
    aligned_lines[0] = re.sub(r'&\s+&', lambda m: f' {m.group(0)[1:]}', aligned_lines[0])

    # Join the aligned lines
    return '\n'.join(aligned_lines)


def generate_latex(df, output_sufix):
    # Save the result
    output_path = f'latex_table-{output_sufix}'
    # df.to_csv(f'{output_path}.csv', float_format='%.2f', index=False)  # float_format='%g'
    with open(f'{output_path}.tex', 'w') as f:
        f.write(cleanup_latex(df))
    verbose(f'Generate from csv to tex: {output_path}.tex', 1)


def generate_histograms(df, output_sufix):
    bins_size = 100
    correlations = ['Second PI', 'From 2 to 5', 'From 2 to 10']
    for metric in [('Time', ''), ('Cost', '-Cost')]:
        count = 0
        fig, axs = plt.subplots(len(correlations), figsize=(9, len(correlations) * 2 - 1))
        fig.subplots_adjust(
            top=0.99, bottom=0.05, left=0.1, right=0.95, hspace=0.5
        )  # Adjust vertical space between subplots
        # Calculate bar width based on figure width
        bar_width = fig.get_size_inches()[0] / (2 * bins_size)

        max_value = 0
        for count, correlation in enumerate(correlations):
            proxy = f'{correlation}{metric[1]} vs All PIs - R2'
            # Set x and y limits
            is_positive = df[proxy].min() >= 0
            axs[count].set_xlim(left=0 if is_positive else -1, right=1)

            # Calculate histogram from the data
            hist, bins = np.histogram(df[proxy], bins=bins_size)
            bin_centers = (bins[:-1] + bins[1:]) / 2

            max_value = max(max_value, hist.max())

            # Line histogram
            axs[count].bar(bin_centers, hist, width=bar_width if is_positive else bar_width / 2)
            title = f'{proxy.split("_", 3)[-1]} histogram'
            axs[count].text(
                0.5, 0.95, title, transform=axs[count].transAxes, ha='center', va='top', fontsize=14, fontweight='bold'
            )

        for ax in axs:
            ax.set_ylim(top=max_value, bottom=-10)

        # Add x-label and y-label for all subplots
        fig.text(0.5, -0.065, f'Pearson correlation of {metric[0]}', ha='center', fontsize=16)
        fig.text(0.01, 0.5, 'Frequency', va='center', rotation='vertical', fontsize=16)
        filename = f'histogram_{metric[0].lower()}-{output_sufix}.pdf'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        verbose(f'Histogram saved: {filename}', 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Adding optional argument
    parser.add_argument('-v', '--verbosity', help='Verbosity level: 0 (default), 1, 2, 3, 4')
    parser.add_argument('-i', '--input_file', help='Input CSV file')
    parser.add_argument(
        '-o', '--output_sufix', help='Output file sufix: default is the current day', default=date.today().isoformat()
    )
    parser.add_argument('--generate_histogram', help='Generate histograms', action='store_true', default=False)
    parser.add_argument('--generate_latex', help='Generate a LaTeX file', action='store_true', default=False)

    # Read arguments from command line
    args = parser.parse_args()

    if args.verbosity:
        verbosity_level = int(args.verbosity)

    if not args.input_file:
        error('Input file expected but not provided (-i)')

    if not args.generate_latex and not args.generate_histogram:
        error('Nothing to be done here!')

    df = load_and_clean_result(args.input_file)

    if args.generate_latex:
        generate_latex(df, args.output_sufix)

    if args.generate_histogram:
        generate_histograms(df, args.output_sufix)
