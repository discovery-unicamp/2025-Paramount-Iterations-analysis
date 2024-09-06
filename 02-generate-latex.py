#!/usr/bin/env python3

import argparse
import re
from datetime import date

import pandas as pd

from experim_aliases import EXPERIM_ALIASES


# ====================================================
# Utility functions
def error(msg):
    print('[ERROR] ', msg)
    exit(1)


def warning(msg, csv_filename=None):
    print('[WARNING] ', msg)


verbosity_level = 3


def verbose(msg, level=0):
    if level <= verbosity_level:
        print(msg)


# ====================================================


def load_and_clean_result(input_file):
    # Clean-up dataframe file
    df = pd.read_csv(input_file)

    remain_columns = ['app_alias', '# instances', 'min time', 'max time']
    remain_columns += [column for column in df.keys() if column.endswith(' - R2')]

    # df[df['min wallclock_time'].isnull()]
    df['min time'] = df['min wallclock_time'].fillna(df['min PIs sum'])
    df['max time'] = df['max wallclock_time'].fillna(df['max PIs sum'])

    # Filter experiments with lass then 3 configurations
    df = df[df['# instances'] > 2]

    df = df[df['Rank 0 min/max PI sample ratio'] > 0.7]

    df['app_alias'] = df['app'].apply(lambda app: f'\\app{{{EXPERIM_ALIASES[app].replace("_", "\\_")}}}')

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
    headers = r'Application &  cfgs &  min time(s) &  max time(s) & \multicolumn{2}{c|}{Second PI} &  & \multicolumn{2}{c|}{From 2 to 5} &  & \multicolumn{2}{c|}{From 2 to 10} &  & \multicolumn{2}{c|}{0.5s} &  & \multicolumn{2}{c|}{0.5s-first}  \\&'
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


def generate_latex(input_file):
    # Save the result
    output_path = f'latex_table-{date.today().isoformat()}'
    # df.to_csv(f'{output_path}.csv', float_format='%.2f', index=False)  # float_format='%g'
    with open(f'{output_path}.tex', 'w') as f:
        f.write(cleanup_latex(load_and_clean_result(input_file)))
    print(f'Generate from csv to tex: {output_path}.tex')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Adding optional argument
    parser.add_argument('-i', '--input_file', help='Input pickle file (.csv) ')
    parser.add_argument('-v', '--verbosity', help='Verbosity level: 0 (default), 1, 2, 3, 4')

    # Read arguments from command line
    args = parser.parse_args()

    if args.verbosity:
        verbosity_level = int(args.verbosity)

    if not args.input_file:
        error('Input file expected but not provided (-i)')

    generate_latex(args.input_file)
