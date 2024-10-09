#!/usr/bin/env python3

# Description: This script reads the input CSV files and plot the PIs charts

import argparse
import glob
import os
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.common_charts import MAX_PIS, plot_raw_chart, plot_relative_chart
from utils.experim_aliases import EXPERIM_ALIASES
from utils.instance_aliases import INSTANCE_ALIASES
from utils.instance_prices import INSTANCE_PRICES

mpl.rcParams['agg.path.chunksize'] = 100000

OVERWRITE = True
SAVEFIG = True
COMPUTE_COST = False

USD_FACTOR = 3.6e6  # 1 hour in milliseconds


# ====================================================
# Utility functions
def error(msg):
    print('ERROR:', msg)
    exit(1)


def warning(msg):
    print('WARNING:', msg)


verbosity_level = 0


def verbose(msg, level=0):
    if level <= verbosity_level:
        print('  ' * (level - 1), msg)


# ====================================================


# Define a custom sorting function
def instance_sort_key(item):
    # Extract size and remaining string information using regex
    size_priority = 0
    if size_match := re.search(r'(\d+\.)?(Standard|small|medium|large|xlarge)', item):
        size_map = {'Standard': 1, 'small': 2, 'medium': 3, 'large': 4, 'xlarge': 5}
        size_priority = size_map.get(size_match.group(2), 0)
    class_priority = 0
    if class_match := re.search(r'(\d+\.)?(standard|highcpu|highmem)', item):
        class_map = {'standard': 1, 'highcpu': 2, 'highmem': 3}
        class_priority = class_map.get(class_match.group(2), 0)

    remaining_string = item
    instance_count = 0
    if instances_match := re.match(r'(.*?)-([0-9]+)', remaining_string):
        instance_count = int(instances_match.group(2))
        remaining_string = instances_match.group(1)
    version_count = 0
    if version_match := re.match(r'(.*?)_v([0-9])', remaining_string):
        version_count = int(version_match.group(2))
        remaining_string = version_match.group(1)
    size_count = 0
    special_count = 0
    if size_match := re.match(r'(.*?)_(.*?)([0-9]+)(s?)', remaining_string):
        size_count = int(size_match.group(3))
        remaining_string = size_match.group(1)
        if size_match.group(4):
            special_count = 1
    number_count = 0
    if number_match := re.match(r'(.*?)([0-9]+)(.*?)', remaining_string):
        if count := number_match.group(2):
            number_count = int(count)
    return (number_count, size_priority, class_priority, size_count, version_count, instance_count, special_count, item)


def process_window(instance_name, dataframe, normal_array, ignore, max_pi):
    data_len = min(len(dataframe), max_pi)
    window = len(normal_array)
    if window <= 1:
        return (data_len, {instance_name: pd.to_numeric(dataframe['time'].iloc[ignore:data_len].values)})
    df = dataframe['time'].iloc[ignore:].reset_index(drop=True)
    data_len = min(len(df), max_pi)
    smooth_array = []
    for idx in range(data_len):
        # Determine the window size and crop weights accordingly
        start_df_idx = int(max(0, idx - window // 2))
        end_df_idx = int(min(data_len, idx + window // 2 + 1))
        start_normal_idx = int(max(0, window // 2 - idx))
        end_normal_idx = start_normal_idx + end_df_idx - start_df_idx
        # Compute the weighted average for the current element
        avg = np.average(df[start_df_idx:end_df_idx], weights=normal_array[start_normal_idx:end_normal_idx])
        smooth_array.append(avg)
    return (data_len, {instance_name: smooth_array})


def plot_charts(pdfs_results_dir, user, app_name, instance_data, ignore, window, max_pi):
    if len(instance_data) < 2:
        warning('Only one experiment, nothing to plot...')
        return
    app_user = '{}_{}'.format(app_name, user.replace('/', '_')).replace('-', '_')[:115]
    max_pi_str = 'all' if max_pi == MAX_PIS else max_pi
    filename_raw = f'{pdfs_results_dir}/{app_user}_{window}_{ignore}_{max_pi_str}.pdf'
    filename_cost = f'{pdfs_results_dir}/cost_{app_user}_{window}_{ignore}_{max_pi_str}.pdf'
    if os.path.exists(filename_raw) and not OVERWRITE:
        warning(f'File already exists: {filename_raw}, skipping...')
        return
    verbose(f'\n\nGenerating {filename_raw} - {filename_cost}', level=2)

    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7), dpi=400)
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(20, 7), dpi=400)
    experim_alias = EXPERIM_ALIASES[app_name]
    isntance_color_order_list = plot_raw_chart(experim_alias, instance_data, ignore, ax1, max_pi, 'time')
    plot_relative_chart(experim_alias, instance_data, window, ignore, ax2, max_pi, isntance_color_order_list)
    if COMPUTE_COST:
        plot_raw_chart(experim_alias, instance_data, ignore, ax3, max_pi, 'time', isntance_color_order_list)
        plot_raw_chart(experim_alias, instance_data, ignore, ax4, max_pi, 'cost')
    for fig in [fig1, fig2]:
        fig.legend(
            handles=ax2.get_legend().legend_handles if ax2.get_legend() else [],
            loc='upper center',
            bbox_to_anchor=(0.1, 0.95, 0.8, 0.11),
            ncol=7,
            mode='expand',
            fancybox=True,
            shadow=True,
        )
    # Clone x-axis indices from the second subplot to the first subplot
    tick_positions = ax2.get_xticks()
    tick_labels = [str(int(pos + 1)) for pos in tick_positions]
    for ax in [ax1, ax2, ax3, ax4]:
        ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(tick_positions))
        ax.set_xticklabels(tick_labels)
        ax.set_xlabel('Iteration')
    for ax in [ax1, ax3]:
        ax.set_ylabel('Execution time (ms)')
    ax2.set_ylabel('Relative performance')
    ax4.set_ylabel('Execution cost (USD)')
    if not SAVEFIG:
        plt.show()
    for fig, filename in [[fig1, filename_raw], [fig2, filename_cost]]:
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)
        if not COMPUTE_COST:
            # Does not save fig2
            break


def process_user_data(pdfs_results_dir, results_data_location, user, ignore, window, max_pi, filter_app_list=[]):
    user_exps = list(filter(lambda x: user in x, results_data_location))
    apps_user = list(set(map(lambda x: Path(x).parts[-2], user_exps)))
    for app in sorted(apps_user):
        if filter_app_list and not any([remove in app.lower() for remove in filter_app_list]):
            continue
        verbose(f'Processing {app}', 1)
        instance_data = {}
        it_counts = set()
        instances = list(filter(lambda x: app in x, user_exps))
        for instance in instances:
            df = pd.read_csv(instance)
            if 'Unnamed: 0' in df.keys():
                df = df.drop(columns='Unnamed: 0')
            if len(df.keys()) == 1:
                df.columns = ['time']
            # Apparently, the results collected by Thais_Camacho switched columns abs_time and time. Adjusting for this case.
            if user == 'Thais_Camacho':
                df['time'] = df['abs_time']
            if 'rank' in df.columns:
                df = df[df['rank'] == 0]
            instance_name = INSTANCE_ALIASES[instance.split('/')[-1][:-4]]
            inst_name, inst_count = instance_name.split('-')
            inst_price = (INSTANCE_PRICES[inst_name] * int(inst_count)) / USD_FACTOR
            df.loc[:, 'cost'] = df['time'] * inst_price
            instance_data[instance_name] = df.reset_index(drop=True)
            it_counts.add(len(df.index))
        # Sort the dictionary based on keys using the custom sorting function
        sorted_instance_data = dict(sorted(instance_data.items(), key=lambda x: instance_sort_key(x[0])))
        verbose(
            f'{app} instance/size/sum(time): {[[name, len(df.index), df['time'].sum()] for name, df in sorted_instance_data.items()]}',
            4,
        )
        current_max_pi = max_pi
        if len(it_counts) > 1:
            warning('Divergent number of iterations')
            it_counts = sorted(list(it_counts))
            # Geralmente retirar o mais baixo e suficiente
            sorted_instance_data = dict(
                (key, df) for key, df in sorted_instance_data.items() if len(df.index) >= it_counts[1]
            )
            if len(it_counts) > 2:
                # If all the executions still have a divergent number of iteratios, we limit the maximun number of PIs.
                warning(f'Limiting the experiments to {it_counts[1]} iterations')
                current_max_pi = min(current_max_pi, it_counts[1])
        try:
            plot_charts(pdfs_results_dir, user, app, sorted_instance_data, ignore, window, current_max_pi)
        except Exception as e:
            warning(f'Error processing app {app}: {e}')
            continue


if __name__ == '__main__':
    # Initialize parser
    parser = argparse.ArgumentParser()

    # Adding optional argument
    parser.add_argument('-i', '--input_dir', help='Input directory')
    parser.add_argument('-o', '--output', help='Output directory')
    parser.add_argument('-v', '--verbosity', help='Verbosity level: 0 (default), 1, 2, 3, 4', type=int)
    parser.add_argument('-s', '--ignore', help='Number of iterations to be ignored', default=0, type=int)
    parser.add_argument('-w', '--window', help='Number of iterations to be groupoed (smooth)', default=1, type=int)
    parser.add_argument(
        '-m', '--max_pi', help='Maximun number of iterations to be considered', default=MAX_PIS, type=int
    )

    # Read arguments from command line
    args = parser.parse_args()

    if args.verbosity:
        verbosity_level = int(args.verbosity)

    if not args.input_dir:
        error('Input dir expected but not provided (-i)')

    if not os.path.exists(args.input_dir):
        error(f'{args.input_dir} is an invalid directory!')

    if not args.output:
        error('Output argument not suplied!')

    if not os.path.exists(args.output):
        warning(f'{args.output} directory not found, creating it!')
        os.makedirs(args.output)

    results_data_location = glob.glob(os.path.join(args.input_dir, '**', '*.csv'), recursive=True)
    usernames = list(set(map(lambda x: Path(x).parts[-3], results_data_location)))
    usernames.sort()

    if not usernames:
        error(f'Any valid result fount at {args.input_dir}')
    verbose(f'Processing input files for users: {usernames}', level=1)

    verbose(
        f'Working with parameters: ignore: {args.ignore},  window: {args.window}, '
        f'max_pi: {"all" if args.max_pi == MAX_PIS else args.max_pi})',
        level=1,
    )
    for user in usernames:
        verbose(f'Processing data from user {user}', level=1)
        process_user_data(args.output, results_data_location, user, args.ignore, args.window, args.max_pi)
