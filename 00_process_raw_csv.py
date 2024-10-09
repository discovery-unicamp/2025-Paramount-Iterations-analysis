#!/usr/bin/env python3

# Description: This scrip is used to read multiuple execution CSV files and select the average execution
# It also plots the comparison of multiple executions PIs

import argparse
import glob
import os
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common_charts import plot_raw_chart, plot_relative_chart
from experim_aliases import EXPERIM_ALIASES

mpl.rcParams['agg.path.chunksize'] = 10000

raw_colums = ['title', 'rank', 'iteration', 'time', 'abs_time']
raw_colums_user02 = ['title', 'rank', 'iteration', 'abs_time', 'time']

colums_type = {'title': str, 'rank': 'Int64', 'iteration': 'Int64', 'time': np.float64, 'abs_time': np.float64}


# ====================================================
# Utility functions
def error(*msg):
    print('ERROR:', *msg)
    exit(1)


def warning(*msg):
    print('WARNING:', *msg)


verbosity_level = 0


def verbose(*msg, level=0):
    if level <= verbosity_level:
        print('  ' * (level - 1), *msg)


# ====================================================


def raw_read_rank_0(exp):
    if os.stat(exp).st_size == 0:
        warning('Empty file: ', exp)
        return pd.DataFrame()
    try:
        if user.lower() == 'user01':
            df = pd.read_csv(exp, header=None, names=['time'], dtype=np.float64)
            if 'npb' in exp:
                # Note that it is used only for reading and comparing a dataset against others
                threads = 64
                # df = df.iloc[threads//2::threads].reset_index(drop=True)  # Get the midle iteration every X iterations
                # df = pd.DataFrame({'time': df.groupby(df.index // threads)['time'].median()}) # Assuming that Rank0 is the median of every 64 iterations
                arrange = np.arange(0, threads // 2 + 1, 1)
                weights_array = np.concatenate((arrange, np.flip(arrange)))
                grouped = df.groupby(df.index // threads)['time']
                # Compute an weighted average discarting the outmost values, favoring the median ones
                df = pd.DataFrame(
                    {'time': grouped.apply(lambda x: np.average(np.sort(x), weights=weights_array[: len(x)]))}
                )
        else:
            colums = raw_colums_user02 if 'user02' in exp else raw_colums  # TODO: check Thais_Camacho
            df = pd.read_csv(exp, header=None, names=colums, usecols=colums[1:], na_values=['na'], dtype=colums_type)
            df.dropna(inplace=True)
            df = df[df['rank'] == 0].reset_index(drop=True)  # pylint: disable=E1136
        return df
    except Exception as e:
        warning('Exception(', e, ') reading file: ', exp)
    return pd.DataFrame()


def plot_charts(charts_dir, user, app_name, instance_name, instance_data, ignore, window, selected='', savefig=True):
    if len(instance_data) < 2:
        warning('Only one experiment, nothing to plot...')
        return
    experim_alias = f'{EXPERIM_ALIASES[app_name]} - {instance_name}'
    verbose(f'Ploting {experim_alias} / {app_name} - {selected}', level=3)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=400)
    # fig, (ax1) = plt.subplots(1, 1, figsize=(10,7), dpi=400)
    isntance_color_order_list = plot_raw_chart(experim_alias, instance_data, ignore, ax1)
    plot_relative_chart(experim_alias, instance_data, window, ignore, ax2, None, isntance_color_order_list)

    handles, labels = ax2.get_legend_handles_labels()
    # labels = [f'$\\textbf{{{label}}}$' if label == selected else label for label in labels]
    labels = [f'*{label}*' if label == selected else label for label in labels]

    # Set font properties based on the condition
    # font_prop = FontProperties(weight='bold' if label == selected else 'normal')

    date_max_name = max([len(i) for i in instance_data])
    ncol = 6
    if date_max_name < 13:
        ncol = 11
    elif date_max_name < 16:
        ncol = 9
    fig.legend(
        handles=handles,
        labels=labels,
        loc='upper center',
        bbox_to_anchor=(0.1, 0.95, 0.8, 0.11),
        ncol=ncol,
        mode='expand',
        fancybox=True,
        shadow=True,
        # prop=font_prop,
    )

    # Clone x-axis indices from the second subplot to the first subplot
    tick_positions = ax2.get_xticks()
    tick_labels = [str(int(pos + 1)) for pos in tick_positions]
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(tick_positions))
        ax.set_xticklabels(tick_labels)
        ax.set_xlabel('Iteration')
    ax1.set_ylabel('Execution time (ms)')
    ax2.set_ylabel('Relative performance')
    if not savefig:
        plt.show()
        return
    app_user = (
        '{}_{}_{}_{}'.format(app_name, instance_name, selected, user.replace('/', '_'))
        .replace('-', '_')
        .replace(' ', '')[:115]
    )
    filename = f'{app_user}_{window}_{ignore}.pdf'
    fig.savefig(f'{charts_dir}/{filename}', bbox_inches='tight')
    plt.close(fig)
    verbose(f'Save at: {charts_dir}/{filename}', level=2)


def seek_result(multiple_execs):
    experiment_sum = {}
    for exp in multiple_execs:
        df = raw_read_rank_0(exp)
        if df.empty:
            continue
        experiment_sum[df['time'].sum()] = df, exp
    if not experiment_sum:
        try:
            warning('*** Any valid result for ' + ' - '.join(multiple_execs[0].split('/')[-4:-2]))
        except:
            warning('*** Any valid result found...')
        return pd.DataFrame(), ''
    # Retrieve the corresponding dataframe and csv path
    sorted_sums = sorted(list(experiment_sum.keys()))
    # selected_idx = int(len(sorted_sums) > 1)  # 1 if more than 1
    selected_idx = (len(sorted_sums) - 1) // 2  # median (lower)
    selected_sum = sorted_sums[selected_idx]
    dataframe, csv_path = experiment_sum[selected_sum]
    # dataframe, csv_path = experiment_sum[str(np.median(sum_list))]
    verbose(f'Selecting instance: {selected_sum} - {csv_path}', level=4)
    return dataframe, csv_path


def get_user_apps(csv_data_location, user):
    selected_path_l = list(filter(lambda x: user in x, csv_data_location))
    if len(selected_path_l) != 1:
        # Olhar para os "dastaset"
        raise Exception(f'Deu ruim aqui, mano {user}')
    selected_path = selected_path_l[0]
    applications = glob.glob(os.path.join(selected_path, '*/'))
    user_apps = list(map(lambda x: x.split('/')[-2], applications))
    return selected_path, sorted(user_apps)


def process_user_data(csv_data_location, charts_dir, user, ignore, window):
    selected_path, user_apps = get_user_apps(csv_data_location, user)
    for bench_name in user_apps:
        verbose(f'Processing {user} benchmark {bench_name}', level=1)
        for instance_path in glob.glob(os.path.join(selected_path, bench_name, '*', '*', '')):
            path_parts = Path(instance_path).parts
            instance_name = path_parts[-1]
            app_name = path_parts[-2]
            multiple_execs = glob.glob(os.path.join(instance_path, '*', '*.csv'))
            dataframe, csv_path = seek_result(multiple_execs)
            if dataframe.empty:
                warning(f'Any {bench_name} data\n\t{multiple_execs}')
                continue
            selected = ''
            instance_data = dict()
            for count, exp in enumerate(multiple_execs, 1):
                df = raw_read_rank_0(exp)
                if df.empty:
                    continue
                df.index = df.index + 1  # Set it=1 to the fisrt iteration
                exp_name = f'Exp. {count}'
                instance_data[exp_name] = df
                if csv_path == exp:
                    selected = exp_name
            dataset_lens = set([len(a) for a in instance_data.values()])
            if len(dataset_lens) != 1:
                warning(f'Inconsistent dataset_lens: {dataset_lens}')
            if len(instance_data) < 2:
                warning(f'Just a single execution for instance {instance_name}')
                break
            plot_charts(charts_dir, user, app_name, instance_name, instance_data, ignore, window, selected)


def select_right_experiment(csv_data_location, csv_output_dir, user):
    selected_path, user_apps = get_user_apps(csv_data_location, user)
    for bench_name in user_apps:
        verbose(f'Processing {user} benchmark {bench_name}', level=1)
        for instance_path in glob.glob(os.path.join(selected_path, bench_name, '*', '*', '')):
            path_parts = Path(instance_path).parts
            instance_name = path_parts[-1]
            dataset_id = instance_name.split('-')[-1]
            app_name = f'{path_parts[-2]}-{dataset_id}' if '-' in instance_name else path_parts[-2]
            output_path = os.path.join(csv_output_dir, bench_name, user, app_name)
            os.makedirs(output_path, exist_ok=True)
            multiple_execs = glob.glob(os.path.join(instance_path, '*', '*.csv'))
            dataframe, csv_path = seek_result(multiple_execs)
            if dataframe.empty:
                warning(f'Any {bench_name} data\n\t{multiple_execs}')
                continue
            result_filename = os.path.join(output_path, f'{instance_name}.csv')

            if user.lower() == 'user01':
                verbose(f'Coping from {csv_path} to {result_filename}', level=2)
                shutil.copyfile(csv_path, result_filename)
            else:
                # Note that this dataset contains only the Rank0
                verbose(f'Generating {result_filename}', level=2)
                dataframe.to_csv(result_filename)

            extra_info_file = f'{os.path.splitext(csv_path)[0]}_info.json'
            if os.path.exists(extra_info_file):
                verbose(f'Coping extra info file {extra_info_file}', level=3)
                shutil.copyfile(extra_info_file, f'{os.path.splitext(result_filename)[0]}_info.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Adding optional argument
    parser.add_argument('-i', '--input_dir', help='Input directory')
    parser.add_argument('-v', '--verbosity', help='Verbosity level: 0 (default), 1, 2, 3, 4')
    parser.add_argument('--csv_data_dir', help='Location to store the CSV resulting files')
    parser.add_argument('--charts_dir', help='Directory to store charts when performing analysis per application')
    parser.add_argument('-s', '--ignore', help='chart: Number of iterations to be ignored', default=0)
    parser.add_argument('-w', '--window', help='chart: Number of iterations to be groupoed (smooth)', default=0)

    # Read arguments from command line
    args = parser.parse_args()

    if args.verbosity:
        verbosity_level = int(args.verbosity)

    if not args.input_dir:
        error('Input directory expected but not provided (-i)')

    csv_data_location = glob.glob(os.path.join(args.input_dir, '*'))
    usernames = list(set(map(lambda x: os.path.basename(x), csv_data_location)))
    usernames.sort()

    if not usernames:
        error(f'Any valid result fount at {args.input_dir}')
    verbose(f'Processing input files for users: {usernames}', level=1)

    if args.csv_data_dir:
        if not os.path.exists(args.csv_data_dir):
            warning(f'Creating CSV output directory: {args.csv_data_dir}')
            os.makedirs(args.csv_data_dir)
        # Select right CSV experiment
        for user in usernames:
            verbose(f'Processing CSV input files for user {user}', level=2)
            select_right_experiment(csv_data_location, args.csv_data_dir, user)

    if args.charts_dir:
        if not os.path.exists(args.charts_dir):
            warning(f'Creating output directory: {args.charts_dir}')
            os.makedirs(args.charts_dir)
        # Generate the mult-exec charts
        for user in usernames:
            verbose(f'Generating comparison charts for user {user}', level=2)
            process_user_data(csv_data_location, args.charts_dir, user, args.ignore, args.window)
    elif args.ignore or args.window:
        warning('Arguments "--ignore" and "--window" are only suported for charts!')
