#!/usr/bin/env python3

# Description: This script is used to parse and extract log information from the CSV files.
# The parsed information is organized in a dictionary and stored on the output file.

import argparse
import glob
import json
import os
import pickle

import numpy as np
import pandas as pd

from utils.instance_aliases import INSTANCE_ALIASES
from utils.instance_prices import INSTANCE_PRICES


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


# =============================================================================================
# Functions to extract data and/or summarize the dataframes built from the CSV files.
# =============================================================================================


def mean_range(df, start, end):
    df2 = df[start:end]
    return {'mean': float(df2.mean()), 'sum': float(df2.sum()), 'size': df2.size}


def wait_its_millis(df, ignore_it, consider_ms):
    # Wait at least *X* iterations, consider at least *Y* milliseconds
    # print(f'wait_its_millis\tignore_it:{ignore_it}\tconsider_ms:{consider_ms}')
    it_end = ignore_it
    exec_time = 0
    while exec_time <= consider_ms and len(df) > it_end + 1:
        it_end += 1
        exec_time += df[it_end]
        # print(f'it_end:{it_end}\texec_time:{exec_time}')
    df2 = df[ignore_it:it_end]
    # print(f'wait_its_millis[{ignore_it}:{it_end}]\tWaiting: {df2.sum()}ms\tignoring: {df[:ignore_it].sum()}ms\n')
    return {'mean': float(df2.mean()), 'sum': float(df2.sum()), 'size': df2.size}


def wait_millis(df, ignore_ms, consider_ms):
    # Wait at least *X* milliseconds, consider at least *Y* milliseconds
    # print(f'wait_millis\tignore_ms:{ignore_ms}\tconsider_ms:{consider_ms}')
    exec_time = 0
    it_start = 0
    # Ignore fisrt part of the execution
    while exec_time <= ignore_ms and len(df) > it_start:
        exec_time += df[it_start]
        it_start += 1
        # print(f'it_start:{it_start}\texec_time:{exec_time}')
    # print(f'Start[{it_start}]\tWaiting: {df[:it_start].sum()}s')
    it_end = it_start + 1
    while exec_time <= ignore_ms + consider_ms and len(df) > it_end:
        exec_time += df[it_end]
        it_end += 1
        # print(f'it_end:{it_end}\texec_time:{exec_time}')
    # print(f'End[{it_end}]\t\tComputed time {df[it_start:it_end].sum()}\t\tTotal exec time {df[:it_end].sum()}')
    df2 = df[it_start:it_end]
    # print(f'wait_millis\tit_start:{it_start}\tit_end:{it_end}\texec_time:{df2.sum()}\tignoring: {df[:it_start].sum()}ms\n')
    return (df2.mean(), df2.sum(), df2.size)


proxy_set = {}
proxy_set['Real'] = lambda df: {'mean': float(df.mean()), 'sum': float(df.sum()), 'size': float(df.size)}, []
proxy_set['Second PI'] = lambda df: {'mean': float(df[1]), 'sum': float(df[1]), 'size': 1}, []
proxy_set['From 2 to 5'] = mean_range, [1, 5]
proxy_set['From 2 to 10'] = mean_range, [1, 10]
proxy_set['0.5_s'] = wait_its_millis, [0, 500]
proxy_set['0.5_s-first'] = wait_its_millis, [1, 500]
# proxy_set['0.5_s-5_first'] = wait_its_millis, [5, 500]
# proxy_set['0.5_s-10ms'] = wait_millis, [10, 500]
# proxy_set['0.5_s-50ms'] = wait_millis, [50, 500]
proxy_set['First 32'] = mean_range, [0, 32]
proxy_set['First 64'] = mean_range, [0, 64]

# =============================================================================================


# =============================================================================================
def parse_instance_dataframe(
    instance_name, df, data, time_conversion_factor, PI_time_col='time', ABS_time_col='abs_time', extra_info=None
):
    df[PI_time_col] = df[PI_time_col] * time_conversion_factor

    # Extract the wallclock time (abs_time)
    if ABS_time_col in df.columns:
        df[ABS_time_col] = df[ABS_time_col] * time_conversion_factor
        wallclock_time = df.groupby('rank')[ABS_time_col].max().max()
        data['wallclock_time'] = float(wallclock_time)
    elif extra_info and 'Time in seconds' in extra_info:
        data['wallclock_time'] = float(extra_info['Time in seconds']) * 1000  # Convert sec to msec

    inst_name, inst_count = instance_name.split('-')
    inst_price = INSTANCE_PRICES[inst_name]
    data['Instance Price'] = inst_price
    data['Instance Name'] = inst_name
    data['Instance Count'] = int(inst_count)

    verbose(f'Instance Price: {inst_price}', 5)
    verbose(f'Instance Name:  {inst_name}', 5)
    verbose(f'Instance Count: {inst_count}', 5)

    data['Total PI Samples'] = samples_total = len(df)
    verbose(f'Total PI Samples registered  : {samples_total}', 5)

    if 'rank' in df:
        df = df[df['rank'] == 0]
    elif extra_info and 'Total processes' in extra_info:
        # TODO: Jeff: Fazer algo parecido com o da selecao no pre-processamento(raw_read_rank_0)??
        num_processes = int(extra_info['Total processes'])
        # df = df[df.index % num_processes == int(num_processes/2)]  # Assuming that the PIs are evenly distributed
        df = pd.DataFrame(
            {'time': df.groupby(df.index // num_processes)['time'].mean()}
        )  # Assuming that Rank0 is the mean of every X iterations

    data['PI Samples rank0'] = samples_rank0 = len(df)
    verbose(f'PI Samples registerd in rank0: {samples_rank0}', 5)
    verbose(f'Total / rank0 PI Samples     : {samples_total/samples_rank0}', 5)

    df = df[PI_time_col].reset_index(drop=True)
    # Extract all the proxy information from the dataframe
    for proxy, operations in proxy_set.items():
        data[proxy] = operations[0](df, *operations[1])
        # print("Proxy:", proxy, "--", data[proxy])


# =============================================================================================


# =============================================================================================
# Parse data generated by user
def parse_user_data(user, parsed_data, csv_files):
    if user in parsed_data:
        warning(f'Already parsed user: {user} -- Skipping it!!!')
        return
    parsed_data['Users'][user] = {}

    verbose(f'Processing CSV files from user: {user}', 2)
    user_csv_files = list(filter(lambda x: user in x, csv_files))

    # Parse User apps
    user_apps = list(set(map(lambda x: x.split('/')[-2], user_csv_files)))
    user_apps.sort()

    parsed_data['Users'][user]['apps'] = {}
    # For each user application
    for app_name in user_apps:
        verbose(f'Processing app {app_name} from user: {user}', 3)

        if app_name in parsed_data['Users'][user]['apps']:
            warning(f'WARNING: App {app_name} already parsed for user {user} -- Skipping it!!!')
            continue
        parsed_data['Users'][user]['apps'][app_name] = {}

        # List of CSV files for user / app
        user_app_csv_files = list(filter(lambda x: app_name in x, user_csv_files))

        for instance_csv_file in user_app_csv_files:
            dataset_name = 'generic'
            if '-' in app_name:
                dataset_name = f'group-{app_name.split('-')[-1]}'

            if dataset_name not in parsed_data['Users'][user]['apps'][app_name]:
                parsed_data['Users'][user]['apps'][app_name][dataset_name] = {}

            # Read experiment CSV file
            instance_name = INSTANCE_ALIASES[instance_csv_file.split('/')[-1].replace('.csv', '')]
            verbose(f'Processing experiment ({dataset_name} {instance_name}) {instance_csv_file}', 4)

            if instance_name in parsed_data['Users'][user]['apps'][app_name][dataset_name]:
                warning(
                    f'WARNING: Instance {instance_name} already parsed for app {app_name} / user {user} -- Skipping it!!!'
                )
                continue

            # Store user/app/experiment information
            parsed_data['Users'][user]['apps'][app_name][dataset_name][instance_name] = {
                'csv_filename': instance_csv_file
            }

            # Parse CSV file
            df = pd.read_csv(instance_csv_file, dtype=np.float64)

            if df.size == 0:
                warning(f'Could not extract information from CSV file: {instance_csv_file}')
                continue
            if len(df.keys()) == 1:
                df.columns = ['time']

            # FIX results from user01's experiments: Convert time from sec to msec
            time_conversion_factor = 1000  # Convert sec to msec
            if any(name in instance_csv_file.lower() for name in ['user01']):
                time_conversion_factor = 0.001  # Convert usec to msec

            # Apparently, the results collected by 'user02' switched columns abs_time and time. Adjusting for this case.
            if user == 'user02':
                PI_time_col = 'abs_time'
                ABS_time_col = 'time'
            else:
                PI_time_col = 'time'
                ABS_time_col = 'abs_time'

            extra_info = None
            extra_info_file = instance_csv_file.replace('.csv', '_info.json')
            if os.path.exists(extra_info_file):
                with open(extra_info_file, 'r') as file:
                    extra_info = json.load(file)

            parse_instance_dataframe(
                instance_name,
                df,
                parsed_data['Users'][user]['apps'][app_name][dataset_name][instance_name],
                time_conversion_factor,
                PI_time_col,
                ABS_time_col,
                extra_info,
            )


# =============================================================================================


if __name__ == '__main__':
    # Initialize parser
    parser = argparse.ArgumentParser()

    # Adding optional argument
    parser.add_argument('-i', '--input_dir', help='Input directory')
    parser.add_argument('-o', '--output_file', help='Output file (.pkl)')
    parser.add_argument('-v', '--verbosity', help='Verbosity level: 0 (default), 1, 2, 3, 4')

    # Read arguments from command line
    args = parser.parse_args()

    if args.verbosity:
        verbosity_level = int(args.verbosity)

    if not args.input_dir:
        error('Input directory expected but not provided (-i)')

    if not args.output_file:
        error('Output filename expected but not provided (-o)')

    if not os.path.exists(args.input_dir):
        error(f'{args.input_dir} is an invalid directory!')

    # CSV files
    verbose(f'Parsing files from {args.input_dir}', 1)
    csv_files = glob.glob(args.input_dir + '/*/*/*/*.csv', recursive=True)

    # User names
    usernames = list(set(map(lambda x: x.split('/', 4)[-3], csv_files)))
    usernames.sort()
    verbose('Usernames:' + str(usernames), 2)

    parsed_data = {'Users': {}}
    for user in usernames:
        parse_user_data(user, parsed_data, csv_files)
        # break

    verbose(f'Storing results at {args.output_file}', 1)
    with open(args.output_file, 'wb') as file:
        pickle.dump(parsed_data, file)
