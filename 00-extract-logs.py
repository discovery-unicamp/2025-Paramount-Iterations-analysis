#!/usr/bin/env python3

# Description: This script is used to parse and extract log information from the CSV files.
# The parsed information is organized in a dictionary and stored on the output file.

import os
import glob
import numpy as np
from datetime import date
import pandas as pd

from instance_aliases import INSTANCE_ALIASES
from instance_prices import INSTANCE_PRICES

#=============================================================================================
# Functions to extract data and/or summarize the dataframes built from the CSV files.
#=============================================================================================

def mean_range(df, start, end):
    df2 = df[start:end]
    return {"mean":float(df2.mean()), "sum":float(df2.sum()), "size":df2.size}

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
    return {"mean":float(df2.mean()), "sum":float(df2.sum()), "size":df2.size}


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
proxy_set['Real'] = lambda df: {"mean":float(df.mean()), "sum":float(df.sum()), "size":float(df.size)}, []
proxy_set['Second PI'] = lambda df: {"mean":float(df[1]), "sum":float(df[1]), "size":1}, []
proxy_set['From 2 to 5'] = mean_range, [1, 5]
proxy_set['From 2 to 10'] = mean_range, [1, 10]
proxy_set['0.5_s'] = wait_its_millis, [0, 500]
proxy_set['0.5_s-first'] = wait_its_millis, [1, 500]
# proxy_set['0.5_s-5_first'] = wait_its_millis, [5, 500]
# proxy_set['0.5_s-10ms'] = wait_millis, [10, 500]
# proxy_set['0.5_s-50ms'] = wait_millis, [50, 500]
proxy_set['First 32'] = mean_range, [0, 32]
proxy_set['First 64'] = mean_range, [0, 64]

#=============================================================================================

#=============================================================================================
def parse_instance_dataframe(instance_name, df, data, time_conversion_factor, PI_time_col = 'time', ABS_time_col = 'abs_time'):

    df[PI_time_col] = df[PI_time_col]*time_conversion_factor

    # Extract the wallclock time (abs_time)
    if ABS_time_col in df.columns:
        df[ABS_time_col] = df[ABS_time_col]*time_conversion_factor
        # Jeferson, não deveria ser o máximo de cada rank?
        wallclock_time = df.groupby('rank')[ABS_time_col].max().max()
        data["wallclock_time"] = float(wallclock_time)

    inst_name, inst_count = instance_name.split('-')
    inst_price = INSTANCE_PRICES[inst_name]
    data["Instance Price"] = inst_price
    data["Instance Name"]  = inst_name
    data["Instance Count"] = inst_count

    verbose(f'Instance Price: {inst_price}', 5)
    verbose(f'Instance Name:  {inst_name}', 5)
    verbose(f'Instance Count: {inst_count}', 5)

    if 'rank' not in df.keys():
        data["PI Samples rank0"] = len(df)
    else:        
        data["PI Samples rank0"] = len(df[df['rank'] == 0])
        df = df[df["rank"] == 0]

    verbose(f'PI Samples registerd in rank0: {data["PI Samples rank0"]}', 5)
    verbose(f'Total PI Samples registered  : {len(df)}', 5)
    verbose(f'Total / rank0 PI Samples     : {len(df)/data["PI Samples rank0"]}', 5)

    df = df[PI_time_col].reset_index(drop=True)
    # Extract all the proxy information from the dataframe
    for proxy, operations in proxy_set.items():
        data[proxy] = operations[0](df, *operations[1])
        #print("Proxy:", proxy, "--", data[proxy])

#=============================================================================================

#=============================================================================================
# Parse data generated by user
def parse_user_data(user, parsed_data, csv_files):

    if user in parsed_data:
        warning(f"Already parsed user: {user} -- Skipping it!!!")
        return
    else:
        parsed_data["Users"][user] = {}

    verbose(f'Processing CSV files from user: {user}',2)
    user_csv_files = list(filter(lambda x: user in x, csv_files))

    # Parse User apps
    user_apps = list(set(map(lambda x: x.split('/')[-4], user_csv_files)))
    user_apps.sort()

    parsed_data["Users"][user]["apps"] = {}
    # For each user application
    for app_name in user_apps:

        verbose(f'Processing app {app_name} from user: {user}',3)

        if app_name in parsed_data["Users"][user]["apps"]:
            warning(f"WARNING: App {app_name} already parsed for user {user} -- Skipping it!!!")
            continue
        else:
            parsed_data["Users"][user]["apps"][app_name] = {}

        # print(f'Processing app {app_name}')

        # List of CSV files for user / app
        user_app_csv_files = list(filter(lambda x: app_name in x, user_csv_files))

        for instance_csv_file in user_app_csv_files:

            dataset_name = instance_csv_file.split('/')[-2]

            if dataset_name not in parsed_data["Users"][user]["apps"][app_name]:
                parsed_data["Users"][user]["apps"][app_name][dataset_name] = {}

            # Read experiment CSV file
            instance_name = INSTANCE_ALIASES[instance_csv_file.split('/')[-1].replace('.csv', '')]
            verbose(f'Processing experiment ({dataset_name} {instance_name}) {instance_csv_file}', 4)

            if instance_name in parsed_data["Users"][user]["apps"][app_name][dataset_name]:
                warning(f"WARNING: Instance {instance_name} already parsed for app {app_name} / user {user} -- Skipping it!!!")
                continue

            # Store user/app/experiment information
            parsed_data["Users"][user]["apps"][app_name][dataset_name][instance_name] = {
                "csv_filename": instance_csv_file
            }

            # Parse CSV file
            df = pd.read_csv(instance_csv_file, dtype=np.float64)
            
            if df.size == 0:
                warning(f'Could not extract information from CSV file: {instance_csv_file}')
                continue
            
            # FIX results from Jeferson's experiments: Convert time from sec to msec
            time_conversion_factor = 1000  # Convert sec to msec
            if any(name in instance_csv_file.lower() for name in ['jeferson']):
                time_conversion_factor = 0.001  # Convert usec to msec

            # Apparently, the results collected by Thais_Camacho switched columns abs_time and time. Adjusting for this case.
            if user == "Thais_Camacho":
                PI_time_col = 'abs_time'
                ABS_time_col = 'time'
            else:
                PI_time_col = 'time'
                ABS_time_col = 'abs_time'

            parse_instance_dataframe(instance_name, 
                                     df, 
                                     parsed_data["Users"][user]["apps"][app_name][dataset_name][instance_name], 
                                     time_conversion_factor,
                                     PI_time_col, ABS_time_col)
#=============================================================================================

# ====================================================
# Utility functions
def error(msg):
    print("ERROR:", msg)
    exit(1)

def warning(msg):
    print("WARNING:", msg)

verbosity_level = 0
def verbose(msg, level=0):
    if level <= verbosity_level:
        print(" "*(level), msg)
# ====================================================

if __name__ == '__main__':

    import argparse
    import pickle

    # Initialize parser
    parser = argparse.ArgumentParser()

    # Adding optional argument
    parser.add_argument("-i", "--input_dir", help = "Input directory")
    parser.add_argument("-o", "--output_file", help = "Output file (.pkl)")
    parser.add_argument("-v", "--verbosity", help = "Verbosity level: 0 (default), 1, 2, 3, 4")

    # Read arguments from command line
    args = parser.parse_args()

    if args.verbosity:
        verbosity_level = int(args.verbosity)

    if not args.input_dir:
        error("Input directory expected but not provided (-i)")

    if not args.output_file:
        error("Output filename expected but not provided (-o)")

    if not os.path.exists(args.input_dir):
        error(f"{args.input_dir} is an invalid directory!")

    root_data_dir = args.input_dir
    verbose(f"Parsing files from {root_data_dir}",1)

    # CSV files
    csv_files = glob.glob(root_data_dir+'/*/*/*/*.csv', recursive=True)

    # User names
    usernames = list(set(map(lambda x: x.split('/', 4)[-3], csv_files)))
    usernames.sort()
    verbose("Usernames:"+str(usernames), 2)

    parsed_data = { "Users": {} }
    for user in usernames:
        parse_user_data(user, parsed_data, csv_files)
        # break

    verbose(f"Storing results at {args.output_file}",1)
    file = open(args.output_file, 'wb')
    pickle.dump(parsed_data, file)
    file.close()
