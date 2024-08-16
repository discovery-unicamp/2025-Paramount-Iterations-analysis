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
# Functions to dump data
#=============================================================================================

def abbreviate(s, max_sz=20):
    if len(s) <= max_sz: 
        return s
    else: 
        half = max_sz//2
        return s[0:half]+"..."+s[-half:]

def print_in_line(v):
    if (type(v) is str) or (type(v) is int) or (type(v) is float): 
        return True
    else:
        #print(type(v))
        return False

def print_object(o, prefix=""):
    new_prefix = prefix+" "
    if type(o) is dict:
        for k, v in o.items():
            if print_in_line(v):
                print(prefix+abbreviate(str(k))+":",v)
            else:
                print(prefix+abbreviate(str(k))+":")
                print_object(v, new_prefix)
    elif type(o) is list:
        print(prefix+"L:",o)
    else:
        print(prefix+abbreviate(str(o)))

# ====================================================
# Utility functions
def error(msg):
    print("ERROR:", msg)
    exit(1)

W_show_csv_filename = False
def warning(msg, csv_filename=None):
    if W_show_csv_filename and csv_filename:
        print("WARNING:", msg, f"({csv_filename})")
    else:
        print("WARNING:", msg)


verbosity_level = 0
def verbose(msg, level=0):
    if level <= verbosity_level:
        print(msg)
# ====================================================

# ====================================================
def wallclock_time_sanity_check(data):
    
    for user, user_data in data["Users"].items():
        verbose("+- "+str(user),1)
        for app, usr_app_data in user_data["apps"].items():
            verbose("|  +- "+str(app),2)
            for ds, usr_app_ds_data in usr_app_data.items():
                verbose("|  |  +- "+str(ds),3)
                for instance, usr_app_ds_instance_data in usr_app_ds_data.items():
                    csv_filename = usr_app_ds_instance_data["csv_filename"]
                    verbose("|  |  |  +- "+str(instance),4)
                    verbose("|  |  |  |  +- PIs sum  : "+str(usr_app_ds_instance_data["Real"]["sum"]),5)
                    if not "wallclock_time" in usr_app_ds_instance_data:
                        warning(f"(missing wallclock_time) {user} - {app} - {ds} - {instance} ",csv_filename)
                    else:
                        # Has wall_clock time
                        wallclock_time = float(usr_app_ds_instance_data["wallclock_time"])
                        sum_of_PIs = float(usr_app_ds_instance_data["Real"]["sum"])
                        verbose("|  |  |  |  +- Wallclock: "+str(wallclock_time),5)
                        if wallclock_time < sum_of_PIs:
                            warning(f"(wallclock time smaller than sum of PIs) {user} - {app} - {ds} - {instance} - wallclock_time: {wallclock_time}, sum of PIs: {sum_of_PIs}",csv_filename)
                        else:
                            PIs_wallclock_ratio = sum_of_PIs / wallclock_time
                            if PIs_wallclock_ratio < 0.90:
                                warning(f"(sum_of_PIs / wallclock time ratio is {PIs_wallclock_ratio:.2f}) -- {user} - {app} - {ds} - {instance}",csv_filename)
# ====================================================
def print_apps(data):    
    for user, user_data in data["Users"].items():
        verbose("+- "+str(user),1)
        for app, usr_app_data in user_data["apps"].items():
            print(f"  \"{app}\":  "",")
# ====================================================

def wallclock_time_sanity_check_by_app(data):

    # Reorganize data: app group -> app -> user -> dataset -> instance -> ...
    import app_group    
    app_data = {}
    for user, user_data in data["Users"].items():
        for app, usr_app_data in user_data["apps"].items():
            group = app_group.app_group[app]
            if not group in app_data:
                app_data[group] = {}
            if not app in app_data[group]:
                app_data[group][app] = {}
            if user in app_data[group][app]:
                warning(f"USER {user} already in app_data[{group}][{app}]")
            else:
                app_data[group][app][user] = usr_app_data

    for group, group_data in app_data.items():
        verbose("+- "+str(group),1)
        for app, app_data in group_data.items():
            verbose("|  +- "+str(app),2)
            for user, user_data in app_data.items():
                verbose("|  |  +- "+str(user),3)
                for ds, usr_app_ds_data in user_data.items():
                    verbose("|  |  |  +- "+str(ds),4)
                    for instance, usr_app_ds_instance_data in usr_app_ds_data.items():
                        csv_filename = usr_app_ds_instance_data["csv_filename"]
                        verbose("|  |  |  |  +- "+str(instance),4)
                        verbose("|  |  |  |  |  +- PIs sum  : "+str(usr_app_ds_instance_data["Real"]["sum"]),5)
                        if not "wallclock_time" in usr_app_ds_instance_data:
                            warning(f"(missing wallclock_time) {user} - {app} - {ds} - {instance} ",csv_filename)
                        else:
                            # Has wall_clock time
                            wallclock_time = float(usr_app_ds_instance_data["wallclock_time"])
                            sum_of_PIs = float(usr_app_ds_instance_data["Real"]["sum"])
                            verbose("|  |  |  |  |  +- Wallclock: "+str(wallclock_time),5)
                            if wallclock_time < sum_of_PIs:
                                warning(f"(wallclock time smaller than sum of PIs) {user} - {app} - {ds} - {instance} - wallclock_time: {wallclock_time}, sum of PIs: {sum_of_PIs}",csv_filename)
                            else:
                                PIs_wallclock_ratio = sum_of_PIs / wallclock_time
                                if PIs_wallclock_ratio < 0.90:
                                    warning(f"(sum_of_PIs / wallclock time ratio is {PIs_wallclock_ratio:.2f}) -- {user} - {app} - {ds} - {instance}",csv_filename)

# ====================================================

def generate_csv_analysis_per_instance(data):

    proxy_metrics = ["Second PI", "From 2 to 5", "From 2 to 10", "0.5_s", "0.5_s-first"]
    csv_fields =  ["group", "app", "user", "dataset", "instance", "wallclock_time", 
                "PIs sum", "PIs/wallclock_time"] + proxy_metrics + ["csv_filename", "warnings"]

    def print_row(row_data):
        for field in csv_fields:
            print(row_data[field], end=",")
        print()

    # Print header
    for field in csv_fields:
        print(field, end=",")
    print()

    # Reorganize data: app group -> app -> user -> dataset -> instance -> ...
    import app_group    
    app_data = {}
    for user, user_data in data["Users"].items():
        for app, usr_app_data in user_data["apps"].items():
            group = app_group.app_group[app]
            if not group in app_data:
                app_data[group] = {}
            if not app in app_data[group]:
                app_data[group][app] = {}
            if user in app_data[group][app]:
                warning(f"USER {user} already in app_data[{group}][{app}]")
            else:
                app_data[group][app][user] = usr_app_data

    # Process reorganized data
    row_data = {}
    for group, group_data in app_data.items():
        verbose("+- "+str(group),1)
        row_data["group"] = group
        for app, app_data in group_data.items():
            verbose("|  +- "+str(app),2)
            row_data["app"] = app
            for user, user_data in app_data.items():
                verbose("|  |  +- "+str(user),3)
                row_data["user"] = user
                for ds, usr_app_ds_data in user_data.items():
                    verbose("|  |  |  +- "+str(ds),4)
                    row_data["dataset"] = ds
                    for instance, usr_app_ds_instance_data in usr_app_ds_data.items():
                        verbose("|  |  |  |  +- "+str(instance),4)
                        row_data["warnings"] = ""
                        row_data["instance"] = instance
                        csv_filename = usr_app_ds_instance_data["csv_filename"]
                        row_data["csv_filename"] = csv_filename
                        row_data["PIs sum"] = usr_app_ds_instance_data["Real"]["sum"]
                        for proxy_metric in proxy_metrics:
                            row_data[proxy_metric] = usr_app_ds_instance_data[proxy_metric]["mean"]

                        if not "wallclock_time" in usr_app_ds_instance_data:
                            row_data["wallclock_time"] = ""
                            row_data["PIs/wallclock_time"] = ""
                            row_data["warnings"] += "(missing wallclock_time) "
                        else:
                            # Has wall_clock time
                            wallclock_time = float(usr_app_ds_instance_data["wallclock_time"])
                            row_data["wallclock_time"] = str(wallclock_time)
                            sum_of_PIs = float(usr_app_ds_instance_data["Real"]["sum"])
                            PIs_wallclock_ratio = sum_of_PIs / wallclock_time
                            row_data["PIs/wallclock_time"] = str(PIs_wallclock_ratio)
                            if wallclock_time < sum_of_PIs:
                                row_data["warnings"] += "(wallclock time smaller than sum of PIs) "
                            else:
                                if PIs_wallclock_ratio < 0.90:
                                    row_data["warnings"] += f"(sum_of_PIs / wallclock time ratio is {PIs_wallclock_ratio:.2f}) "
                        print_row(row_data)

# ====================================================

import matplotlib.pyplot as plt
import statistics

def plot_correlation(X_values, 
                     X_label,
                     Y_values, 
                     Y_label,
                     user, 
                     app_name, 
                     ds,
                     instance_names, 
                     filename_suffix, 
                     plot_ideal,
                     charts_output_directory):
    
    if len(instance_names) < 3:
        print(f'WARNING!!! Not enough instances to plot a correlation {user}: {app_name}')
        return

    fig, ax = plt.subplots(layout='constrained')

    # Compute R^2 value
    correlation_matrix = np.corrcoef(X_values, Y_values)
    correlation_xy = correlation_matrix[0, 1]
    r_squared = correlation_xy**2

    sum_coorelation = sum(X_values)/sum(Y_values)
    min_coorelation = min(X_values)/min(Y_values)
    median_coorelation = statistics.median(X_values)/statistics.median(Y_values)
    
    #print(f'Correlation {filename_suffix} {user} - {app_name} - {ds} (R^2 = {r_squared:.3f}): (sum {sum_coorelation:.3f}) - (min {min_coorelation:.3f}) - (median {median_coorelation:.3f})')

    correlation_factor_str = ''
    # if abs(r_squared - 1.0) < 0.1 and abs(median_coorelation - 1.0) > 0.01:
        # TEST: Fixing correlation by the median_coorelation
        # correlation_factor_str = f' (factor {median_coorelation:.3f})'
        # print(f'Fixing correlation by factor of {median_coorelation:.3f}: {app_name}')
        # Y_values = [value * median_coorelation for value in Y_values]

    # Plot ideal trendline (x=y)
    min_val = min(min(X_values), min(Y_values))
    max_val = max(max(X_values), max(Y_values))

    if plot_ideal:
        plt.plot([min_val, max_val], [min_val, max_val], color='#aaaaaa', linestyle='-', linewidth=1, label='_nolegend_', zorder=0)

    # Plot trendline
    fit = np.polyfit(X_values, Y_values, 1)
    poly = np.poly1d(fit)
    plt.plot([min_val, max_val], poly([min_val, max_val]), color='#ff000070', linestyle='--', linewidth=2, label='_nolegend_', zorder=5)
    # plt.plot(X_values, poly(X_values), color='#ff000050', linestyle='--', linewidth=1.3, label='_nolegend_', zorder=5)
    # plt.plot([min(X_values), max(X_values)], poly([min(X_values), max(X_values)]), color='#ff000050', linestyle=(0, (5, 10)), linewidth=1.3, label='_nolegend_', zorder=5)

    # Plot each point with its corresponding instance name
    for i, (x, y) in enumerate(zip(X_values, Y_values)):
        plt.scatter(x, y, label=instance_names[i], zorder=10)

    plt.xlabel(X_label)
    plt.ylabel(Y_label)

    # Set title including the R^2 value
    plt.title(f'{X_label} vs {Y_label} - $R^2 = {r_squared:.2f}$\n{app_name}-{ds}{correlation_factor_str}')

    fig.legend(loc='outside lower center', ncol=4, fancybox=True, shadow=True, mode="expand")

    basename = f'{user.replace("/", "-")}_{app_name[:20]}-{ds}-{r_squared:.2f}'
    filename = os.path.join(charts_output_directory, f'{basename}-{filename_suffix}.pdf')

    if 0:
        plt.pause(1)  # Display for 1 second
    else:
        plt.savefig(filename)
    plt.close()
    return filename



def generate_csv_analysis_per_application(data, charts_output_directory):
    import scipy

    proxy_metrics = ["Second PI", "From 2 to 5", "From 2 to 10", "0.5_s", "0.5_s-first"]
    proxy_metrics_2 = ["R2*", "R2", "Intercept", "Slope", "Intercept/min PIs sum", "chartname"]
    csv_fields =  ["Idx", "group", "app", "user", "dataset", "# instances", "min wallclock_time", "max wallclock_time", 
                   "min PIs sum", "max PIs sum"] + \
                  [ f"{pm} vs All PIs - {pm2}" for pm in proxy_metrics for pm2 in proxy_metrics_2 ] +\
                  ["warnings"]

    def print_row(row_data):
        for field in csv_fields:
            print(row_data[field], end=",")
        print()

    # Print header
    for field in csv_fields:
        print(field, end=",")
    print()

    # Reorganize data: app group -> app -> user -> dataset -> instance -> ...
    import app_group    
    app_data = {}
    for user, user_data in data["Users"].items():
        for app, usr_app_data in user_data["apps"].items():
            group = app_group.app_group[app]
            if not group in app_data:
                app_data[group] = {}
            if not app in app_data[group]:
                app_data[group][app] = {}
            if user in app_data[group][app]:
                warning(f"USER {user} already in app_data[{group}][{app}]")
            else:
                app_data[group][app][user] = usr_app_data

    # Process reorganized data
    row_data = { "Idx" : 1 }
    for group, group_data in app_data.items():
        verbose("+- "+str(group),1)
        row_data["group"] = group
        for app, app_data in group_data.items():
            verbose("|  +- "+str(app),2)
            row_data["app"] = app
            for user, user_data in app_data.items():
                verbose("|  |  +- "+str(user),3)
                row_data["user"] = user
                for ds, usr_app_ds_data in user_data.items():
                    verbose("|  |  |  +- "+str(ds),4)
                    row_data["dataset"] = ds
                    row_data["warnings"] = ""

                    wall_clock_time_l = []
                    PIs_sum_l = []
                    proxy_metrics_l = { pm : [] for pm in proxy_metrics}
                    instance_names_l = []

                    # proxy_metrics = ["Second PI", "From 2 to 5", "From 2 to 10", "0.5_s", "0.5_s-first"]
                    for instance, usr_app_ds_instance_data in usr_app_ds_data.items():
                        instance_names_l.append(instance)
                        if "wallclock_time" in usr_app_ds_instance_data:
                            wall_clock_time_l.append(float(usr_app_ds_instance_data["wallclock_time"]))
                        PIs_sum_l.append(float(usr_app_ds_instance_data["Real"]["sum"]))
                        for pm in proxy_metrics:
                            proxy_metrics_l[pm].append(float(usr_app_ds_instance_data[pm]["mean"]))

                    row_data["# instances"] = len(PIs_sum_l)

                    # Summarize results
                    if len(wall_clock_time_l) > 0:
                        row_data["min wallclock_time"] = min(wall_clock_time_l)
                        row_data["max wallclock_time"] = max(wall_clock_time_l) 
                    else:
                        row_data["min wallclock_time"] = ""
                        row_data["max wallclock_time"] = ""
                    row_data["min PIs sum"] = min(PIs_sum_l)
                    row_data["max PIs sum"] = max(PIs_sum_l)

                    for pm in proxy_metrics:
                        # Compute correlation between pm and All PIs.
                        if (len(PIs_sum_l) >= 2):
                            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(proxy_metrics_l[pm], PIs_sum_l)
                            row_data[f"{pm} vs All PIs - R2"] = r_value
                            row_data[f"{pm} vs All PIs - Intercept"] = intercept
                            row_data[f"{pm} vs All PIs - Slope"] = slope
                            row_data[f"{pm} vs All PIs - Intercept/min PIs sum"] = intercept / row_data["min PIs sum"]

                            correlation_matrix = np.corrcoef(proxy_metrics_l[pm], PIs_sum_l)
                            correlation_xy = correlation_matrix[0, 1]
                            r_squared = correlation_xy**2
                            row_data[f"{pm} vs All PIs - R2*"] = r_squared

                        else:
                            row_data[f"{pm} vs All PIs - R2"] = ""
                            row_data[f"{pm} vs All PIs - R2*"] = ""
                            row_data[f"{pm} vs All PIs - Intercept"] = ""
                            row_data[f"{pm} vs All PIs - Slope"] = ""
                            row_data[f"{pm} vs All PIs - Intercept/min PIs sum"] = ""
                            row_data["warnings"] = f"(number of samples - {len(PIs_sum_l)} - too small for statistics)"


                    for pm in proxy_metrics:
                        row_data[f"{pm} vs All PIs - chartname"] = ""
                    if (charts_output_directory):
                        # Plot chart 
                        if (len(wall_clock_time_l) >= 3):
                            plot_correlation(X_values=PIs_sum_l, 
                                             X_label="Sum of PIs (ms)",
                                            Y_values=wall_clock_time_l, 
                                            Y_label='Total execution time (ms)',
                                            user = user,
                                            app_name = app,
                                            ds = ds,
                                            instance_names = instance_names_l,
                                            filename_suffix = 'wallclock_vs_sum_pi',
                                            plot_ideal = True,
                                            charts_output_directory = charts_output_directory)

                        if (len(PIs_sum_l) >= 3):
                            for pm in proxy_metrics:
                                filename = plot_correlation(X_values=PIs_sum_l, 
                                                X_label="Sum of PIs (ms)",
                                                Y_values=proxy_metrics_l[pm], 
                                                Y_label=pm+' (ms)',
                                                user = user,
                                                app_name = app,
                                                ds = ds,
                                                instance_names = instance_names_l,
                                                filename_suffix = pm.lower().replace(" ","_") + '_vs_sum_pi',
                                                plot_ideal = False,
                                                charts_output_directory = charts_output_directory)
                                row_data[f"{pm} vs All PIs - chartname"] = filename

                              
                    print_row(row_data)    
                    row_data["Idx"] += 1


# ====================================================

if __name__ == '__main__':

    import argparse
    import pickle

    # Initialize parser
    parser = argparse.ArgumentParser()

    # Adding optional argument
    parser.add_argument("-i", "--input_file", help = "Input pickle file (.pkl) ")
    parser.add_argument("-v", "--verbosity", help = "Verbosity level: 0 (default), 1, 2, 3, 4")
    parser.add_argument("-d", "--dump_data", help="Dump data", action="store_true")
    parser.add_argument("--W_show_csv_filename", help="Show CSV filename on warnings", action="store_true")
    parser.add_argument("--analysis_per_instance", help="Generate CSV table with analysis per instance", action="store_true")
    parser.add_argument("--analysis_per_application", help="Generate CSV table with analysis per application", action="store_true")
    parser.add_argument("--application_charts_dir", help="Directory to store charts when performing analysis per application")

    # Read arguments from command line
    args = parser.parse_args()

    if args.verbosity:
        verbosity_level = int(args.verbosity)

    if args.W_show_csv_filename:
        W_show_csv_filename = True

    if not args.input_file:
        error("Input file expected but not provided (-i)")

    if not os.path.exists(args.input_file):
        error(f"{args.input_dir} is an invalid file!")

    verbose(f"Loading data from  {args.input_file}",1)
    file = open(args.input_file, 'rb')
    data = pickle.load(file)
    file.close()
    
    if args.dump_data:
        print_object(data, ".")

    #wallclock_time_sanity_check(data)
    #print_apps(data)
    #wallclock_time_sanity_check_by_app(data)
    if args.analysis_per_instance:
        generate_csv_analysis_per_instance(data)

    if args.analysis_per_application:
        generate_csv_analysis_per_application(data, args.application_charts_dir)

# ====================================================


