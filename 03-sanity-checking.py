#!/usr/bin/env python3

# Description: This script reads the pickle pre-processed data derive CSVs and charts with them

import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

from utils.app_group import app_group
from utils.colors import COLORS
from utils.experim_aliases import EXPERIM_ALIASES

# =============================================================================================
# Functions to dump data
# =============================================================================================


def abbreviate(s, max_sz=20):
    if len(s) <= max_sz:
        return s
    half = max_sz // 2
    return s[0:half] + '...' + s[-half:]


def print_in_line(v):
    if (type(v) is str) or (type(v) is int) or (type(v) is float):
        return True
    else:
        # print(type(v))
        return False


def print_object(o, prefix=''):
    new_prefix = prefix + ' '
    if type(o) is dict:
        for k, v in o.items():
            if print_in_line(v):
                print(prefix + abbreviate(str(k)) + ':', v)
            else:
                print(prefix + abbreviate(str(k)) + ':')
                print_object(v, new_prefix)
    elif type(o) is list:
        print(prefix + 'L:', o)
    else:
        print(prefix + abbreviate(str(o)))


# ====================================================
# Utility functions
def error(msg):
    print('ERROR:', msg)
    exit(1)


W_show_csv_filename = False


def warning(msg, csv_filename=None):
    if W_show_csv_filename and csv_filename:
        print('WARNING:', msg, f'({csv_filename})')
    else:
        print('WARNING:', msg)


verbosity_level = 0


def verbose(msg, level=0):
    if level <= verbosity_level:
        print(msg)


# def rounded_linspace(start, end, num_points):
#     """
#     Generate evenly spaced, rounded values between `start` and `end` with `num_points` points.

#     Parameters:
#     - start: The starting value of the range.
#     - end: The ending value of the range.
#     - num_points: The number of points to generate.

#     Returns:
#     - A numpy array of rounded values.
#     """
#     # Generate intermediate values
#     values = np.linspace(start, end, num_points)

#     # Calculate the spacing between points
#     spacing = (end - start) / (num_points - 1)

#     # Determine precision based on spacing
#     if spacing > 0:
#         # Calculate the precision based on the magnitude of spacing
#         precision = -int(np.floor(np.log10(spacing)))
#         precision = max(precision, 0)  # Ensure precision is at least 0
#     else:
#         precision = 0

#     # Round values
#     rounded_values = np.round(values, precision)

#     # print(start, end, num_points, precision, '--->',rounded_values)

#     return rounded_values


# ====================================================


costs_subdir = 'costs'
pareto_subdir = 'pareto'

DISCARD_THRESHOLD = 1.2  # Discard values greathar than 20% of the reference
WEIGHT_TIME = 1  # Multiply the proportional time weight
WEIGHT_COST = 1  # Multiply the proportional cost weight


# ====================================================
def wallclock_time_sanity_check(data):
    for user, user_data in data['Users'].items():
        verbose('+- ' + str(user), 1)
        for app, usr_app_data in user_data['apps'].items():
            verbose('|  +- ' + str(app), 2)
            for ds, usr_app_ds_data in usr_app_data.items():
                verbose('|  |  +- ' + str(ds), 3)
                for instance, usr_app_ds_instance_data in usr_app_ds_data.items():
                    csv_filename = usr_app_ds_instance_data['csv_filename']
                    verbose('|  |  |  +- ' + str(instance), 4)
                    verbose('|  |  |  |  +- PIs sum  : ' + str(usr_app_ds_instance_data['Real']['sum']), 5)
                    if 'wallclock_time' not in usr_app_ds_instance_data:
                        warning(f'(missing wallclock_time) {user} - {app} - {ds} - {instance} ', csv_filename)
                    else:
                        # Has wall_clock time
                        wallclock_time = float(usr_app_ds_instance_data['wallclock_time'])
                        sum_of_PIs = float(usr_app_ds_instance_data['Real']['sum'])
                        verbose('|  |  |  |  +- Wallclock: ' + str(wallclock_time), 5)
                        if wallclock_time < sum_of_PIs:
                            warning(
                                f'(wallclock time smaller than sum of PIs) {user} - {app} - {ds} - {instance} - wallclock_time: {wallclock_time}, sum of PIs: {sum_of_PIs}',
                                csv_filename,
                            )
                        else:
                            PIs_wallclock_ratio = sum_of_PIs / wallclock_time
                            if PIs_wallclock_ratio < 0.90:
                                warning(
                                    f'(sum_of_PIs / wallclock time ratio is {PIs_wallclock_ratio:.2f}) -- {user} - {app} - {ds} - {instance}',
                                    csv_filename,
                                )


# ====================================================
def print_apps(data):
    for user, user_data in data['Users'].items():
        verbose('+- ' + str(user), 1)
        for app, usr_app_data in user_data['apps'].items():
            print(f'  "{app}":  ' ',')


# ====================================================


def wallclock_time_sanity_check_by_app(data):
    # Reorganize data: app group -> app -> user -> dataset -> instance -> ...
    app_data = {}
    for user, user_data in data['Users'].items():
        for app, usr_app_data in user_data['apps'].items():
            group = app_group[app]
            if group not in app_data:
                app_data[group] = {}
            if app not in app_data[group]:
                app_data[group][app] = {}
            if user in app_data[group][app]:
                warning(f'USER {user} already in app_data[{group}][{app}]')
            else:
                app_data[group][app][user] = usr_app_data

    for group, group_data in app_data.items():
        verbose('+- ' + str(group), 1)
        for app, app_data in group_data.items():
            verbose('|  +- ' + str(app), 2)
            for user, user_data in app_data.items():
                verbose('|  |  +- ' + str(user), 3)
                for ds, usr_app_ds_data in user_data.items():
                    verbose('|  |  |  +- ' + str(ds), 4)
                    for instance, usr_app_ds_instance_data in usr_app_ds_data.items():
                        csv_filename = usr_app_ds_instance_data['csv_filename']
                        verbose('|  |  |  |  +- ' + str(instance), 4)
                        verbose('|  |  |  |  |  +- PIs sum  : ' + str(usr_app_ds_instance_data['Real']['sum']), 5)
                        if 'wallclock_time' not in usr_app_ds_instance_data:
                            warning(f'(missing wallclock_time) {user} - {app} - {ds} - {instance} ', csv_filename)
                        else:
                            # Has wall_clock time
                            wallclock_time = float(usr_app_ds_instance_data['wallclock_time'])
                            sum_of_PIs = float(usr_app_ds_instance_data['Real']['sum'])
                            verbose('|  |  |  |  |  +- Wallclock: ' + str(wallclock_time), 5)
                            if wallclock_time < sum_of_PIs:
                                warning(
                                    f'(wallclock time smaller than sum of PIs) {user} - {app} - {ds} - {instance} - wallclock_time: {wallclock_time}, sum of PIs: {sum_of_PIs}',
                                    csv_filename,
                                )
                            else:
                                PIs_wallclock_ratio = sum_of_PIs / wallclock_time
                                if PIs_wallclock_ratio < 0.90:
                                    warning(
                                        f'(sum_of_PIs / wallclock time ratio is {PIs_wallclock_ratio:.2f}) -- {user} - {app} - {ds} - {instance}',
                                        csv_filename,
                                    )


# ====================================================


def generate_csv_analysis_per_instance(data):
    proxy_metrics = ['Second PI', 'From 2 to 5', 'From 2 to 10', '0.5_s', '0.5_s-first']
    csv_fields = (
        ['group', 'app', 'user', 'dataset', 'instance', 'wallclock_time', 'PIs sum', 'PIs/wallclock_time']
        + proxy_metrics
        + ['csv_filename', 'warnings']
    )

    def print_row(row_data):
        for field in csv_fields:
            print(row_data[field], end=',')
        print()

    # Print header
    for field in csv_fields:
        print(field, end=',')
    print()

    # Reorganize data: app group -> app -> user -> dataset -> instance -> ...
    app_data = {}
    for user, user_data in data['Users'].items():
        for app, usr_app_data in user_data['apps'].items():
            group = app_group[app]
            if group not in app_data:
                app_data[group] = {}
            if app not in app_data[group]:
                app_data[group][app] = {}
            if user in app_data[group][app]:
                warning(f'USER {user} already in app_data[{group}][{app}]')
            else:
                app_data[group][app][user] = usr_app_data

    # Process reorganized data
    row_data = {}
    for group, group_data in app_data.items():
        verbose('+- ' + str(group), 1)
        row_data['group'] = group
        for app, app_data in group_data.items():
            verbose('|  +- ' + str(app), 2)
            row_data['app'] = app
            for user, user_data in app_data.items():
                verbose('|  |  +- ' + str(user), 3)
                row_data['user'] = user
                for ds, usr_app_ds_data in user_data.items():
                    verbose('|  |  |  +- ' + str(ds), 4)
                    row_data['dataset'] = ds
                    for instance, usr_app_ds_instance_data in usr_app_ds_data.items():
                        verbose('|  |  |  |  +- ' + str(instance), 4)
                        row_data['warnings'] = ''
                        row_data['instance'] = instance
                        csv_filename = usr_app_ds_instance_data['csv_filename']
                        row_data['csv_filename'] = csv_filename
                        row_data['PIs sum'] = usr_app_ds_instance_data['Real']['sum']
                        for proxy_metric in proxy_metrics:
                            row_data[proxy_metric] = usr_app_ds_instance_data[proxy_metric]['mean']

                        if 'wallclock_time' not in usr_app_ds_instance_data:
                            row_data['wallclock_time'] = ''
                            row_data['PIs/wallclock_time'] = ''
                            row_data['warnings'] += '(missing wallclock_time) '
                        else:
                            # Has wall_clock time
                            wallclock_time = float(usr_app_ds_instance_data['wallclock_time'])
                            row_data['wallclock_time'] = str(wallclock_time)
                            sum_of_PIs = float(usr_app_ds_instance_data['Real']['sum'])
                            PIs_wallclock_ratio = sum_of_PIs / wallclock_time
                            row_data['PIs/wallclock_time'] = str(PIs_wallclock_ratio)
                            if wallclock_time < sum_of_PIs:
                                row_data['warnings'] += '(wallclock time smaller than sum of PIs) '
                            else:
                                if PIs_wallclock_ratio < 0.90:
                                    row_data['warnings'] += (
                                        f'(sum_of_PIs / wallclock time ratio is {PIs_wallclock_ratio:.2f}) '
                                    )
                        print_row(row_data)


# ====================================================


def plot_correlation(X_values, X_label, Y_values, Y_label, user, app_name, ds, instance_names, plot_ideal, filename):
    app_alias = EXPERIM_ALIASES[app_name]

    if len(instance_names) < 3:
        print(f'WARNING!!! Not enough instances to plot a correlation {user}: {app_alias}/{app_name}')
        return

    fig, ax = plt.subplots(layout='constrained')

    # Compute R^2 value
    correlation_matrix = np.corrcoef(X_values, Y_values)
    correlation_xy = correlation_matrix[0, 1]
    r_squared = correlation_xy**2

    # sum_coorelation = sum(X_values) / sum(Y_values)
    # min_coorelation = min(X_values) / min(Y_values)
    # median_coorelation = statistics.median(X_values) / statistics.median(Y_values)

    # print(f'Correlation {filename_suffix} {user} - {app_name} - {ds} (R^2 = {r_squared:.3f}): '
    #       f'(sum {sum_coorelation:.3f}) - (min {min_coorelation:.3f}) - (median {median_coorelation:.3f})')

    correlation_factor_str = ''
    # if abs(r_squared - 1.0) < 0.1 and abs(median_coorelation - 1.0) > 0.01:
    # TEST: Fixing correlation by the median_coorelation
    # correlation_factor_str = f' (factor {median_coorelation:.3f})'
    # print(f'Fixing correlation by factor of {median_coorelation:.3f}: {app_name}')
    # Y_values = [value * median_coorelation for value in Y_values]

    # Plot each point with its corresponding instance name
    for i, (x, y) in enumerate(zip(X_values, Y_values)):
        plt.scatter(x, y, label=instance_names[i], zorder=10, color=COLORS[i])

    # Get the current x and y limits
    data_xlim = plt.xlim()
    data_ylim = plt.ylim()

    # Bring xlim and ylim to 0 if they are close to 0
    fit_threshould = 0.2
    min_xlim, max_xlim = data_xlim
    if abs(min_xlim / max_xlim) < fit_threshould:
        data_xlim = (0, max_xlim)
    min_ylim, max_ylim = data_ylim
    if abs(min_ylim / max_ylim) < fit_threshould:
        data_ylim = (0, max_ylim)

    # Plot trendline
    fit = np.polyfit(X_values, Y_values, 1)
    poly = np.poly1d(fit)

    # Calculate the corresponding y values for the x limits
    y_at_xmin = poly(data_xlim[0])
    y_at_xmax = poly(data_xlim[1])

    initial_point = min(data_xlim[0], data_ylim[0])
    final_point = max(data_xlim[1], data_ylim[1])

    # Plot ideal trendline (x=y)
    trend_lines = dict()
    if plot_ideal:
        (trend_lines['Ideal'],) = plt.plot(
            [initial_point, final_point],
            [initial_point, final_point],
            color='#aaaaaa',
            linestyle='-',
            linewidth=1,
            label='_nolegend_',
            zorder=0,
        )

    # Plot the line using the current x limits and the corresponding y values
    (trend_lines['Trend'],) = plt.plot(
        [data_xlim[0], data_xlim[1]],
        [y_at_xmin, y_at_xmax],
        color='#ff000070',
        linestyle='--',
        linewidth=2,
        label='_nolegend_',
        zorder=5,
    )

    trend_legend = plt.legend(trend_lines.values(), trend_lines.keys(), loc=4)

    # Set title including the R^2 value
    plt.title(f'{X_label} vs {Y_label} - $R^2 = {r_squared:.2f}$\n{app_alias}-{ds}{correlation_factor_str}')

    # c5.xlarge-1
    instance_max_name = max([len(i) for i in instance_names])
    ncol = 3
    if instance_max_name < 13:
        ncol = 5
    elif instance_max_name < 16:
        ncol = 4
    fig.legend(loc='outside lower center', ncol=ncol, fancybox=True, shadow=True, mode='expand')
    fig.add_artist(trend_legend)

    # # Set a fixed number of ticks on axis
    # # x_ticks = ax.get_xticks()
    # rounded_values = rounded_linspace(data_xlim[0], data_xlim[1], 7)
    # ax.xaxis.set_ticks(rounded_values)
    # # ax.set_xticklabels([f"{tick:.2e}" for tick in rounded_values])

    # # y_ticks = ax.get_yticks()
    # ax.yaxis.set_ticks(rounded_linspace(data_ylim[0], data_ylim[1], 7))
    # # ax.yaxis.set_ticks(np.linspace(*data_ylim, 8))

    # Ensure the plot limits are ajusted
    plt.xlim(data_xlim)
    plt.ylim(data_ylim)

    plt.xlabel(X_label)
    plt.ylabel(Y_label)

    plt.ticklabel_format(axis='both', style='sci', scilimits=(-2, 2))

    if 0:
        plt.pause(1)  # Display for 1 second
    else:
        plt.savefig(filename)
    plt.close()
    return filename


# Function to determine Pareto efficiency
def pareto_efficient_mask(df, col1='time', col2='cost'):
    pareto_mask = np.ones(df.shape[0], dtype=bool)
    for i, row in df.iterrows():
        # Compare against all other rows
        for j, other_row in df.iterrows():
            if (row[col1] >= other_row[col1]) and (row[col2] >= other_row[col2]) and (i != j):
                pareto_mask[i] = False
                break
    return pareto_mask


def plot_pareto_comparison(df_ref, df_comparison, pm, filename):
    pareto_ref = df_ref[pareto_efficient_mask(df_ref)]
    pareto_comp_mask = pareto_efficient_mask(df_comparison)
    pareto_comp = df_comparison[pareto_comp_mask]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=400)

    # Plot Real points
    ax1.scatter(df_ref['time prop.'], df_ref['cost prop.'], label='All Points - Real', color=COLORS[0], s=60)
    ax1.scatter(pareto_ref['time prop.'], pareto_ref['cost prop.'], label='Pareto Front - Real', color=COLORS[1], s=60)
    # Plot a cross in the selected instance by proxy
    ax1.scatter(
        df_ref[pareto_comp_mask]['time prop.'],
        df_ref[pareto_comp_mask]['cost prop.'],
        label=f'Pareto Front - Proxy {pm}',
        color=COLORS[3],
        marker='x',
        s=50,
    )
    # Plot proxy points
    ax2.scatter(
        df_comparison['time prop.'],
        df_comparison['cost prop.'],
        label=f'All Points - Proxy {pm}',
        color=COLORS[2],
        s=60,
    )
    ax2.scatter(
        pareto_comp['time prop.'],
        pareto_comp['cost prop.'],
        label=f'Pareto Front - Proxy {pm}',
        color=COLORS[3],
        marker='x',
        s=60,
    )
    # Common parameters
    for kind, ax, metric in ['Real', ax1, 'time'], [pm, ax2, 'cost']:
        ax.axvline(1.0, linestyle='-.', color='violet', alpha=0.5)
        ax.axhline(1.0, linestyle='-.', color='violet', alpha=0.5)
        ax.axhline(1.2, linestyle='--', color='gray', alpha=0.5)
        ax.axvline(1.2, linestyle='--', color='gray', alpha=0.5)
        ax.set_xlabel('Proportional time')
        ax.set_ylabel('Proportional cost')
        ax.legend()
        ax.set_title(f'Pareto Efficient Points - {kind}')
        # Save ax independently
        # extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        # fig.savefig(filename.replace('.pdf', f'_{metric}.pdf'), bbox_inches=extent.expanded(1.2, 1.2))

    plt.savefig(filename)
    plt.close('all')


def calculate_correlations(instance_names_l, proxy_metrics_l, PIs_sum_l, PIs_cost_l, basename, charts_dir):
    metrics = ['time', 'cost']
    proxies = ['From 2 to 5', 'From 2 to 10']
    result = {}
    df_real = pd.DataFrame({'time': PIs_sum_l, 'cost': PIs_cost_l})
    data = {'real': df_real}
    data.update(
        {pm: pd.DataFrame({'time': proxy_metrics_l[pm], 'cost': proxy_metrics_l[f'{pm}-Cost']}) for pm in proxies}
    )

    for df in data.values():
        for metric, counter_metric in zip(metrics, metrics[::-1]):
            df[f'{metric}/{counter_metric}'] = df[metric] / df[counter_metric]
        for column in df.columns:
            df[f'{column} prop.'] = df[column] / df[column].min()
        df['cost-benefit'] = WEIGHT_TIME * df['time/cost prop.'] + WEIGHT_COST * df['cost/time prop.']

    cost_benefit_real = df_real.loc[df_real['cost-benefit'].idxmin()]

    # Generate proportional and time/cost limited by 1.2x cost/time stats
    for metric, counter_metric in zip(metrics, metrics[::-1]):
        idx_min_real_criteria = df_real[metric][df_real[f'{counter_metric} prop.'] < DISCARD_THRESHOLD].idxmin()
        for pm, df in [(pm, data[pm]) for pm in proxies]:
            cost_benefit_proxy_based = df_real.loc[data[pm]['cost-benefit'].idxmin()]
            # Calculate the fastest/cheapest
            result[f'Prop. {pm} - {metric}'] = df_real[metric][df[metric].idxmin()] / df_real[metric].min()
            # Calculate the fastest/cheapest considerim the counter-metric limited
            min_proxy_criteria_idx = df[metric][df[f'{counter_metric} prop.'] < DISCARD_THRESHOLD].idxmin()
            # result[f'Max {DISCARD_THRESHOLD}x {counter_metric} - {pm}'] = (
            result[f'Max {DISCARD_THRESHOLD}x {pm} - {metric}'] = (
                df_real[metric][min_proxy_criteria_idx] / df_real[metric][idx_min_real_criteria]
            )
            # result[f'Max {DISCARD_THRESHOLD}x {counter_metric} - error - {pm}'] = (
            result[f'Max {DISCARD_THRESHOLD}x - error - {pm} - {metric}'] = (
                df_real[counter_metric][min_proxy_criteria_idx] / df_real[counter_metric][idx_min_real_criteria]
            )

            # Generate cost-benefit stats
            result[f'cost-benefit {pm} - {metric}'] = cost_benefit_proxy_based[metric] / cost_benefit_real[metric]

    # df_real['instances'] = instance_names_l
    if charts_dir and any([result[key] > 2 for key in result]):
        for pm, df in [(pm, data[pm]) for pm in proxies]:
            filename = os.path.join(charts_dir, pareto_subdir, f'{basename}-{pm}.pdf'.replace(' ', '_').lower())
            plot_pareto_comparison(data['real'], df, pm, filename)
    return result


def generate_csv_analysis_per_application(data, charts_dir):
    proxy_metrics = ['Second PI', 'From 2 to 5', 'From 2 to 10']
    proxy_metrics_2 = ['R2*', 'R2', 'Intercept', 'Slope', 'Intercept/min PIs sum', 'chartname']
    csv_fields = (
        [
            'Idx',
            'group',
            'app',
            'user',
            'dataset',
            '# instances',
            'min wallclock_time',
            'max wallclock_time',
            'min PIs sum',
            'max PIs sum',
            'Rank 0 min PI samples',
            'Rank 0 max PI samples',
            'Rank 0 min/max PI sample ratio',
        ]
        + [
            f'{mode} {pm} - {metric}'
            for pm in ['From 2 to 5', 'From 2 to 10']
            for mode in ['Prop.', f'Max {DISCARD_THRESHOLD}x', f'Max {DISCARD_THRESHOLD}x - error -', 'cost-benefit']
            for metric in ['time', 'cost']
        ]
        + ['Wallclock vs All PIs - chartname']
        + [
            f'{pm_type} vs All PIs - {pm2}'
            for pm in proxy_metrics
            for pm2 in proxy_metrics_2
            for pm_type in (pm, f'{pm}-Cost')
        ]
        + ['warnings']
    )

    def print_row(row_data):
        for field in csv_fields:
            print(row_data[field], end=',')
        print()

    # Print header
    for field in csv_fields:
        print(field, end=',')
    print()

    # Reorganize data: app group -> app -> user -> dataset -> instance -> ...
    app_data = {}
    for user, user_data in data['Users'].items():
        for app, usr_app_data in user_data['apps'].items():
            group = app_group[app]
            if group not in app_data:
                app_data[group] = {}
            if app not in app_data[group]:
                app_data[group][app] = {}
            if user in app_data[group][app]:
                warning(f'USER {user} already in app_data[{group}][{app}]')
            else:
                app_data[group][app][user] = usr_app_data

    # Process reorganized data
    row_data = {'Idx': 1}
    for group, group_data in app_data.items():
        verbose('+- ' + str(group), 1)
        row_data['group'] = group
        for app, group_app_data in group_data.items():
            verbose('|  +- ' + str(app), 2)
            row_data['app'] = app
            for user, user_data in group_app_data.items():
                verbose('|  |  +- ' + str(user), 3)
                row_data['user'] = user
                for ds, usr_app_ds_data in user_data.items():
                    verbose('|  |  |  +- ' + str(ds), 4)
                    row_data['dataset'] = ds
                    row_data['warnings'] = ''

                    wall_clock_time_l = []
                    wall_clock_cost_l = []
                    PIs_sum_l = []
                    PIs_cost_l = []
                    proxy_metrics_l = {pm: [] for pm in proxy_metrics}
                    proxy_metrics_l.update({f'{pm}-Cost': [] for pm in proxy_metrics})
                    instance_names_l = []

                    rank0_min_samples = ''
                    rank0_max_samples = ''

                    # proxy_metrics = ["Second PI", "From 2 to 5", "From 2 to 10", "0.5_s", "0.5_s-first"]
                    # for instance, usr_app_ds_instance_data in usr_app_ds_data.items():
                    for instance in sorted(usr_app_ds_data):
                        usr_app_ds_instance_data = usr_app_ds_data[instance]
                        instance_names_l.append(instance)
                        instance_cost = (
                            usr_app_ds_instance_data['Instance Price'] * usr_app_ds_instance_data['Instance Count']
                        ) / 3.6e6  # 1 hour in milliseconds
                        if 'wallclock_time' in usr_app_ds_instance_data:
                            wall_clock_time_l.append(float(usr_app_ds_instance_data['wallclock_time']))
                            wall_clock_cost_l.append(float(usr_app_ds_instance_data['wallclock_time']) * instance_cost)
                        PIs_sum_l.append(float(usr_app_ds_instance_data['Real']['sum']))
                        PIs_cost_l.append(float(usr_app_ds_instance_data['Real']['sum']) * instance_cost)
                        for pm in proxy_metrics:
                            proxy_metrics_l[pm].append(float(usr_app_ds_instance_data[pm]['mean']))
                            proxy_metrics_l[f'{pm}-Cost'].append(
                                float(usr_app_ds_instance_data[pm]['mean'] * instance_cost)
                            )

                        rank0_samples = usr_app_ds_instance_data['PI Samples rank0']
                        if rank0_min_samples == '':
                            rank0_min_samples = rank0_samples
                        if rank0_samples < rank0_min_samples:
                            rank0_min_samples = rank0_samples
                        if rank0_max_samples == '':
                            rank0_max_samples = rank0_samples
                        if rank0_samples > rank0_max_samples:
                            rank0_max_samples = rank0_samples

                    row_data.update(
                        calculate_correlations(
                            instance_names_l,
                            proxy_metrics_l,
                            PIs_sum_l,
                            PIs_cost_l,
                            f'{user.replace("/", "-")}-{ds}',
                            charts_dir,
                        )
                    )

                    row_data['# instances'] = len(PIs_sum_l)
                    row_data['Rank 0 min PI samples'] = rank0_min_samples
                    row_data['Rank 0 max PI samples'] = rank0_max_samples
                    row_data['Rank 0 min/max PI sample ratio'] = rank0_min_samples / rank0_max_samples

                    # Summarize results
                    if len(wall_clock_time_l) > 0:
                        row_data['min wallclock_time'] = min(wall_clock_time_l)
                        row_data['max wallclock_time'] = max(wall_clock_time_l)
                    else:
                        row_data['min wallclock_time'] = ''
                        row_data['max wallclock_time'] = ''
                    row_data['min PIs sum'] = min(PIs_sum_l)
                    row_data['max PIs sum'] = max(PIs_sum_l)
                    row_data['min PIs cost'] = min(PIs_cost_l)
                    row_data['max PIs cost'] = max(PIs_cost_l)

                    for pm in proxy_metrics_l:
                        # Compute correlation between pm and All PIs.
                        if len(PIs_sum_l) < 2:
                            row_data[f'{pm} vs All PIs - R2'] = ''
                            row_data[f'{pm} vs All PIs - R2*'] = ''
                            row_data[f'{pm} vs All PIs - Intercept'] = ''
                            row_data[f'{pm} vs All PIs - Slope'] = ''
                            row_data[f'{pm} vs All PIs - Intercept/min PIs sum'] = ''
                            row_data['warnings'] = f'(number of samples - {len(PIs_sum_l)} - too small for statistics)'
                            continue
                        PIs_reference_l = PIs_cost_l if pm.endswith('-Cost') else PIs_sum_l
                        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
                            proxy_metrics_l[pm], PIs_reference_l
                        )
                        row_data[f'{pm} vs All PIs - R2'] = r_value
                        row_data[f'{pm} vs All PIs - Intercept'] = intercept
                        row_data[f'{pm} vs All PIs - Slope'] = slope
                        row_data[f'{pm} vs All PIs - Intercept/min PIs sum'] = intercept / row_data['min PIs sum']

                        correlation_matrix = np.corrcoef(proxy_metrics_l[pm], PIs_reference_l)
                        correlation_xy = correlation_matrix[0, 1]
                        r_squared = correlation_xy**2
                        row_data[f'{pm} vs All PIs - R2*'] = r_squared

                    row_data['Wallclock vs All PIs - chartname'] = ''
                    for pm in proxy_metrics_l:
                        row_data[f'{pm} vs All PIs - chartname'] = ''
                    if charts_dir:
                        # Plot chart
                        basename = f'{user.replace("/", "-")}_{app[:20]}-{ds}'
                        if len(wall_clock_time_l) >= 3:
                            filename = os.path.join(charts_dir, f'{basename}-wallclock_vs_sum_pi.pdf')
                            plot_correlation(
                                Y_values=PIs_sum_l,
                                Y_label='Sum of PIs (ms)',
                                X_values=wall_clock_time_l,
                                X_label='Total execution time (ms)',
                                user=user,
                                app_name=app,
                                ds=ds,
                                instance_names=instance_names_l,
                                plot_ideal=True,
                                filename=filename,
                            )
                            row_data['Wallclock vs All PIs - chartname'] = filename

                            filename = os.path.join(charts_dir, f'{basename}-wallclock_vs_sum_pi-cost.pdf')
                            plot_correlation(
                                Y_values=PIs_cost_l,
                                Y_label='Sum of PIs cost (USD)',
                                X_values=wall_clock_cost_l,
                                X_label='Total execution cost (USD)',
                                user=user,
                                app_name=app,
                                ds=ds,
                                instance_names=instance_names_l,
                                plot_ideal=True,
                                filename=filename,
                            )
                            row_data['Wallclock cost vs All PIs cost - chartname'] = filename

                        if len(PIs_sum_l) >= 3:
                            for pm in proxy_metrics_l:
                                filename_suffix = pm.lower().replace(' ', '_') + '_vs_sum_pi'
                                filename = os.path.join(charts_dir, f'{basename}-{filename_suffix}.pdf')
                                # unit_s = 'USD' if pm.endswith('-Cost') else 'ms'
                                unit_s = 'ms'
                                if pm.endswith('-Cost'):
                                    continue
                                    unit_s = 'USD'
                                    filename = os.path.join(charts_dir, f'{basename}-{filename_suffix}.pdf')
                                plot_correlation(
                                    X_values=PIs_sum_l,
                                    X_label=f'Sum of PIs ({unit_s})',
                                    Y_values=proxy_metrics_l[pm],
                                    Y_label=f'{pm} ({unit_s})',
                                    user=user,
                                    app_name=app,
                                    ds=ds,
                                    instance_names=instance_names_l,
                                    plot_ideal=False,
                                    filename=filename,
                                )
                                row_data[f'{pm} vs All PIs - chartname'] = filename

                    print_row(row_data)
                    row_data['Idx'] += 1
        # break


# ====================================================

if __name__ == '__main__':
    # Initialize parser
    parser = argparse.ArgumentParser()

    # Adding optional argument
    parser.add_argument('-i', '--input_file', help='Input pickle file (.pkl) ')
    parser.add_argument('-v', '--verbosity', help='Verbosity level: 0 (default), 1, 2, 3, 4')
    parser.add_argument('-d', '--dump_data', help='Dump data', action='store_true')
    parser.add_argument('--W_show_csv_filename', help='Show CSV filename on warnings', action='store_true')
    parser.add_argument(
        '--analysis_per_instance', help='Generate CSV table with analysis per instance', action='store_true'
    )
    parser.add_argument(
        '--analysis_per_application', help='Generate CSV table with analysis per application', action='store_true'
    )
    parser.add_argument(
        '--application_charts_dir', help='Directory to store charts when performing analysis per application'
    )

    # Read arguments from command line
    args = parser.parse_args()

    if args.verbosity:
        verbosity_level = int(args.verbosity)

    if args.W_show_csv_filename:
        W_show_csv_filename = True

    if not args.input_file:
        error('Input file expected but not provided (-i)')

    if not os.path.exists(args.input_file):
        error(f'{args.input_file} not found!')

    verbose(f'Loading data from  {args.input_file}', 1)
    with open(args.input_file, 'rb') as file:
        data = pickle.load(file)

    if args.dump_data:
        print_object(data, '.')

    # wallclock_time_sanity_check(data)
    # print_apps(data)
    # wallclock_time_sanity_check_by_app(data)
    if args.analysis_per_instance:
        generate_csv_analysis_per_instance(data)

    if args.analysis_per_application:
        if charts_dir := args.application_charts_dir:
            os.makedirs(os.path.join(charts_dir, pareto_subdir), exist_ok=True)
            os.makedirs(os.path.join(charts_dir, costs_subdir), exist_ok=True)
            os.makedirs(charts_dir, exist_ok=True)
        generate_csv_analysis_per_application(data, args.application_charts_dir)

# ====================================================
