import warnings

import numpy as np
import pandas as pd
from dask.distributed import Client
from colors import COLORS

NUM_WORKERS = 3
MEMORY_LIMIT = f'{NUM_WORKERS}GB'

MAX_PIS = int(1e22)

# Suppress specific warning message
warnings.filterwarnings('ignore', message='Port 8787 is already in use')


def get_normal_array(window):
    if window <= 1:
        return []
    arrange = np.arange(1, window // 2 + 1, 1)
    weights_array = np.concatenate((arrange, [arrange[0] + arrange[-1]], np.flip(arrange)))
    return list(weights_array)


def isntance_color_order_by_variation(instance_data):
    instance_variations = [
        [instance_name, count, df['time'].max() - df['time'].min()]
        for count, (instance_name, df) in enumerate(instance_data.items())
    ]
    isntance_color_order_list = sorted(
        [
            (instance_name, color, zorder)
            for zorder, (instance_name, color, _) in enumerate(
                sorted(instance_variations, key=lambda x: x[2], reverse=True), 1
            )
        ],
        key=lambda x: x[1],
    )
    return isntance_color_order_list


def plot_raw_chart(app_name, instance_data, ignore, ax, max_pi=None, metric='time', isntance_color_order_list=None):
    # Plot comparison chart
    if not max_pi:
        max_pi = MAX_PIS
    if not isntance_color_order_list:
        isntance_color_order_list = isntance_color_order_by_variation(instance_data)
    for instance_name, color, zorder in isntance_color_order_list:
        dataframe = instance_data[instance_name]
        ax.plot(metric, data=dataframe.iloc[ignore:max_pi], color=COLORS[color], zorder=zorder, label=instance_name)
    ax.set_title(f'{app_name} - {metric.title()}')
    return isntance_color_order_list


def process_window(instance_name, dataframe, normal_array, ignore, max_pi):
    data_len = min(len(dataframe), max_pi)
    # max_data_len = max(len(dataframe), data_len)
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


def plot_relative_chart(app_name, instance_data, window, ignore, ax, max_pi=None, isntance_color_order_list=None):
    # Plot relative chart
    if not max_pi:
        max_pi = MAX_PIS
    normal_array = get_normal_array(window)
    title = f'{app_name} - Relative'
    if len(normal_array) > 1:
        title += f' - Window {len(normal_array)}'
    if ignore:
        title += f' - Ignore {ignore}'
    if max_pi != MAX_PIS:
        title += f' - Max {max_pi}'
    concat_data = {}
    data_lengths = []
    client = Client(n_workers=NUM_WORKERS, memory_limit='5GB')
    futures = []
    for instance_name, dataframe in instance_data.items():
        future_dataframe = client.scatter(dataframe)
        future = client.submit(process_window, instance_name, future_dataframe, normal_array, ignore, max_pi)
        futures.append(future)
    results = client.gather(futures)
    client.close()
    for result in results:
        if result is not None:
            data_lengths.append(result[0])
            concat_data.update(result[1])
    if len(data_lengths) == 0:
        print(f'No data for processing: {app_name}')
        return
    data_lengths.sort()
    if data_lengths[0] != data_lengths[-1]:
        print(f'Incompatible lengths for app: {app_name}')
        return
    try:
        df2 = pd.DataFrame(concat_data)
    except ValueError:
        print(f'Error combining data for app: {app_name}')
        return
    # Normilize data
    dfn = df2.div(df2.mean(axis=1), axis=0)
    dfn.reset_index(drop=True)
    dfn.index = dfn.index + ignore + 1
    if isntance_color_order_list:
        for instance_name, color, zorder in isntance_color_order_list:
            dfn.plot(ax=ax, y=instance_name, color=COLORS[color], zorder=zorder)
    else:
        for count, instance_name in enumerate(dfn.columns):
            dfn.plot(ax=ax, y=instance_name, color=COLORS[count])

    # Set the title of the plot
    ax.set_title(title)
    ax.legend().set_visible(False)
