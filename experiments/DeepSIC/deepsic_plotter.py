import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from experiments.csv_api import convert_string_to_value
from dir_definitions import RESULTS_DIR

SAVEFILE_DIR = os.path.join(RESULTS_DIR, 'DeepSIC', 'results')
VARIABLES_DICT = {
    'block': 'Block no.',
    'seed': 'Seed',
    'num_users': 'Number of users',
    'num_antennas': 'Number of antennas',
    'fading_coefficient': 'Fading coefficient',
    'tracking_snr': 'Tracking SNR [dB]',
    'num_layers': 'Number of DeepSIC layers',
    'hidden_size': 'DeepSIC block hidden layer size',
    'num_pilots': 'Number of pilots per block',
}
METRICS_DICT = {
    'error_rate': 'SER',
    'confidence': 'Confidence',
    'skip_ratio': 'Skip Ratio',
    'run_time': 'Run Time',
}
METHODS = ['Pre-train', 'EKF', 'LF-VCL', 'VCL', 'CL', 'Retrain']
COLORS = ['#ac564b', '#2ca02c', '#9467bd', '#1f77b4', '#17bec9', '#ff7f0e', '#d62728', '#7f7f7f', '#bcbd22', '#e377c2']
MARKERS = ["o", "v", "^", "s", "P", "X", "p", 'd']

def plot_deepsic_experiment_results(x_variable, y_variable, constraints, select_criteria, **kwargs):

    if x_variable not in VARIABLES_DICT.keys():
        raise ValueError(f"Invalid x_variable {x_variable}, must be one of {VARIABLES_DICT.keys()}")
    if y_variable not in METRICS_DICT.keys():
        raise ValueError(f"Invalid y_variable {y_variable}, must be one of {METRICS_DICT.keys()}")

    # Database path
    data_path = os.path.join(RESULTS_DIR, 'DeepSIC', 'database.csv')
    df = pd.read_csv(data_path)
    df = df.map(convert_string_to_value)

    # Filter based on the constraints
    constraints = {k: [v] if not isinstance(v, list) else v for k, v in constraints.items()}
    for key, constraint in constraints.items():
        df = df[df[key].isin(constraint)]

    if kwargs.get('ax') is not None:
        ax = kwargs.get('ax')
    else:
        fig, ax = plt.subplots()

    idx = 0
    marker_idx = 0
    x_min, x_max, y_min, y_max = None, None, None, None
    # Iterate over all tracking methods
    for tracking_method in METHODS:

        # Filter dataframe to get entries for this tracking method alone
        method_df = df[df['tracking_method'] == tracking_method]
        method_df = method_df[method_df[x_var] != '-']
        method_df = method_df[method_df[y_var] != '-']
        if len(method_df) == 0:
            continue

        # Find all combinations of number of epochs/batches for the method
        epoch_batch_pairs = set(zip(method_df['num_epochs'], method_df['num_batches']))
        epoch_batch_pairs = sorted(list(epoch_batch_pairs))

        for num_epochs, num_batches in epoch_batch_pairs:

            # Filter dataframe for the specific number of epochs and batches
            epochs_batches_df = method_df[(method_df['num_epochs'] == num_epochs) &
                                          (method_df['num_batches'] == num_batches)]

            # Find the experiment for which y_variable is minimized according to selection criteria
            chosen_params = epochs_batches_df.loc[epochs_batches_df.index[0]]
            if len(epochs_batches_df) != 1:
                chosen_params_idx = (
                    epochs_batches_df[epochs_batches_df[select_criteria["variable"]] ==
                                      select_criteria['value']][y_variable].idxmin()
                    if select_criteria['metric'] == 'min' else
                    epochs_batches_df[epochs_batches_df[select_criteria["variable"]] ==
                                      select_criteria['value']][y_variable].idxmax()
                )
                chosen_params = epochs_batches_df.loc[chosen_params_idx]

            num_blocks = chosen_params['num_blocks']

            # Generate mask for rows with the chosen parameters
            excluded_params = ['seed', x_variable, y_variable, 'savefile'] + list(METRICS_DICT.keys())
            mask = np.logical_and.reduce([method_df[col] == chosen_params[col] for col in chosen_params.index if
                                          col not in excluded_params])
            chosen_params_df = method_df[mask]
            chosen_params_str = ', '.join(f'{col}={val}' for col, val in chosen_params.items() if
                                          col not in excluded_params)
            print(f'{idx+1}: {tracking_method} and parameters: {chosen_params_str}')

            # Process results
            if x_variable == 'block':
                if y_variable not in ['error_rate', 'confidence']:
                    raise ValueError("Per-frame metrics are only supported for SER and Confidence")
                x_val = np.arange(1, num_blocks+1)
                y_val = np.zeros([chosen_params_df.shape[0], num_blocks], dtype=float)
                i = 0
                for _, row in chosen_params_df.iterrows():
                    savefile = row['savefile']
                    savefile_path = os.path.join(SAVEFILE_DIR, savefile)
                    sers, confs = pickle.load(open(savefile_path, 'rb'))
                    data = sers if y_variable == 'error_rate' else confs
                    if row[select_criteria['variable']] == select_criteria['value']:
                        y_val[i, :] = (np.cumsum(data) / np.arange(1, len(data) + 1))
                    i += 1
                y_val = y_val.mean(axis=0)


            else:
                chosen_params_df = chosen_params_df.groupby(x_variable)[y_variable].mean().reset_index()
                x_val = chosen_params_df[x_variable].values
                y_val = chosen_params_df[y_variable].values

            # Plot results
            if (num_epochs != '-' or num_batches != '-') and tracking_method not in ["Pre-train", "Retrain"]:
                label = f'{tracking_method}-{num_epochs}'
            else:
                label = f'{tracking_method}'

            if y_variable == 'run_time':
                y_val /= num_blocks

            ax.plot(x_val, y_val,
                    linestyle='--' if tracking_method=="Pre-train" else '-.' if tracking_method=="Retrain" else '-',
                    marker= '' if tracking_method in ["Pre-train", "Retrain"] else MARKERS[marker_idx % 10],
                    markevery = int(len(x_val) / 10) + 1 if len(x_val) > 12 else 1,
                    markersize = 6,
                    zorder = 10 if tracking_method in ["Pre-train", "Retrain"] else 3,
                    color = COLORS[idx % 10],
                    label = label)

            x_min = x_val.min() if x_min is None else min(x_min, x_val.min())
            x_max = x_val.max() if x_max is None else max(x_max, x_val.max())
            y_min = y_val.min() if y_min is None else min(y_min, y_val.min())
            y_max = y_val.max() if y_max is None else max(y_max, y_val.max())

            idx += 1
            marker_idx = marker_idx if tracking_method in ["Pre-train", "Retrain"] else marker_idx + 1

    plot_x_label = VARIABLES_DICT[x_variable]
    plot_y_label = METRICS_DICT[y_variable]
    x_range = x_max - x_min if x_max - x_min > 0 else x_max
    x_lim = [x_min - x_range * 0.05, x_max + x_range * 0.05]

    # Plot settings
    ax.set_xlabel(plot_x_label)
    ax.set_ylabel(plot_y_label)
    log_x = kwargs.get('log_scale_x')
    if log_x:
        ax.set_xscale("log", base=log_x)
    log_y = kwargs.get('log_scale_y')
    if log_y:
        ax.set_yscale("log", base=log_y)
        ax.yaxis.set_major_locator(ticker.LogLocator(base=log_y, numticks=10))
    if x_lim is not None:
        ax.set_xlim(x_lim)
    ax.grid(True, which='both')
    ax.legend(fancybox=True, framealpha=0.7, ncol = 2, title='Method')

    return ax


if __name__ == '__main__':

    x_var = 'tracking_snr'
    y_var = 'error_rate'
    params = {
        'version': 1.0,
        'channel_type': 'Cost2100',
        'tracking_snr': [8, 9, 10, 11, 12],
        'fading_coefficient': 0.65,
        'constellation': 'QPSK',
        'num_pilots': ['-', 128],
        'num_epochs': ['-', 4, 20],
        'num_batches': ['-', 1],
        'tracking_lr': ['-', 0.0005],
        'hidden_size': 6,
        'num_blocks': 75,
        'state_model': ['-', 1],
        'diag_loading': ['-', 0],
        'normalization': ['-', 'mean']
    }

    """
    If the params defined above are not sufficient to isolate a single plot for each combination of 
    (method, num_epochs, num_batches) in the database, a plot will be selected based on the selection criteria.
    Example: For y_variable='error_rate', select_variable='tracking_snr', select_value=10, and select_criteria='min',
    the plot for which the error rate is minimized when the tracking SNR is 10dB will be selected.
    """
    plot_select_criteria = {
        'variable': 'tracking_snr',
        'value': 11,
        'metric': 'min', # 'min'/'max'
    }

    log_scale_x = None # Log scale in the x-axis
    log_scale_y = 10 # Log scale in the y-axis

    plot_deepsic_experiment_results(x_var, y_var, params, plot_select_criteria,
                                    log_scale_x=log_scale_x, log_scale_y=log_scale_y)

    plt.tight_layout()
    plt.show()
