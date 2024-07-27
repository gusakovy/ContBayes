import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from experiments.csv_api import convert_string_to_value
from dir_definitions import RESULTS_DIR

SAVEFILE_DIR = os.path.join(RESULTS_DIR, 'DeepSIC', 'results')
RESULTS = ['ber', 'confidence', 'skip_ratio', 'run_time', 'savefile']
METHODS = ['EKF', 'SqrtEKF', 'LF-VCL', 'GD', 'Joint-Learning', 'Retrain']
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
          '#7f7f7f', '#bcbd22', '#17becf']

# User input
x_variable = 'tracking_snr'  # use 'time' for per-frame metrics.
y_variable = 'ber'
constraints = {'version': 0.3,
               'channel_type': 'Synthetic',
               'fading_coefficient': 0.7,
               'constellation': 'QPSK',
               'num_pilots': ['-', 128, 256],
               'num_epochs': ['-', 1],
               'num_batches': ['-', 1],
               'hidden_size': 12,
               'normalization': ['-', 'mean']}

select_criteria = 'tracking_snr'
select_value = 13

plot_title = f"BER vs SNR - {constraints['channel_type']} channel, {constraints['num_pilots'][-1]} pilots"
plot_x_label = "SNR [dB]"
plot_y_label = "BER"
log_scale_x = None
log_scale_y = 10

# Database path
data_path = os.path.join(RESULTS_DIR, 'DeepSIC', 'database.csv')
df = pd.read_csv(data_path)
df = df.map(convert_string_to_value)

# Filter based on the constraints
constraints = {k: [v] if not isinstance(v, list) else v for k, v in constraints.items()}
for key, constraint in constraints.items():
    df = df[df[key].isin(constraint)]

legend_labels = []
fig, ax = plt.subplots()

idx = 1
# Iterate over all tracking methods
for tracking_method in METHODS:

    # Filter dataframe to get entries for this tracking method alone
    method_df = df[df['tracking_method'] == tracking_method]

    # Find all combinations of number of epochs/batches for the method
    epoch_batch_pairs = set(zip(method_df['num_epochs'], method_df['num_batches']))
    epoch_batch_pairs = sorted(list(epoch_batch_pairs))

    for num_epochs, num_batches in epoch_batch_pairs:

        # Filter dataframe for the specific number of epochs and batches
        epochs_batches_df = method_df[(method_df['num_epochs'] == num_epochs) &
                                      (method_df['num_batches'] == num_batches)]

        # Find the experiment for which y_variable is minimized according to selection criteria
        chosen_params_idx = epochs_batches_df[epochs_batches_df[select_criteria] == select_value]['ber'].idxmin()
        chosen_params = epochs_batches_df.loc[chosen_params_idx]
        num_blocks = chosen_params['num_blocks']

        # Generate mask for rows with the best parameters
        excluded_params = ['seed', x_variable, y_variable] + RESULTS
        mask = np.logical_and.reduce([method_df[col] == chosen_params[col] for col in chosen_params.index if
                                      col not in excluded_params])
        best_params_df = method_df[mask]
        best_params_str = ', '.join(f'{col}={val}' for col, val in chosen_params.items() if
                                    col not in excluded_params)
        print(f'{idx}: {tracking_method} and parameters: {best_params_str}')

        # Process results
        if x_variable == 'time':
            if y_variable not in ['ber', 'confidence']:
                raise ValueError("Per-frame metrics are only supported for BER and Confidence")
            x_val = []
            y_val = []
            for _, row in best_params_df.iterrows():
                savefile = row['savefile']
                savefile_path = os.path.join(SAVEFILE_DIR, savefile)
                bers, confs = pickle.load(open(savefile_path, 'rb'))
                data = bers if y_variable == 'ber' else confs
                if row[select_criteria] == select_value:
                    x_val = np.arange(1, num_blocks + 1)
                    y_val = np.cumsum(bers) / np.arange(1, len(bers) + 1)
        else:
            x_val = best_params_df[x_variable].values
            y_val = best_params_df[y_variable].values

        # Plot results
        if ((num_epochs != '-' or num_batches != '-') and
                tracking_method != "Nothing"):
            label = f'{idx} - {tracking_method}_{num_epochs}_{num_batches}'
        else:
            label = f'{idx} - {tracking_method}'

        if y_variable == 'run_time':
            y_val /= num_blocks

        ax.plot(x_val, y_val,
                linestyle='--' if tracking_method in ["Joint-Learning", "Retrain"] else '-',
                marker='',
                color = COLORS[idx % 10],
                label = label)

        idx += 1

# Plot settings
plt.xlabel(plot_x_label)
plt.ylabel(plot_y_label)
plt.title(plot_title)
if log_scale_x is not None:
    plt.xscale("log", base=log_scale_x)
if log_scale_y is not None:
    plt.yscale("log", base=log_scale_y)
    ax = plt.gca()
    ax.yaxis.set_major_locator(ticker.LogLocator(base=log_scale_y, numticks=10))

plt.grid(True, which='both')
plt.legend(title="Method")
plt.show()
