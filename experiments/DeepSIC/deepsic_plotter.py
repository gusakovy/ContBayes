import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from experiments.csv_api import convert_string_to_value
from dir_definitions import RESULTS_DIR

RESULTS = ['ber', 'confidence', 'skip_ratio', 'run_time']
METHODS = ['Nothing', 'EKF', 'SqrtEKF', 'SVI', 'GD']

# User input
x_variable = 'num_pilots'
y_variable = 'run_time'
constraints = {'version': [0.1],
               'num_epochs': ['-', 1],
               'tracking_snr': [13],
               'num_layers': [4],
               'hidden_size': [48],
               'diag_loading': ['-', 0]}
select_criteria = 'tracking_snr'
select_value = 13

plot_title = "Update time vs. Number of Pilots"
plot_x_label = "Number of Pilots"
plot_y_label = "Update time[s]"
log_scale_x = 2
log_scale_y = 10

# Database path
data_path = os.path.join(RESULTS_DIR, 'DeepSIC', 'database.csv')
df = pd.read_csv(data_path)
df = df.map(convert_string_to_value)

# Filter based on the constraints
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
        epochs_batches_df = method_df[(method_df['num_epochs'] == num_epochs) & (method_df['num_batches'] == num_batches)]

        # Find the experiment for which y_variable is minimized according to selection criteria
        chosen_params_idx = epochs_batches_df[epochs_batches_df[select_criteria] == select_value]['ber'].idxmin()
        chosen_params = epochs_batches_df.loc[chosen_params_idx]

        # Generate mask for rows with the best parameters
        excluded_params = ['tracking_method', x_variable, y_variable] + RESULTS
        mask = np.logical_and.reduce([method_df[col] == chosen_params[col] for col in chosen_params.index if
                                      col not in excluded_params])
        best_params_df = method_df[mask]

        best_params_str = ', '.join(f'{col}={val}' for col, val in chosen_params.items() if
                                    col not in excluded_params)
        print(f'{idx}: {tracking_method} and parameters: {best_params_str}')

        # Sort values for better plot
        best_params_df = best_params_df.sort_values(by=x_variable)

        # Plot results
        if y_variable == 'run_time':
            ax.plot(best_params_df[x_variable], best_params_df[y_variable] / chosen_params['num_blocks'],
                    linestyle='--' if tracking_method in ["Nothing", "Oracle"] else '-',
                    marker='' if tracking_method == "Nothing" else '.')
        else:
            ax.plot(best_params_df[x_variable], best_params_df[y_variable],
                    linestyle='--' if tracking_method in ["Nothing", "Oracle"] else '-',
                    marker='' if tracking_method == "Nothing" else '.')

        # Save best parameters for the legend label
        if ((chosen_params["num_epochs"] != '-' or chosen_params["num_batches"] != '-') and
                chosen_params["tracking_method"] != "Nothing"):
            legend_labels.append(f'{idx} - {tracking_method} '
                                 f'({chosen_params["num_epochs"]} epochs, {chosen_params["num_batches"]} batches)')
        else:
            legend_labels.append(f'{idx} - {tracking_method}')

        idx += 1

# Plot settings
plt.xlabel(plot_x_label)
plt.ylabel(plot_y_label)
plt.title(plot_title)
if log_scale_x is not None:
    plt.xscale("log", base=log_scale_x)
if log_scale_y is not None:
    plt.yscale("log", base=log_scale_y)
plt.legend(legend_labels, title="Method")
plt.show()
