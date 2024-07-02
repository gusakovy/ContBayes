import os
import time
from functools import partial
import matplotlib.pyplot as plt
import torch
import pyro
from dir_definitions import EXP_DIR, RESULTS_DIR
from contbayes.Detectors.DeepSIC import DeepSIC
from contbayes.Detectors.BayesianDeepSIC import BayesianDeepSIC
from contbayes.Trackers.EKF import DeepsicEKF
from contbayes.Trackers.SqrtEKF import DeepsicSqrtEKF
from contbayes.Trackers.OnlineTracker import DeepsicTracker
from contbayes.Channels.sed_channel import SEDChannel
from contbayes.Channels.cost_channel import Cost2100Channel
from contbayes.Utilities.data_utils import prepare_experiment_data, prepare_single_batch
from experiments.experiment import Experiment
from experiments.config import Config
from experiments.csv_api import append_dict_to_csv, is_matching_row_in_csv
from experiments.DeepSIC.deepsic_helper import deepsic_parameter_combinations
from experiments.DeepSIC.deepsic_callbacks import *


class DeepsicExperiment(Experiment):
    """
    Represents a single instance of a DeepSIC experiment.

    :param param_dict: dictionary of experiment parameters
    :param verbose: whether to print progress messages
    """


    def __init__(self, param_dict: dict, verbose: bool = False):
        super().__init__(param_dict, verbose)

    def _validate_experiment(self):
        if is_matching_row_in_csv(self.params, os.path.join(RESULTS_DIR, 'DeepSIC', 'database.csv')):
            raise ValueError("Experiment already logged.")

    def _init_channel(self):
        match self.params['channel_type']:
            case 'Synthetic':
                self.channel = SEDChannel(modulation_type=self.params['constellation'],
                                          num_users=self.params['num_users'],
                                          num_antennas=self.params['num_antennas'],
                                          fading_coefficient=self.params['fading_coefficient'],
                                          linear_channel=self.params['linear'])
            case 'Cost2100':
                self.channel = Cost2100Channel(modulation_type=self.params['constellation'],
                                               num_users=self.params['num_users'],
                                               num_antennas=self.params['num_antennas'],
                                               fading_coefficient=self.params['fading_coefficient'],
                                               linear_channel=self.params['linear'])

            case _:
                raise ValueError(f"Channel type '{self.params['channel_type']}' not supported.")

    def _init_model(self):
        if self.params['tracking_method'] != 'GD':
            pyro.clear_param_store()
            self.model = BayesianDeepSIC(modulation_type=self.params['constellation'],
                                         num_users=self.params['num_users'],
                                         num_ant=self.params['num_antennas'],
                                         num_iterations=self.params['num_layers'],
                                         hidden_dim=self.params['hidden_size'])

        else:
            self.model = DeepSIC(modulation_type=self.params['constellation'],
                                 num_users=self.params['num_users'],
                                 num_ant=self.params['num_antennas'],
                                 num_iterations=self.params['num_layers'],
                                 hidden_dim=self.params['hidden_size'])

        self._init_tracker()

    def _init_tracker(self):
        match self.params['tracking_method']:
            case 'SVI' | 'GD':
                self.tracker = DeepsicTracker(detector=self.model,
                                              num_epochs=self.params['num_epochs'],
                                              num_batches=self.params['num_batches'],
                                              learning_rate=self.params['tracking_lr'])

            case 'EKF':
                self.tracker = DeepsicEKF(detector=self.model,
                                          state_model=self.params['state_model'],
                                          process_noise_var=self.params['process_noise_var'],
                                          diag_loading=self.params['diag_loading'],
                                          update_limit=self.params['update_limit'],
                                          obs_reduction=self.params['reduction'] if
                                              self.params['reduction'] != '-' else None,
                                          obs_normalization=self.params['normalization'] if
                                              self.params['normalization'] != 'none' else None)
            case 'SqrtEKF':
                self.tracker = DeepsicSqrtEKF(detector=self.model,
                                              state_model=self.params['state_model'],
                                              process_noise_var=self.params['process_noise_var'],
                                              diag_loading=self.params['diag_loading'],
                                              update_limit=self.params['update_limit'],
                                              obs_reduction=self.params['reduction'] if
                                                  self.params['reduction'] != '-' else None,
                                              obs_normalization=self.params['normalization'] if
                                                  self.params['normalization'] != 'none' else None)

            case 'Nothing':
                self.tracker = None

            case _:
                raise ValueError(f"Unrecognized tracking method {self.params['tracking_method']} for Bayesian DeepSIC")

    def _reset_environment(self):
        pyro.set_rng_seed(self.params['seed'])
        torch.cuda.manual_seed(self.params['seed'])

    def _warm_start(self):
        warm_start_path = os.path.join(RESULTS_DIR, 'DeepSIC', 'warm_starts', str(self.params['channel_type']),
                                       'bayesian' if self.params['tracking_method'] != 'GD' else 'frequentist')

        try:
            self.model.load_model(os.path.join(warm_start_path, f"{self.params['constellation']}_"
                                                                f"{self.params['num_users']}_"
                                                                f"{self.params['num_layers']}_"
                                                                f"{self.params['hidden_size']}.pt"))
        except FileNotFoundError:
            if self.verbose:
                print("Warming up...")

            if not os.path.exists(warm_start_path):
                os.makedirs(warm_start_path)

            rx, labels = prepare_single_batch(channel=self.channel,
                                              num_samples=self.params['training_dim'],
                                              frame_idx=0,
                                              snr=self.params['warm_start_snr'])

            losses = [[[] for _ in range(self.params['num_layers'])] for _ in range(self.params['num_users'])]

            loss_callback = bayesian_deepsic_loss_callback if self.params['tracking_method'] != 'GD' \
                else deepsic_loss_callback

            self.runtime = time.time()
            self.model.fit(rx=rx, labels=labels, num_epochs=self.params['training_epochs'],
                           lr=self.params['training_lr'],
                           batch_size=self.params['training_batch_size'],
                           callback=partial(loss_callback, losses=losses))
            self.runtime = time.time() - self.runtime

            rx, labels = prepare_single_batch(channel=self.channel,
                                              num_samples=self.params['test_dim'],
                                              frame_idx=0,
                                              snr=self.params['warm_start_snr'])
            ber, confidence = self.model.test_model(rx=rx, labels=labels)

            with open(os.path.join(warm_start_path, f"{self.params['constellation']}_"
                                                    f"{self.params['num_users']}_"
                                                    f"{self.params['num_layers']}_"
                                                    f"{self.params['hidden_size']}.txt"), 'w') as file:

                file.write(f"Parameters\nTraining size: {self.params['training_dim']}\n"
                           f"Epochs: {self.params['training_epochs']}\nBatch size: {self.params['training_batch_size']}"
                           f"\nLearning Rate: {self.params['training_lr']}\n\n")
                file.write(f"Results\nBER: {ber.item()}\nConfidence: {confidence.item()}\nRuntime: {self.runtime}")

            self.model.save_model(os.path.join(warm_start_path, f"{self.params['constellation']}_"
                                                                f"{self.params['num_users']}_"
                                                                f"{self.params['num_layers']}_"
                                                                f"{self.params['hidden_size']}.pt"))

            fig, axs = plt.subplots(self.params['num_users'], self.params['num_layers'])
            for user_idx in range(self.params['num_users']):
                for layer_idx in range(self.params['num_layers']):
                    axs[user_idx, layer_idx].plot(losses[user_idx][layer_idx])
                    axs[user_idx, layer_idx].set_title(f"User {user_idx}, Layer {layer_idx}")
            plt.suptitle("Per-block warm start loss curves")
            fig.set_size_inches(16, 16)
            if not os.path.exists(os.path.join(warm_start_path, "loss_curves")):
                os.makedirs(os.path.join(warm_start_path, "loss_curves"))

            plt.savefig(os.path.join(warm_start_path, "loss_curves", f"{self.params['constellation']}_"
                                                                     f"{self.params['num_users']}_"
                                                                     f"{self.params['num_layers']}_"
                                                                     f"{self.params['hidden_size']}.svg"),
                        dpi=200, format='svg')

        if self.verbose:
            print("Warm start loaded")

    def _track(self):
        self.runtime = time.time()
        if self.tracker is not None:
            loader = prepare_experiment_data(channel=self.channel,
                                             num_pilots=self.params['num_pilots'],
                                             num_frames=self.params['num_blocks'],
                                             snr=self.params['tracking_snr'])
            self.tracker.run(loader)
        self.runtime = time.time() - self.runtime

    def _test(self):
        rx, labels = prepare_single_batch(channel=self.channel,
                                          num_samples=self.params['test_dim'],
                                          frame_idx=self.params['num_blocks'] - 1,
                                          snr=self.params['tracking_snr'])
        ber, confidence = self.model.test_model(rx=rx, labels=labels)
        if self.params['tracking_method'] != 'Nothing':
            skip_ratio = (self.tracker.skip_counter.total_skipped /
                          (self.params['num_blocks'] * self.params['num_users'] * self.params['num_layers']))
        else:
            skip_ratio = '-'
        return {'ber': ber.item(), 'confidence': confidence.item(), 'skip_ratio': skip_ratio, 'run_time': self.runtime}

    def _log(self, metrics):
        log_path = os.path.join(RESULTS_DIR, 'DeepSIC', 'database.csv')
        results_dict = self.params | metrics
        append_dict_to_csv(data_dict=results_dict, file_name=log_path)


if __name__ == '__main__':

    path = os.path.join(EXP_DIR, 'DeepSIC')
    config_path = Config(os.path.join(path, 'deepsic_config.yaml'))
    run_configurations = deepsic_parameter_combinations(config_path)

    for params in run_configurations:

        try:
            exp = DeepsicExperiment(params, verbose=True)
        except ValueError as error:
            print(error)
            continue

        print(params)
        exp.run()
