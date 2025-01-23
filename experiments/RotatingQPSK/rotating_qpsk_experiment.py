import os
import math
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.utils.data as data
import pyro
import pyro.distributions as dist
import tyxe
from contbayes.BNN.BayesNN import FullCovBNN
from contbayes.Trackers.EKF import EKF
from constellation_circle_dynamics import ConstellationCircleDynamics, plot_decision_zones
from dir_definitions import RESULTS_DIR

save_dir = os.path.join(RESULTS_DIR, 'RotatingQPSK')
os.makedirs(save_dir, exist_ok=True)

class Receiver(nn.Module):
    def __init__(self):
        super(Receiver, self).__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 4)
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        out1 = self.activation(self.fc1(inputs))
        out2 = self.softmax(self.fc2(out1))
        return out2

    def predict(self, inputs, **kwargs):
        return self.forward(inputs)


def test_model(model, inputs, labels):
    true_labels = []
    pred_labels = []
    probabilities = []

    model.eval()
    with torch.no_grad():
        output = model.predict(inputs, num_predictions=32)
        probabilities.extend(output.max(-1).values.tolist())
        true_labels.extend(labels.squeeze().tolist())
        pred_labels.extend(output.argmax(-1).tolist())

    # Calculate accuracy
    accuracy = sum(1 for true, pred in zip(true_labels, pred_labels) if true == pred) / len(true_labels)

    return accuracy


def reset_bayesian_receiver():
    prior = tyxe.priors.IIDPrior(dist.Normal(0, 1))
    observation_model = tyxe.likelihoods.Categorical(dataset_size=4000, logit_predictions=False)
    pyro.clear_param_store()
    bnn = FullCovBNN(
        model_type="classifier",
        output_dim=4,
        net_builder=Receiver,
        prior=prior,
        likelihood=observation_model,
        init_cov_scale=1e-4
    )
    bnn.predict(torch.tensor([1.0, 1.0]))

    return bnn


def lf_vcl_trainer(model, dataloader, num_epochs):
    optim = pyro.optim.Adam({"lr": 5e-4})
    for obs in tqdm(dataloader):
        x_train, y_train = obs
        train_dataset = data.TensorDataset(x_train, y_train)
        train_loader = data.DataLoader(train_dataset, batch_size=len(x_train))
        model.fit(train_loader, optim, num_epochs)


def sgd_trainer(model, dataloader, num_epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    for obs in tqdm(dataloader):
        x_train, y_train = obs
        for j in range(num_epochs):
            optimizer.zero_grad()
            outputs = model(x_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()


def generate_experiment_data(channel, num_frames, log_num_obs):

    label_blocks = torch.zeros(0, 0)
    receive_blocks = torch.zeros(0, 0)
    for t in range(num_frames):
        rx, labels = channel.generate_samples(n=2 ** log_num_obs, randomize=True)
        label_blocks = labels if t == 0 else torch.cat([label_blocks, labels])
        receive_blocks = rx if t == 0 else torch.cat([receive_blocks, rx])

    dataset = data.TensorDataset(receive_blocks, label_blocks)
    dataloader = data.DataLoader(dataset, batch_size=int(2 ** log_num_obs), shuffle=False)

    return dataloader


def run_acc_vs_obs_experiment(
    snr: float,
    speed: float,
    num_frames: int,
    log_num_obs: list,
    num_epochs: int,
    num_experiments: int = 10,
):
    pyro.set_rng_seed(0)

    num_observations = 2**np.array(log_num_obs)
    optimal_accuracy = ((1 - 1/2 * math.erfc(math.sqrt(snr)/2))**2)

    methods = ["EKF", "LF-VCL", "CL", "Optimal Detector"]
    accuracy_array = np.zeros((len(methods), len(log_num_obs), num_experiments))
    runtime_array = np.zeros((len(methods), len(log_num_obs), num_experiments))

    print("Accuracy vs. number of observations experiment")
    print("==============================================")
    for obs_idx, log_obs in enumerate(log_num_obs):
        for exp_idx in range(num_experiments):

            print(f"--------------------------------------------------\n"
                  f"Number of observations = {2**log_obs} | Experiment {exp_idx+1}\n"
                  f"--------------------------------------------------")
            channel = ConstellationCircleDynamics(n_points=4, noise_var=1/snr, speed=speed, energy=1)
            loader = generate_experiment_data(channel, num_frames, log_obs)
            method_idx = 0

            # EKF
            print(f"{methods[method_idx]}")
            BayesianReceiver = reset_bayesian_receiver()
            Kalman_Filter = EKF(
                obs_model=BayesianReceiver,
                state_model=1,
                process_noise_var=1e-4,
                obs_reduction=None,
            )
            start = time.time()
            Kalman_Filter.run(dataloader=loader, callback=None)
            end = time.time()
            runtime_array[method_idx, obs_idx, exp_idx] = end-start
            test_rx, test_labels = channel.generate_samples(10000, advance=False)
            acc = test_model(Kalman_Filter.obs_model, test_rx, test_labels)
            accuracy_array[method_idx, obs_idx, exp_idx] = acc
            method_idx += 1


            # LF-VCL
            print(f"{methods[method_idx]}")
            BayesianReceiver = reset_bayesian_receiver()
            start = time.time()
            lf_vcl_trainer(BayesianReceiver, loader , num_epochs)
            end = time.time()
            runtime_array[method_idx, obs_idx, exp_idx] = end-start
            test_rx, test_labels = channel.generate_samples(10000, advance=False)
            acc = test_model(BayesianReceiver, test_rx, test_labels)
            accuracy_array[method_idx, obs_idx, exp_idx] = acc
            method_idx += 1

            # CL
            print(f"{methods[method_idx]}")
            FrequentistReceiver = Receiver()
            start = time.time()
            sgd_trainer(FrequentistReceiver, loader, num_epochs)
            end = time.time()
            runtime_array[method_idx, obs_idx, exp_idx] = end-start
            test_rx, test_labels = channel.generate_samples(10000, advance=False)
            acc = test_model(FrequentistReceiver, test_rx, test_labels)
            accuracy_array[method_idx, obs_idx, exp_idx] = acc
            method_idx += 1

            accuracy_array[method_idx, obs_idx, exp_idx] = optimal_accuracy
            runtime_array[method_idx, obs_idx, exp_idx] = 0

    accuracy_array = accuracy_array.mean(axis=-1)
    runtime_array = runtime_array.mean(axis=-1)

    plt.figure()
    print(f"Accuracies: {accuracy_array}")
    for i, method in enumerate(methods):
        plt.plot(
            num_observations,
            accuracy_array[i, :],
            marker='o',
            label=methods[i] + (f"-{num_epochs}" if "CL" in methods[i] else "")
        )
    plt.legend()
    plt.xscale("log", base=2)
    plt.xlabel("Number of Observations")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Number of Observations")
    plt.savefig(os.path.join(save_dir, "accuracy vs number of observations.png"))


    plt.figure()
    for i, method in enumerate(methods):
        plt.plot(num_observations, runtime_array[i], label=methods[i])
    plt.legend()
    plt.xscale("log", base=2)
    plt.xlabel("Number of Observations")
    plt.ylabel("Runtime[s]")
    plt.title("Runtime vs. Number of Observations")
    plt.savefig(os.path.join(save_dir, "runtime vs number of observations.png"))


def run_decision_zones_plotter(
    snr: float,
    speed: float,
    num_frames: int,
    log_num_obs: int,
    num_epochs: int
):
    print("Decision zones plotter")
    print("======================")
    pyro.set_rng_seed(0)

    methods = ["EKF", "LF-VCL", "CL"]
    channel = ConstellationCircleDynamics(n_points=4, noise_var=1 / snr, speed=speed, energy=1)
    loader = generate_experiment_data(channel, num_frames, log_num_obs)
    fig, axs = plt.subplots(1, len(methods))
    method_idx = 0

    # EKF
    print(f"{methods[method_idx]}")
    BayesianReceiver = reset_bayesian_receiver()
    Kalman_Filter = EKF(
        obs_model=BayesianReceiver,
        state_model=1,
        process_noise_var=5e-4,
        obs_reduction=None,
    )
    Kalman_Filter.run(dataloader=loader, callback=None)
    plot_decision_zones(Kalman_Filter.obs_model, channel, axis=axs[method_idx])
    axs[method_idx].title.set_text(f"Bayesian model EKF")
    method_idx += 1


    # LF-VCL
    print(f"{methods[method_idx]}")
    BayesianReceiver = reset_bayesian_receiver()
    lf_vcl_trainer(BayesianReceiver, loader, num_epochs)
    plot_decision_zones(BayesianReceiver, channel, axis=axs[method_idx])
    axs[method_idx].title.set_text(f"Bayesian model VCL-{num_epochs}")
    method_idx += 1

    # CL
    print(f"{methods[method_idx]}")
    FrequentistReceiver = Receiver()
    sgd_trainer(FrequentistReceiver, loader, num_epochs)
    plot_decision_zones(FrequentistReceiver, channel, axis=axs[method_idx])
    axs[method_idx].title.set_text(f"Non-Bayesian model CL-{num_epochs}")
    method_idx += 1


    fig.set_size_inches(len(methods) * 5, 4)
    fig.show()
    fig.savefig(
        os.path.join(save_dir, f"decision-zones-{2**log_num_obs}-{snr}-{num_frames}.png"),
        bbox_inches='tight',
        dpi=300
    )


if __name__ == "__main__":
    run_acc_vs_obs_experiment(
        snr = 16,
        speed=5e-4,
        num_frames=500,
        log_num_obs = [2, 4 ,6, 8],
        num_epochs=4,
        num_experiments=5
    )
    run_decision_zones_plotter(
        snr = 16,
        speed=5e-4,
        num_frames=500,
        log_num_obs=4,
        num_epochs=4
    )
