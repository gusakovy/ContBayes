from contbayes.BNN.BayesNN import FullCovBNN
from functools import partial

import torch
from torch import Tensor
import torch.nn as nn
import torch.utils.data as data

import pyro
import pyro.distributions as dist
import pyro.infer.autoguide as ag

import tyxe

HIDDEN_BASE_SIZE = 8
NUM_EPOCHS = 500
LR = 4e-3
MODULATION_TYPE = "BPSK"


class DeepSicNetBuilder(nn.Module):
    def __init__(self, num_classes, num_users, num_ant):
        super(DeepSicNetBuilder, self).__init__()
        hidden_size = HIDDEN_BASE_SIZE * num_classes
        rx_size = num_ant if MODULATION_TYPE == "BPSK" else 2 * num_ant
        linear_input = rx_size + (num_classes - 1) * (num_users - 1)
        self.fc1 = nn.Linear(linear_input, hidden_size)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out0 = self.activation(self.fc1(inputs))
        out1 = self.softmax(self.fc2(out0))
        return out1


class BayesianDeepSIC:
    def __init__(self,
                 num_classes: int,
                 num_users: int,
                 num_ant: int,
                 num_iterations: int):
        self.num_classes = num_classes
        self.num_users = num_users
        self.num_ant = num_ant
        self.num_layers = num_iterations
        prior = tyxe.priors.IIDPrior(dist.Normal(0, 1))
        observation_model = tyxe.likelihoods.Categorical(dataset_size=1000, logit_predictions=False)
        guide = partial(ag.AutoMultivariateNormal, init_scale=1e-4)
        self.bnn_block = FullCovBNN(model_type="classifier",
                                    output_dim=4,
                                    net_builder=partial(DeepSicNetBuilder,
                                                        num_classes=self.num_classes,
                                                        num_users=self.num_users,
                                                        num_ant=self.num_ant),
                                    prior=prior,
                                    likelihood=observation_model,
                                    guide_builder=guide)
        # init parameters by
        print(self.bnn_block.torch_net)
        zero_input = torch.zeros(1, self.num_ant + (self.num_classes - 1) * (self.num_users-1))
        self.bnn_block.predict(zero_input)
        self.initial_pred = (1 / self.num_classes) * torch.ones(1, (self.num_classes - 1) * self.num_users)
        self.param_matrix = [[self.bnn_block.get_params() for _ in range(self.num_layers)]
                             for _ in range(self.num_users)]

    def _pred_and_rx_to_input(self, layer_num, rx, pred=None):
        if layer_num == 0:
            input_dim = rx.size(0)
            inputs = torch.cat([self.initial_pred.repeat(input_dim, 1), rx], dim=1)
        else:
            assert rx.size(0) == pred.size(0)
            inputs = torch.cat([pred, rx], dim=1)
        return inputs

    def _layer_transition(self, layer_num, rx, pred=None, truncate=True) -> Tensor:
        inputs = self._pred_and_rx_to_input(layer_num, rx, pred)
        if truncate:
            layer_outputs = torch.zeros(rx.size(0), (self.num_classes - 1) * self.num_users)
        else:
            layer_outputs = torch.zeros(rx.size(0), self.num_classes * self.num_users)

        for user_idx in range(self.num_users):
            user_indices = range((self.num_classes - 1) * user_idx, (self.num_classes - 1) * (user_idx + 1))
            self.bnn_block.set_params(self.param_matrix[user_idx][layer_num])
            user_inputs = inputs[..., [i for i in range(self.num_ant + (self.num_classes - 1) * self.num_users)
                                       if i not in user_indices]]
            if truncate:
                layer_outputs[..., user_indices] = self.bnn_block.predict(user_inputs)[..., :(self.num_classes - 1)]
            else:
                user_indices = range(self.num_classes * user_idx, self.num_classes * (user_idx + 1))
                layer_outputs[..., user_indices] = self.bnn_block.predict(user_inputs)

        return layer_outputs

    def predict(self, rx) -> Tensor:
        rx = torch.atleast_2d(rx)
        predictions = None
        for layer_idx in range(self.num_layers):
            predictions = self._layer_transition(layer_idx, rx, predictions, truncate=(layer_idx < self.num_layers - 1))
        return predictions.view(-1, self.num_users, self.num_classes)

    def _train_block(self, layer_num, user_num, dataloader, num_epochs=NUM_EPOCHS) -> None:
        print(f"Training block [user = {user_num}][layer = {layer_num}]")
        optimizer = pyro.optim.Adam({"lr": LR})
        self.bnn_block.set_params(self.param_matrix[user_num][layer_num])
        self.bnn_block.fit(dataloader, optimizer, num_epochs)
        self.param_matrix[user_num][layer_num] = self.bnn_block.get_params()

    def _train_layer(self, layer_num, tx, rx, pred=None) -> Tensor:
        inputs = self._pred_and_rx_to_input(layer_num, rx, pred)

        for user_idx in range(self.num_users):
            user_indices = range((self.num_classes - 1) * user_idx, (self.num_classes - 1) * (user_idx + 1))
            user_inputs = inputs[..., [i for i in range(self.num_ant + (self.num_classes - 1) * self.num_users)
                                       if i not in user_indices]]
            user_labels = tx[..., user_idx].squeeze()
            dataset = data.TensorDataset(user_inputs, user_labels)
            loader = data.DataLoader(dataset, batch_size=len(inputs) // 5)
            self._train_block(layer_num=layer_num, user_num=user_idx, dataloader=loader)

        return self._layer_transition(layer_num, rx, pred)

    def fit(self, tx, rx) -> None:
        predictions = None
        for layer_idx in range(self.num_layers):
            predictions = self._train_layer(layer_idx, tx, rx, predictions)

    def test_model(self, tx, rx):
        predictions = self.predict(rx)
        confidence = predictions.max(-1).values.mean()
        hard_decisions = predictions.argmax(-1)
        bit_errors = torch.sum(torch.abs(hard_decisions - tx) % 4)
        bits_per_symbol = 1 if MODULATION_TYPE == "BPSK" else 2
        bit_error_rate = bit_errors / (tx.numel() * bits_per_symbol)
        return bit_error_rate, confidence

