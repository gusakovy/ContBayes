from functools import partial
import torch
from torch import Tensor
import torch.utils.data as data
from torch.utils.data import DataLoader
import pyro
import pyro.distributions as dist
import tyxe
from contbayes.BNN.BayesNN import FullCovBNN
from contbayes.Detectors.DeepSICblock import DeepSICblock

MODULATION_TYPES = ["BPSK", "QPSK"]


class BayesianDeepSIC:
    """
    Bayesian DeepSIC model where every block is a BayesNN module.

    :param modulation_type: modulation type ("BPSK" or "QPSK")
    :param num_users: number of users
    :param num_ant: number of receive antennas
    :param num_iterations: number of soft interference cancellation (SIC) iterations
    :param hidden_dim: size of the hidden layer of each DeepSIC block
    :param verbose: whether to print the structure of a DeepSIC block after initialization
    """

    def __init__(self,
                 modulation_type: str,
                 num_users: int,
                 num_ant: int,
                 num_iterations: int,
                 hidden_dim: int,
                 verbose: bool = False):
        if modulation_type not in MODULATION_TYPES:
            raise ValueError(f"Modulation type must be one of {MODULATION_TYPES}.")
        self.modulation_type = modulation_type
        self.num_classes = 2 if self.modulation_type == "BPSK" else 4
        self.num_users = num_users
        self.num_ant = num_ant
        self.num_layers = num_iterations
        self.rx_size = num_ant if self.modulation_type == "BPSK" else 2 * self.num_ant
        self.verbose = verbose
        prior = tyxe.priors.IIDPrior(dist.Normal(0, 1))
        observation_model = tyxe.likelihoods.Categorical(dataset_size=1000, logit_predictions=False)
        self.bnn_block = FullCovBNN(model_type="classifier",
                                    output_dim=self.num_classes,
                                    net_builder=partial(DeepSICblock,
                                                        modulation_type=self.modulation_type,
                                                        num_users=self.num_users,
                                                        num_ant=self.num_ant,
                                                        hidden_dim=hidden_dim),
                                    prior=prior,
                                    likelihood=observation_model)
        # init parameters
        if self.verbose:
            print(self.bnn_block.torch_net)
        zero_input = torch.zeros(1, self.rx_size + (self.num_classes - 1) * (self.num_users-1))
        with torch.no_grad():
            self.bnn_block.predict(zero_input)
        self.param_matrix = [[self.bnn_block.get_params() for _ in range(self.num_layers)]
                             for _ in range(self.num_users)]

    def pred_and_rx_to_input(self, layer_num: int, rx: Tensor, pred: Tensor = None) -> Tensor:
        """
        Prepare input for Bayesian DeepSIC model.

        :param layer_num: layer number
        :param rx: receive input
        :param pred: prediction of the previous layer
        :return: input for the BayesianDeepSIC model
        """

        if layer_num == 0:
            input_dim = rx.size(0)
            initial_pred = (1 / self.num_classes) * torch.ones(1, (self.num_classes - 1) * self.num_users)
            inputs = torch.cat([initial_pred.repeat(input_dim, 1), rx], dim=1)
        else:
            assert rx.size(0) == pred.size(0)
            inputs = torch.cat([pred, rx], dim=1)
        return inputs

    def layer_transition(self, layer_num: int, rx: Tensor, pred: Tensor = None, truncate: bool = True) -> Tensor:
        """
        Pass the data through a layer of the Bayesian DeepSIC model.

        :param layer_num: layer number
        :param rx: receive input
        :param pred: prediction of the previous layer
        :param truncate: whether to truncate the input to the number of classes (usually done for the last layer)
        :return: output of the layer (aggregated predictions of the blocks in the layer)
        """

        inputs = self.pred_and_rx_to_input(layer_num, rx, pred)
        if truncate:
            layer_outputs = torch.zeros(rx.size(0), (self.num_classes - 1) * self.num_users)
        else:
            layer_outputs = torch.zeros(rx.size(0), self.num_classes * self.num_users)

        for user_idx in range(self.num_users):
            user_indices = range((self.num_classes - 1) * user_idx, (self.num_classes - 1) * (user_idx + 1))
            self.bnn_block.set_params(self.param_matrix[user_idx][layer_num])
            user_inputs = inputs[..., [i for i in range(self.rx_size + (self.num_classes - 1) * self.num_users)
                                       if i not in user_indices]]
            if truncate:
                layer_outputs[..., user_indices] = self.bnn_block.predict(user_inputs)[..., :(self.num_classes - 1)]
            else:
                user_indices = range(self.num_classes * user_idx, self.num_classes * (user_idx + 1))
                layer_outputs[..., user_indices] = self.bnn_block.predict(user_inputs)

        return layer_outputs.detach()

    def predict(self, rx: Tensor) -> Tensor:
        """
        Forward pass of the Bayesian DeepSIC model.

        :param rx: receive input
        :return: output of the DeepSIC model (aggregated predictions of the blocks in the last layer)
        """

        rx = torch.atleast_2d(rx)
        predictions = None
        for layer_idx in range(self.num_layers):
            predictions = self.layer_transition(layer_idx, rx, predictions, truncate=(layer_idx < self.num_layers - 1))
        return predictions.view(-1, self.num_users, self.num_classes)

    def _train_block(self, layer_num: int, user_num: int, dataloader: DataLoader, num_epochs: int = 250,
                     lr: float = 1e-3, callback: callable = None):
        """Train a single block of the Bayesian DeepSIC model using Stochastic Variational Inference (SVI)."""

        if self.verbose:
            print(f"Training block [user = {user_num}][layer = {layer_num}]")
        optimizer = pyro.optim.Adam({"lr": lr})
        self.bnn_block.set_params(self.param_matrix[user_num][layer_num])
        self.bnn_block.fit(data_loader=dataloader,
                           optim=optimizer,
                           num_epochs=num_epochs,
                           callback=None if callback is None else partial(callback, user=user_num, layer=layer_num))
        self.param_matrix[user_num][layer_num] = self.bnn_block.get_params()

    def _train_layer(self, layer_num: int, rx: Tensor, labels: Tensor, pred: Tensor = None, num_epochs: int = 250,
                     lr: float = 1e-3, batch_size: int = None, callback: callable = None) -> Tensor:
        """Train a layer of the Bayesian DeepSIC model using Stochastic Variational Inference (SVI)."""

        inputs = self.pred_and_rx_to_input(layer_num, rx, pred)
        if batch_size is None:
            batch_size = len(inputs)

        for user_idx in range(self.num_users):
            user_indices = range((self.num_classes - 1) * user_idx, (self.num_classes - 1) * (user_idx + 1))

            user_inputs = inputs[..., [i for i in range(self.rx_size + (self.num_classes - 1) * self.num_users)
                                       if i not in user_indices]]
            user_labels = labels[..., user_idx].squeeze()
            dataset = data.TensorDataset(user_inputs, user_labels)
            loader = data.DataLoader(dataset, batch_size=batch_size)
            self._train_block(layer_num=layer_num,
                              user_num=user_idx,
                              dataloader=loader,
                              num_epochs=num_epochs,
                              lr=lr,
                              callback=callback)

        return self.layer_transition(layer_num, rx, pred)

    def fit(self, rx: Tensor, labels: Tensor, num_epochs: int = 250, lr: float = 1e-3, batch_size: int = None,
            callback: callable = None):
        """
        Train the Bayesian DeepSIC model using Stochastic Variational Inference (SVI).

        :param rx: receive input
        :param labels: labels of the data
        :param num_epochs: number of training epochs
        :param lr: learning rate
        :param batch_size: batch size
        :param callback: callback function with input variables bnn, i, e, user_num, layer_num (passed into _train_block
             for each DeepSIC block)
        """

        predictions = None
        for layer_idx in range(self.num_layers):
            predictions = self._train_layer(layer_num=layer_idx,
                                            rx=rx,
                                            labels=labels,
                                            pred=predictions,
                                            num_epochs=num_epochs,
                                            lr=lr,
                                            batch_size=batch_size,
                                            callback=callback)

    def test_model(self, rx: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
        """
        Compute the bit error rate and confidence of the model on the test data.

        :param rx: receive input
        :param labels: labels of the data
        :return: bit error rate and confidence of the model on the test data
        """

        predictions = self.predict(rx)
        confidence = predictions.max(-1).values.mean()
        hard_decisions = predictions.argmax(-1)
        bit_errors = torch.sum(torch.abs(hard_decisions - labels) % 3)
        bits_per_symbol = 1 if self.modulation_type == "BPSK" else 2
        bit_error_rate = bit_errors / (labels.numel() * bits_per_symbol)
        return bit_error_rate, confidence

    def save_model(self, path: str):
        """
        Save the model to disk.

        :param path: path to save the model to
        """

        torch.save(self.param_matrix, path)

    def load_model(self, path: str):
        """
        Load the model from disk.

        :param path: path to load the model from
        """

        self.param_matrix = torch.load(path)
