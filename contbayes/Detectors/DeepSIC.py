import torch
from torch import Tensor
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.nn.functional import one_hot
from contbayes.Detectors.DeepSICblock import DeepSICblock

MODULATION_TYPES = ["BPSK", "QPSK"]


class DeepSIC:
    """
    DeepSIC model as defined in https://ieeexplore.ieee.org/document/9242305.

    :param modulation_type: modulation type ("BPSK" or "QPSK")
    :param num_users: number of users
    :param num_ant: number of receive antennas
    :param num_iterations: number of soft interference cancellation (SIC) iterations
    :param hidden_dim: size of the hidden layer of each DeepSIC block
    :param verbose: whether to print the structure of the DeepSIC block after initialization
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
        self.detector = [[DeepSICblock(modulation_type=self.modulation_type,
                                       num_users=self.num_users,
                                       num_ant=self.num_ant,
                                       hidden_dim=hidden_dim) for _ in range(self.num_layers)]
                         for _ in range(self.num_users)]

    def pred_and_rx_to_input(self, layer_num: int, rx: Tensor, pred: Tensor = None) -> Tensor:
        """
        Prepare input for DeepSIC model.

        :param layer_num: layer number
        :param rx: receive input
        :param pred: prediction of the previous layer
        :return: input for the DeepSIC model
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
        Pass the data through a layer of the DeepSIC model.

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
            user_inputs = inputs[..., [i for i in range(self.rx_size + (self.num_classes - 1) * self.num_users)
                                       if i not in user_indices]]
            if truncate:
                layer_outputs[..., user_indices] = \
                    self.detector[user_idx][layer_num](user_inputs)[..., :(self.num_classes - 1)]
            else:
                user_indices = range(self.num_classes * user_idx, self.num_classes * (user_idx + 1))
                layer_outputs[..., user_indices] = self.detector[user_idx][layer_num](user_inputs)

        return layer_outputs.detach()

    def predict(self, rx: Tensor) -> Tensor:
        """
        Forward pass of the DeepSIC model.

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
        """Train a single block of the DeepSIC model using Stochastic Gradient Descent (SGD)."""

        if self.verbose:
            print(f"Training block [user = {user_num}][layer = {layer_num}]")
        optimizer = torch.optim.Adam(self.detector[user_num][layer_num].parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        for i in range(num_epochs):
            running_loss = 0
            for batch in dataloader:
                rx, labels = batch
                optimizer.zero_grad()
                pred = self.detector[user_num][layer_num].forward(rx)
                loss = criterion(input=pred, target=labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            if callback is not None:
                callback(epoch_num=i, net=self.detector[user_num][layer_num], loss=running_loss,
                         user_num=user_num, layer_num=layer_num)

    def _train_layer(self, layer_num: int, rx: Tensor, labels: Tensor, pred: Tensor = None, num_epochs: int = 250,
                     lr: float = 1e-3, batch_size: int = None, callback: callable = None) -> Tensor:
        """Train a layer of the DeepSIC model using Stochastic Gradient Descent (SGD)."""

        inputs = self.pred_and_rx_to_input(layer_num, rx, pred)
        if batch_size is None:
            batch_size = len(inputs)

        for user_idx in range(self.num_users):
            user_indices = range((self.num_classes - 1) * user_idx, (self.num_classes - 1) * (user_idx + 1))

            user_inputs = inputs[..., [i for i in range(self.rx_size + (self.num_classes - 1) * self.num_users)
                                       if i not in user_indices]]
            user_labels = one_hot(labels[..., user_idx].squeeze(), self.num_classes).type(Tensor)
            dataset = data.TensorDataset(user_inputs, user_labels)
            loader = data.DataLoader(dataset, batch_size=batch_size)
            self._train_block(layer_num=layer_num,
                              user_num=user_idx,
                              dataloader=loader,
                              num_epochs=num_epochs,
                              lr=lr,
                              callback=callback)

        return self.layer_transition(layer_num, rx, pred)

    def fit(self, rx: Tensor, labels: Tensor, num_epochs: int = 250, lr: float = 1e-3,
            batch_size: int = None, callback: callable = None, **kwargs):
        """
        Train the DeepSIC model using Stochastic Gradient Descent (SGD).

        :param rx: receive input
        :param labels: labels of the data
        :param num_epochs: number of training epochs
        :param lr: learning rate
        :param batch_size: batch size
        :param callback: callback function with input variables epoch_num, net, loss, user_num, layer_num (passed into
             _train_block for each DeepSIC block)
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

        torch.save(self.detector, path)

    def load_model(self, path: str):
        """
        Load the model from disk.

        :param path: path to load the model from
        """

        self.detector = torch.load(path)
