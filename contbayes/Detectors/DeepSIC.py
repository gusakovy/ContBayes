from config import MODULATION_TYPE
from contbayes.Detectors.DeepSICblock import DeepSICblock

import torch
from torch import Tensor
import torch.utils.data as data
from torch.nn.functional import one_hot


class DeepSIC:
    def __init__(self,
                 num_classes: int,
                 num_users: int,
                 num_ant: int,
                 num_iterations: int,
                 hidden_dim: int,
                 verbose: bool = False):
        self.num_classes = num_classes
        self.num_users = num_users
        self.num_ant = num_ant
        self.num_layers = num_iterations
        self.rx_size = num_ant if MODULATION_TYPE == "BPSK" else 2 * self.num_ant
        self.verbose = verbose
        self.detector = [[DeepSICblock(num_classes=self.num_classes,
                                       num_users=self.num_users,
                                       num_ant=self.num_ant,
                                       hidden_dim=hidden_dim) for _ in range(self.num_layers)]
                         for _ in range(self.num_users)]

    def pred_and_rx_to_input(self, layer_num, rx, pred=None):
        if layer_num == 0:
            input_dim = rx.size(0)
            initial_pred = (1 / self.num_classes) * torch.ones(1, (self.num_classes - 1) * self.num_users)
            inputs = torch.cat([initial_pred.repeat(input_dim, 1), rx], dim=1)
        else:
            assert rx.size(0) == pred.size(0)
            inputs = torch.cat([pred, rx], dim=1)
        return inputs

    def layer_transition(self, layer_num, rx, pred=None, truncate=True) -> Tensor:
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

    def predict(self, rx) -> Tensor:
        rx = torch.atleast_2d(rx)
        predictions = None
        for layer_idx in range(self.num_layers):
            predictions = self.layer_transition(layer_idx, rx, predictions, truncate=(layer_idx < self.num_layers - 1))
        return predictions.view(-1, self.num_users, self.num_classes)

    def _train_block(self, layer_num, user_num, dataloader, num_epochs=250, lr=7e-4, callback=None) -> None:
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
                callback(i, self.detector[user_num][layer_num], running_loss, user_num, layer_num)

    def _train_layer(self, layer_num, rx, labels, pred=None, num_epochs=250,
                     lr=7e-4, batch_size=None, callback=None) -> Tensor:

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

    def fit(self, rx, labels, num_epochs=250, lr=7e-4, batch_size=None, callback=None) -> None:
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

    def test_model(self, rx, labels):
        predictions = self.predict(rx)
        confidence = predictions.max(-1).values.mean()
        hard_decisions = predictions.argmax(-1)
        bit_errors = torch.sum(torch.abs(hard_decisions - labels) % 4)
        bits_per_symbol = 1 if MODULATION_TYPE == "BPSK" else 2
        bit_error_rate = bit_errors / (labels.numel() * bits_per_symbol)
        return bit_error_rate, confidence

    def save_model(self, path):
        torch.save(self.detector, path)

    def load_model(self, path):
        self.detector = torch.load(path)
