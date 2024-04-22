import torch.nn as nn
from torch import Tensor


class DeepSICblock(nn.Module):
    """
    Single block of a DeepSIC model.

    :param modulation_type: modulation type ("BPSK" or "QPSK")
    :param num_users: number of users
    :param num_ant: number of antennas
    :param hidden_dim: size of the hidden layer
    """

    def __init__(self, modulation_type: str, num_users: int, num_ant: int, hidden_dim: int):
        super(DeepSICblock, self).__init__()
        num_classes = 2 if modulation_type == "BPSK" else 4
        rx_size = num_ant if modulation_type == "BPSK" else 2 * num_ant
        linear_input = rx_size + (num_classes - 1) * (num_users - 1)
        self.fc1 = nn.Linear(linear_input, hidden_dim)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Forward pass of the network.

        :param inputs: inputs tensor
        :return: outputs tensor
        """

        out0 = self.activation(self.fc1(inputs))
        out1 = self.softmax(self.fc2(out0))
        return out1
