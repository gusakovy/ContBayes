from config import MODULATION_TYPE
import torch
import torch.nn as nn


class DeepSICblock(nn.Module):
    def __init__(self, num_classes, num_users, num_ant, hidden_dim):
        super(DeepSICblock, self).__init__()
        hidden_size = hidden_dim * num_classes
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
