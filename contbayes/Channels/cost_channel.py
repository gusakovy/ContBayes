import os
import math
import scipy.io
import numpy as np
import torch
from torch import Tensor
from dir_definitions import RESOURCES_DIR

SCALING_COEFFICIENT = 0.25
MAX_FRAMES = 25

COST2100_DIR = os.path.join(RESOURCES_DIR, 'cost2100_channel')
MODULATION_TYPES = ["BPSK", "QPSK"]

class Cost2100Channel:
    """
    COST2100 memory-less MIMO channel.

    :param modulation_type: modulation type ("BPSK" or "QPSK")
    :param linear_channel: whether the channel is linear or non-linear
    """

    def __init__(self, modulation_type: str, num_users: int, num_antennas: int,
                 fading_coefficient: float, linear_channel: bool = False):
        if modulation_type not in MODULATION_TYPES:
            raise ValueError(f"Modulation type must be one of {MODULATION_TYPES}.")
        self.modulation_type = modulation_type
        self.num_users = num_users
        self.num_antennas = num_antennas
        self.constellation_points = torch.tensor([-1, 1], dtype=torch.float) if self.modulation_type == "BPSK" \
            else torch.tensor([[math.cos(math.pi * (k + 1 / 2) / 2) +
                                math.sin(math.pi * (k + 1 / 2) / 2) * 1j] for k in range(4)], dtype=torch.complex64)
        self.fading_coefficient = fading_coefficient
        self.linear_channel = linear_channel


    def _calculate_channel(self, frame_ind: int) -> Tensor:
        """Calculate the channel information matrix."""

        total_h = torch.zeros(self.num_users, self.num_antennas)
        main_folder = 1 + (frame_ind // MAX_FRAMES)
        for i in range(1, self.num_users + 1):
            path_to_mat = os.path.join(COST2100_DIR, f'{main_folder}', f'h_{i}.mat')
            h_user = torch.tensor(scipy.io.loadmat(path_to_mat)['norm_channel'][frame_ind % MAX_FRAMES,:self.num_users],
                                  dtype=torch.float)
            total_h[i - 1] = SCALING_COEFFICIENT * h_user / self.fading_coefficient

        total_h[np.arange(self.num_users), np.arange(self.num_users)] = 1
        return total_h

    @staticmethod
    def _compute_channel_signal_convolution(h: Tensor, tx: Tensor) -> Tensor:
        """Compute the convolution of a channel matrix and a tensor of constellation points."""

        conv = h @ tx.T
        return conv

    def transmit(self, s: Tensor, snr: float, frame_ind: int = 0) -> Tensor:
        """
        Simulate transmission of symbols.

        :param s: symbols to be transmitted
        :param snr: signal-to-noise ratio
        :param frame_ind: time frame index
        :return: transmitted signal
        """

        h = self._calculate_channel(frame_ind)
        tx = self.constellation_points[s].squeeze()

        if self.modulation_type != "BPSK":
            h = torch.stack([h, torch.zeros(h.size())], dim=-1)
            h = torch.view_as_complex(h)
        conv = Cost2100Channel._compute_channel_signal_convolution(h, tx)
        var = 10 ** (-0.1 * snr)
        if self.modulation_type == "BPSK":
            w = torch.sqrt(torch.tensor(var)) * torch.randn(self.num_antennas, tx.size(0))
        else:
            w_real = torch.sqrt(torch.tensor(var)) / 2 * torch.randn(self.num_antennas, tx.size(0))
            w_imag = torch.sqrt(torch.tensor(var)) / 2 * torch.randn(self.num_antennas, tx.size(0)) * 1j
            w = w_real + w_imag
        y = conv + w
        if not self.linear_channel:
            y = torch.tanh(0.5 * y)
        return y
