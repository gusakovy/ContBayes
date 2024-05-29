import math
import torch
from torch import Tensor

MODULATION_TYPES = ["BPSK", "QPSK"]


class SEDChannel:
    """
    Synthetic memory-less MIMO channel.

    :param modulation_type: modulation type ("BPSK" or "QPSK")
    :param fading_coefficient: channel fading coefficient between 0 (full fading) and 1 (no fading).
    :param linear_channel: whether the channel is linear or non-linear
    """

    def __init__(self, modulation_type: str, num_users: int, num_antennas: int,
                 fading_coefficient: float, linear_channel: bool = False):
        if modulation_type not in MODULATION_TYPES:
            raise ValueError(f"Modulation type must be one of {MODULATION_TYPES}.")
        if fading_coefficient > 1 or fading_coefficient < 0:
            raise ValueError("Fading coefficient must be between 0 and 1.")
        self.modulation_type = modulation_type
        self.num_users = num_users
        self.num_antennas = num_antennas
        self.constellation_points = torch.tensor([-1, 1], dtype=torch.complex64) if self.modulation_type == "BPSK"\
            else torch.tensor([[math.cos(math.pi * (k + 1/2) / 2) +
                                math.sin(math.pi * (k + 1/2) / 2) * 1j] for k in range(4)], dtype=torch.complex64)
        self.fading_coefficient = fading_coefficient
        self.linear_channel = linear_channel

    def _calculate_channel(self, frame_ind: int) -> Tensor:
        """Calculate the channel information matrix."""

        H_row = torch.tensor([i for i in range(self.num_antennas)])
        H_row = torch.tile(H_row, [self.num_users, 1]).T
        H_column = torch.tensor([i for i in range(self.num_users)])
        H_column = torch.tile(H_column, [self.num_antennas, 1])
        H = torch.exp(-torch.abs(H_row - H_column))
        if self.fading_coefficient < 1:
            H = self._add_fading(H, frame_ind)
        return H

    def _add_fading(self, h: Tensor, frame_ind: int) -> Tensor:
        """Apply channel fading to channel information matrix."""

        deg_array = torch.tensor([51, 39, 33, 21])
        fade_mat = (self.fading_coefficient + (1 - self.fading_coefficient) *
                    torch.cos(2 * torch.pi * frame_ind / deg_array))
        fade_mat = torch.tile(fade_mat.reshape(1, -1), [self.num_antennas, 1])
        return h * fade_mat

    @staticmethod
    def _compute_channel_signal_convolution(h: Tensor, tx: Tensor) -> Tensor:
        """Compute the convolution of a channel matrix and a tensor of constellation points."""

        conv = h @ tx.T
        return conv

    def transmit(self, s: Tensor, snr: float, frame_ind: int = 0) -> Tensor:
        """
        Simulate transmission of symbols.

        :param s: symbols be transmitted
        :param snr: signal-to-noise ratio
        :param frame_ind: time frame index
        :return: transmitted signal
        """

        h = self._calculate_channel(frame_ind)
        tx = self.constellation_points[s].squeeze()

        if self.modulation_type != "BPSK":
            h = torch.stack([h, torch.zeros(h.size())], dim=-1)
            h = torch.view_as_complex(h)
        conv = SEDChannel._compute_channel_signal_convolution(h, tx)
        var = 10 ** (-0.1 * snr)
        if self.modulation_type == "BPSK":
            w = torch.sqrt(torch.tensor(var)) * torch.randn(self.num_antennas, tx.size(1))
        else:
            w_real = torch.sqrt(torch.tensor(var)) / 2 * torch.randn(self.num_antennas, tx.size(0))
            w_imag = torch.sqrt(torch.tensor(var)) / 2 * torch.randn(self.num_antennas, tx.size(0)) * 1j
            w = w_real + w_imag
        y = conv + w
        if not self.linear_channel:
            y = torch.tanh(0.5 * y)
        return y
