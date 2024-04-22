import torch
from torch import Tensor

MODULATION_TYPES = ["BPSK", "QPSK"]


class SEDChannel:
    """
    Synthetic memory-less MIMO channel.

    :param modulation_type: modulation type ("BPSK" or "QPSK")
    :param fading_coefficient: channel fading coefficient
    :param linear_channel: whether the channel is linear or non-linear
    """

    def __init__(self, modulation_type: str, fading_coefficient: float = None, linear_channel: bool = False):
        if modulation_type not in MODULATION_TYPES:
            raise ValueError(f"Modulation type must be one of {MODULATION_TYPES}.")
        self.modulation_type = modulation_type
        self.fading_coefficient = fading_coefficient
        self.linear_channel = linear_channel

    def calculate_channel(self, n_ant: int, n_user: int, frame_ind: int, fading: bool = False) -> Tensor:
        """
        Calculate the channel information matrix.

        :param n_ant: number of receive antennas
        :param n_user: number of users
        :param frame_ind: time frame index
        :param fading: whether to apply channel fading
        :return: channel information matrix
        """

        H_row = torch.tensor([i for i in range(n_ant)])
        H_row = torch.tile(H_row, [n_user, 1]).T
        H_column = torch.tensor([i for i in range(n_user)])
        H_column = torch.tile(H_column, [n_ant, 1])
        H = torch.exp(-torch.abs(H_row - H_column))
        if fading:
            H = self._add_fading(H, n_ant, frame_ind)
        return H

    def _add_fading(self, h: Tensor, n_ant: int, frame_ind: int) -> Tensor:
        """Apply channel fading to channel information matrix."""

        deg_array = torch.tensor([51, 39, 33, 21])
        fade_mat = (self.fading_coefficient + (1 - self.fading_coefficient) *
                    torch.cos(2 * torch.pi * frame_ind / deg_array))
        fade_mat = torch.tile(fade_mat.reshape(1, -1), [n_ant, 1])
        return h * fade_mat

    def transmit(self, s: Tensor, h: Tensor, snr: float, n_ant: int) -> Tensor:
        """
        Simulate transmission of symbols.

        :param s: constellation points of symbols to be transmitted
        :param snr: signal-to-noise ratio
        :param h: channel information matrix
        :param n_ant: number of receive antennas
        :return: transmitted signal
        """

        if self.modulation_type != "BPSK":
            h = torch.stack([h, torch.zeros(h.size())], dim=-1)
            h = torch.view_as_complex(h)
        conv = SEDChannel._compute_channel_signal_convolution(h, s)
        var = 10 ** (-0.1 * snr)
        if self.modulation_type == "BPSK":
            w = torch.sqrt(torch.tensor(var)) * torch.randn(n_ant, s.size(1))
        else:
            w_real = torch.sqrt(torch.tensor(var)) / 2 * torch.randn(n_ant, s.size(1))
            w_imag = torch.sqrt(torch.tensor(var)) / 2 * torch.randn(n_ant, s.size(1)) * 1j
            w = w_real + w_imag
        y = conv + w
        if not self.linear_channel:
            y = torch.tanh(0.5 * y)
        return y

    @staticmethod
    def _compute_channel_signal_convolution(h: Tensor, s: Tensor) -> Tensor:
        """Compute the convolution of a channel matrix and a tensor of constellation points."""

        conv = h @ s
        return conv
