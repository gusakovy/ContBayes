import torch
from config import MODULATION_TYPE, H_COEF, LINEAR_CHANNEL
Tensor = torch.tensor


class SEDChannel:
    @staticmethod
    def calculate_channel(n_ant: int, n_user: int, frame_ind: int, fading: bool) -> Tensor:
        H_row = Tensor([i for i in range(n_ant)])
        H_row = torch.tile(H_row, [n_user, 1]).T
        H_column = Tensor([i for i in range(n_user)])
        H_column = torch.tile(H_column, [n_ant, 1])
        H = torch.exp(-torch.abs(H_row - H_column))
        if fading:
            H = SEDChannel._add_fading(H, n_ant, frame_ind)
        return H

    @staticmethod
    def _add_fading(h: Tensor, n_ant: int, frame_ind: int) -> Tensor:
        deg_array = Tensor([51, 39, 33, 21])
        fade_mat = H_COEF + (1 - H_COEF) * torch.cos(2 * torch.pi * frame_ind / deg_array)
        fade_mat = torch.tile(fade_mat.reshape(1, -1), [n_ant, 1])
        return h * fade_mat

    @staticmethod
    def transmit(s: Tensor, h: Tensor, snr: float, n_ant: int) -> Tensor:
        """
        The MIMO SED Channel
        :param s: to transmit symbol words
        :param snr: signal-to-noise ratio
        :param h: channel function
        :param n_ant: number of receive antennas
        :return: received word
        """
        if MODULATION_TYPE != "BPSK":
            h = torch.stack([h, torch.zeros(h.size())], dim=-1)
            h = torch.view_as_complex(h)
        conv = SEDChannel._compute_channel_signal_convolution(h, s)
        var = 10 ** (-0.1 * snr)
        if MODULATION_TYPE == "BPSK":
            w = torch.sqrt(Tensor(var)) * torch.randn(n_ant, s.size(1))
        else:
            w_real = torch.sqrt(Tensor(var)) / 2 * torch.randn(n_ant, s.size(1))
            w_imag = torch.sqrt(Tensor(var)) / 2 * torch.randn(n_ant, s.size(1)) * 1j
            w = w_real + w_imag
        y = conv + w
        if not LINEAR_CHANNEL:
            y = torch.tanh(0.5 * y)
        return y

    @staticmethod
    def _compute_channel_signal_convolution(h: Tensor, s: Tensor) -> Tensor:
        conv = h @ s
        return conv
