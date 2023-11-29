from math import prod
import torch
import torch.nn as nn
from torch import Tensor


def autograd_jacobian(model: nn.Module, inputs: Tensor, truncate: bool):
    """
    Returns the Jacobian of the model with respect to its weights. If multiple inputs are given, the jacobian
    matrices are concatenated along the first dimension.

    :param model: Pytorch Neural Network
    :param inputs: Inputs at which to calculate the Jacobian
    :param truncate: Flag specifying if the last dimension of the categorical output should be truncated
    :return Jacobian: Matrix of the model with respect to its weights
    """

    model_params = [param for param in model.parameters()]
    n_params = sum(([prod(param.data.size()) for param in model_params]))

    n_obs = inputs.size(0)
    outputs = model(inputs)
    out_dim = (outputs.size(-1) - 1) if truncate else outputs.size(-1)

    Jacobian = torch.zeros(n_obs * out_dim, n_params)

    for obs_idx in range(n_obs):
        for class_idx in range(out_dim):
            entry_idx = 0
            model.zero_grad()
            outputs[obs_idx][class_idx].backward(retain_graph=True)
            for param in model_params:
                grad = param.grad.flatten()
                Jacobian[obs_idx * out_dim + class_idx, entry_idx: entry_idx + grad.size(0)] = grad
                entry_idx += grad.size(0)

    return Jacobian


def cov_to_scale_and_tril(cov: Tensor) -> tuple[Tensor, Tensor]:
    L_c = torch.linalg.cholesky(cov)
    scale = torch.diagonal(L_c)
    tril = torch.diag(L_c).pow(-1) * L_c
    return scale, tril


def scale_and_tril_to_cov(scale: Tensor, tril: Tensor) -> Tensor:
    cov = tril @ torch.diag(scale).pow(2) @ tril.T
    return cov


def categorical_cov(p: Tensor) -> Tensor:
    cov = torch.diag(p.flatten()) - p.view(-1, 1) @ p.view(1, -1)
    return cov



