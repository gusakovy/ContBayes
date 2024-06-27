from math import prod
import torch
import torch.nn as nn
from torch import Tensor

REDUCTION_TYPES = [
    "label"
]

NORM_TYPES = [
    "mean",
    "square"
]


def autograd_jacobian(model: nn.Module,
                      inputs: Tensor,
                      output_dim: int,
                      truncate: bool,
                      reduction: str = None,
                      normalization: str = None,
                      **kwargs):
    """
    Returns the Jacobian of the model with respect to its weights. If multiple inputs are given, the jacobian
    matrices are concatenated along the first dimension.

    :param model: Pytorch Neural Network
    :param inputs: Inputs at which to calculate the Jacobian
    :param output_dim: Dimension of the Neural Network output
    :param truncate: Flag specifying if the last dimension of the categorical output should be truncated
    :param reduction: Specifies the reduction to apply to the output: 'label' for per-label sum or None.
    :param normalization: Specifies the normalization to apply after reduction: 'mean' or None.
    :return Jacobian: Matrix of the model with respect to its weights
    """

    model_params = [param for param in model.parameters()]
    n_params = sum(([prod(param.data.size()) for param in model_params]))

    n_obs = inputs.size(0)
    out_dim = (output_dim - 1) if truncate else output_dim

    match reduction:
        case "label":
            if "labels" not in kwargs.keys():
                raise ValueError('No labels provided')
            labels = kwargs["labels"]
            distinct_labels = sorted(set(labels.tolist()))
            num_distinct_labels = len(distinct_labels)

            Jacobian = torch.zeros(num_distinct_labels * out_dim, n_params)

            for label_idx, label in enumerate(distinct_labels):
                if normalization == "mean":
                    outputs = model(inputs[torch.where(labels == label)]).mean(dim=0)
                else:
                    outputs = model(inputs[torch.where(labels == label)]).sum(dim=0)
                for class_idx in range(out_dim):
                    entry_idx = 0
                    model.zero_grad()
                    outputs[class_idx].backward(retain_graph=True)
                    for param in model_params:
                        grad = param.grad.flatten()
                        Jacobian[label_idx * out_dim + class_idx, entry_idx: entry_idx + grad.size(0)] = grad
                        entry_idx += grad.size(0)

        case _:
            outputs = model(inputs)
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
    scale = torch.diag(L_c)
    tril = torch.reciprocal(scale) * L_c
    return scale, tril


def scale_and_tril_to_cov(scale: Tensor, tril: Tensor) -> Tensor:
    cov = (tril * scale.pow(2)) @ tril.T
    return cov


def cholesky_to_scale_and_tril(cholesky: Tensor) -> tuple[Tensor, Tensor]:
    scale = torch.diag(cholesky)
    tril = torch.reciprocal(scale) * cholesky
    return scale, tril


def scale_and_tril_to_cholesky(scale: Tensor, tril: Tensor) -> Tensor:
    cholesky = scale * tril
    return cholesky


def categorical_cov(p: Tensor) -> Tensor:
    cov = torch.diag(p.flatten()) - p.view(-1, 1) @ p.view(1, -1)
    return cov


def multiple_categorical_cov(prob_vector: Tensor, num_classes, reduction=None, normalization=None, **kwargs):
    prob_vector = prob_vector.flatten()
    num_obs = prob_vector.size(0) // (num_classes - 1)
    full_cov = categorical_cov(prob_vector)
    cov_column = torch.stack([full_cov[obs_idx*(num_classes - 1):(obs_idx+1)*(num_classes - 1),
                                       obs_idx*(num_classes - 1):(obs_idx+1)*(num_classes - 1)]
                              for obs_idx in range(num_obs)])

    match reduction:
        case "label":
            cov = reduce_tensor(cov_column, normalization="square" if normalization == "mean" else None, **kwargs)
            cov = torch.block_diag(*list(cov))
            return cov
        case None:
            cov = torch.block_diag(*list(cov_column))
            return cov
        case _:
            raise ValueError(f"Unrecognized reduction method. Recognised reductions: {REDUCTION_TYPES}")


def reduce_tensor(inputs: Tensor, normalization = None, **kwargs):

        if "labels" not in kwargs.keys():
            raise ValueError('No labels provided')


        labels = kwargs["labels"]
        assert labels.size(0) == inputs.size(0)
        distinct_labels = sorted(set(labels.tolist()))

        match normalization:
            case None:
                counts = {label:1 for label in distinct_labels}
            case "mean":
                counts = {label:torch.sum(labels == label, dim=0) for label in distinct_labels}
            case "square":
                counts = {label:torch.sum(labels == label, dim=0).pow(2) for label in distinct_labels}
            case _:
                raise ValueError(f"Unrecognized normalization method. Recognised reductions: {NORM_TYPES}")


        reduced_tensor = torch.stack([torch.sum(inputs[torch.where(labels == label)], dim=0) / counts[label]
                                      for label in distinct_labels])

        return reduced_tensor
