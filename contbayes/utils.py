from math import prod
import torch
import torch.nn as nn
from torch import Tensor

REDUCTION_TYPES = [
    "label"
]


def autograd_jacobian(model: nn.Module,
                      inputs: Tensor,
                      output_dim: int,
                      truncate: bool,
                      reduction: str = None,
                      **kwargs):
    """
    Returns the Jacobian of the model with respect to its weights. If multiple inputs are given, the jacobian
    matrices are concatenated along the first dimension.

    :param model: Pytorch Neural Network
    :param inputs: Inputs at which to calculate the Jacobian
    :param output_dim: Dimension of the Neural Network output
    :param truncate: Flag specifying if the last dimension of the categorical output should be truncated
    :param reduction: Specifies the reduction to apply to the output: 'label' for per-label sum or None.
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

            Jacobian = torch.zeros(out_dim * (out_dim + 1), n_params)

            for label in range(out_dim + 1):
                outputs = model(inputs[torch.where(labels == label)]).sum(dim=0)
                for class_idx in range(out_dim):
                    entry_idx = 0
                    model.zero_grad()
                    outputs[class_idx].backward(retain_graph=True)
                    for param in model_params:
                        grad = param.grad.flatten()
                        Jacobian[label * out_dim + class_idx, entry_idx: entry_idx + grad.size(0)] = grad
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
    scale = torch.diagonal(L_c)
    tril = torch.diag(L_c).pow(-1) * L_c
    return scale, tril


def scale_and_tril_to_cov(scale: Tensor, tril: Tensor) -> Tensor:
    cov = tril @ torch.diag(scale).pow(2) @ tril.T
    return cov


def categorical_cov(p: Tensor) -> Tensor:
    cov = torch.diag(p.flatten()) - p.view(-1, 1) @ p.view(1, -1)
    return cov


def multiple_categorical_cov(prob_vector: Tensor, num_classes, reduction=None, **kwargs):
    prob_vector = prob_vector.flatten()
    num_obs = prob_vector.size(0) // (num_classes - 1)
    match reduction:
        case 'label':
            if "labels" not in kwargs.keys():
                raise ValueError('No labels provided')
            labels = kwargs["labels"]
            assert num_obs == labels.size(0)
            cov = torch.zeros(num_classes * (num_classes - 1), num_classes * (num_classes - 1))
            for obs_idx in range(num_obs):
                cov[labels[obs_idx] * (num_classes - 1): (labels[obs_idx] + 1) * (num_classes - 1),
                    labels[obs_idx] * (num_classes - 1): (labels[obs_idx] + 1) * (num_classes - 1)] += \
                    categorical_cov(prob_vector[obs_idx * (num_classes - 1): (obs_idx + 1) * (num_classes - 1)])
            return cov
        case None:
            cov = torch.zeros(num_obs * (num_classes - 1), num_obs * (num_classes - 1))
            for obs_idx in range(num_obs):
                cov[obs_idx * (num_classes - 1): (obs_idx + 1) * (num_classes - 1),
                    obs_idx * (num_classes - 1): (obs_idx + 1) * (num_classes - 1)] = \
                    categorical_cov(prob_vector[obs_idx * (num_classes - 1): (obs_idx + 1) * (num_classes - 1)])
            return cov
        case _:
            raise ValueError(f"Unrecognized reduction method. Recognised reductions: {REDUCTION_TYPES}")


def reduce_tensor(inputs: Tensor, reduction=None, **kwargs):
    match reduction:
        case "label":
            if "num_classes" not in kwargs.keys():
                raise ValueError('Specify number of classes')
            if "labels" not in kwargs.keys():
                raise ValueError('No labels provided')
            num_classes = kwargs["num_classes"]
            num_obs = inputs.size(0) // (num_classes - 1)
            labels = kwargs["labels"]
            reduced_tensor = torch.zeros(num_classes, num_classes - 1, inputs.size(1))
            inputs = inputs.view(num_obs, num_classes - 1, -1)
            for class_idx in range(num_classes):
                reduced_tensor[class_idx, ..., ...] = torch.sum(inputs[torch.where(labels == class_idx)], dim=0)
            return reduced_tensor.view(num_classes * (num_classes - 1), -1)
        case None:
            return
        case _:
            raise ValueError(f"Unrecognized reduction method. Recognised reductions: {REDUCTION_TYPES}")
