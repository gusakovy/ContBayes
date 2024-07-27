from dataclasses import dataclass
from typing import Type
from functools import partial
import numpy as np
import torch.nn as nn
from torch import Tensor
import pyro
import pyro.infer.autoguide as ag
import pyro.distributions as dist
import tyxe
from tyxe.priors import Prior
from tyxe.priors import util as prior_util
from tyxe.likelihoods import Likelihood
from contbayes.Utilities.math_utils import scale_and_tril_to_cov, cov_to_scale_and_tril, autograd_jacobian


MODEL_TYPES = [
    "classifier",
    "regressor"
]


@dataclass
class BNNParams:
    """Data class for BayesNN model parameters"""

    loc: Tensor
    scale: Tensor
    tril: Tensor


class FullCovBNN(tyxe.VariationalBNN):
    """
    Bayesian Neural Network class based on TyXe VariationalBNN with the additional functionality of interacting
    with the weights of the model directly and computing the Jacobian of the model with respect to the weights.
    The Bayesian model has a full covariance matrix, hence the parameters of the model are comprised of a mean tensor
    `loc` and tensors `scale` and `scale_tril` which are the square root of the matrix D and the matrix L in the LDLT
    decomposition of the covariance matrix respectively.

    :param model_type: "classifier" or "regressor"
    :param output_dim: dimensionality of the output
    :param net_builder: `torch.nn.Module` which the Bayesian Neural Network is built on
    :param prior: `tyxe.priors.Prior` import Prior to serve as a prior distribution
    :param likelihood: `tyxe.likelihoods.Likelihood` to serve as an observation model
    :param init_cov_scale: scale for the initialization of the covariance matrix
    """

    def __init__(self,
                 model_type: str,
                 output_dim: int,
                 net_builder: Type[nn.Module],
                 prior: Prior,
                 likelihood: Likelihood,
                 init_cov_scale: float = 1e-4):

        if model_type not in MODEL_TYPES:
            raise ValueError(f"Model type must be one of {MODEL_TYPES}.")
        super().__init__(net=net_builder(),
                         prior=prior,
                         likelihood=likelihood,
                         net_guide_builder=partial(ag.AutoMultivariateNormal, init_scale=init_cov_scale))
        self.type = model_type
        self.output_dim = output_dim
        self.torch_net = net_builder()

    def update_prior(self, new_prior):
        self.prior = new_prior
        self.prior.update_(self.net)

    def set_prior_to_current_diagonal(self):
        """
        Updates the prior of the Bayesian neural network to be diagonal of the current distribution of the weights,
        that is, if the distributions of the weights is N(mu,Sigma), the prior will become N(mu,diag(Sigma)).
        Intended for the implementation of variational continual learning (VCL).
        """

        new_prior_dict = {}
        param_index = 0
        for module_name, module in self.net.named_modules():
            for site_name, site in list(prior_util.named_pyro_samples(module, recurse=False)):
                full_name = module_name + "." + site_name
                shape = site.event_shape
                dim = len(shape)
                num_params = np.prod(shape)
                loc_tensor = self.net_guide.loc[param_index: param_index + num_params].detach().clone().reshape(
                    site.event_shape)
                scale_tensor = self.net_guide.scale[param_index: param_index + num_params].detach().clone().reshape(
                    site.event_shape)
                new_prior_dict[full_name] = dist.Normal(loc=loc_tensor, scale=scale_tensor).expand(shape).to_event(dim)
                param_index += num_params
        new_prior = tyxe.priors.DictPrior(new_prior_dict)
        self.update_prior(new_prior)

    @staticmethod
    def get_loc() -> Tensor:
        """
        Getter for the mean vector of the weights of the model.

        :return: mean vector of the weights of the model
        """

        loc = pyro.get_param_store().get_param('net_guide.loc').detach().clone()
        return loc

    def set_loc(self, new_loc: Tensor):
        """
        Setter for the mean vector of the weights of the model.

        :param new_loc: new mean vector for the weights of the model
        """

        if new_loc.squeeze().size() != self.net_guide.loc.data.size():
            raise ValueError(f"new_loc size {list(new_loc.size())} is not compatible with model's loc size "
                             f"{list(self.net_guide.loc.size())}.")
        self.net_guide.loc.data = new_loc.squeeze()

    @staticmethod
    def get_scale() -> Tensor:
        """
        Getter for the scale matrix of the weights of the model.

        :return: scale matrix of the weights of the model
        """

        scale = pyro.get_param_store().get_param('net_guide.scale').detach().clone()
        return scale

    @staticmethod
    def set_scale(new_scale: Tensor):
        """
        Setter for the scale matrix of the weights of the model.

        :param new_scale: new scale matrix for the weights of the model
        """

        if new_scale.size() != pyro.get_param_store().get_param('net_guide.scale').size():
            raise ValueError(f"new_scale shape {list(new_scale.size())} is not compatible with model's scale shape "
                             f"{list(pyro.get_param_store().get_param('net_guide.scale').size())}.")
        pyro.get_param_store().__setitem__('net_guide.scale', new_scale)

    @staticmethod
    def get_tril() -> Tensor:
        """
        Getter for the scale_tril matrix of the weights of the model.

        :return: scale_tril matrix of the weights of the model
        """

        scale_tril = pyro.get_param_store().get_param('net_guide.scale_tril').detach().clone()
        return scale_tril

    @staticmethod
    def set_tril(new_tril: Tensor):
        """
        Setter for the scale_tril matrix of the weights of the model.

        :param new_tril: new scale_tril matrix for the weights of the model
        """

        if new_tril.size() != pyro.get_param_store().get_param('net_guide.scale_tril').size():
            raise ValueError(f"new_tril shape {list(new_tril.size())} is not compatible with model's scale shape "
                             f"{list(pyro.get_param_store().get_param('net_guide.scale_tril').size())}.")
        pyro.get_param_store().__setitem__('net_guide.scale_tril', new_tril)

    @staticmethod
    def get_cov() -> Tensor:
        """
        Getter for the covariance matrix of the weights of the model.

        :return: covariance matrix of the weights of the model
        """

        return scale_and_tril_to_cov(FullCovBNN.get_scale(), FullCovBNN.get_tril())

    @staticmethod
    def set_cov(cov: Tensor):
        """
        Setter for the covariance matrix of the weights of the model.

        :param cov: new covariance matrix for the weights of the model
        """

        scale, scale_tril = cov_to_scale_and_tril(cov)
        FullCovBNN.set_scale(scale)
        FullCovBNN.set_tril(scale_tril)

    def get_params(self) -> BNNParams:
        """
        Getter for the parameters of the model.

        :return: parameters of the model
        """

        return BNNParams(self.get_loc(), self.get_scale(), self.get_tril())

    def set_params(self, new_params: BNNParams):
        """
        Setter for the parameters of the model.

        :param new_params: new parameters for the model
        """

        new_loc, new_scale, new_tril = new_params.loc, new_params.scale, new_params.tril
        if new_loc is not None:
            self.set_loc(new_loc)
        if new_scale is not None:
            self.set_scale(new_scale)
        if new_tril is not None:
            self.set_tril(new_tril)

    def jacobian(self, inputs: Tensor, truncate: bool = None, reduction: str = None, normalization: str = None,
                 labels: Tensor = None) -> Tensor:
        """
        Compute the Jacobian of the model with respect to its weights. If multiple inputs are given, the jacobian
        matrices are concatenated along the first dimension.

        :param inputs: tensor of inputs
        :param truncate: whether to truncate the last dimension the Jacobian (intended for classifiers)
        :param reduction: reduction method for the Jacobian. For 'label' reduction, an additional
            keyword argument 'labels' of the labels corresponding to the inputs is necessary to perform the reduction
        :param normalization: normalization to be applied after reduction: 'mean' or None
        :param labels: labels necessary to perform per-label reduction
        :return: Jacobian of the model with respect to its weights
        """

        if reduction == 'label' and labels is None:
            raise ValueError(f"No labels provided for {reduction} reduction.")

        if truncate is None:
            truncate = True if self.type == "classifier" else False

        param_vector = self.get_loc()
        nn.utils.vector_to_parameters(param_vector, self.torch_net.parameters())

        return autograd_jacobian(model=self.torch_net,
                                 inputs=inputs,
                                 output_dim=self.output_dim,
                                 truncate=truncate,
                                 reduction=reduction,
                                 normalization=normalization,
                                 labels=labels if reduction == 'label' else None)
