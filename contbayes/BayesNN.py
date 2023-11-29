import torch.nn as nn
from torch import Tensor
import pyro
from pyro.infer.autoguide import AutoGuide
import tyxe
from tyxe.priors import Prior
from tyxe.likelihoods import Likelihood
from contbayes.utils import scale_and_tril_to_cov, cov_to_scale_and_tril, autograd_jacobian


MODEL_TYPES = [
    "classifier",
    "regressor"
]


class FullCovBNN(tyxe.VariationalBNN):

    def __init__(self,
                 model_type: str,
                 output_dim: int,
                 net_builder: callable,
                 prior: Prior,
                 likelihood: Likelihood,
                 guide_builder: AutoGuide):

        if model_type not in MODEL_TYPES:
            raise ValueError(f"Unknown model type: {model_type}. Available types include: {MODEL_TYPES}")
        self.type = model_type
        self.output_dim = output_dim
        super().__init__(net=net_builder(),
                         prior=prior,
                         likelihood=likelihood,
                         net_guide_builder=guide_builder)
        self.torch_net = net_builder()

    @staticmethod
    def get_loc() -> Tensor:
        loc = pyro.get_param_store().get_param('net_guide.loc').detach().clone()
        return loc

    def set_loc(self, new_value: Tensor) -> None:
        assert new_value.squeeze().size() == self.net_guide.loc.data.size()
        self.net_guide.loc.data = new_value.squeeze()

    @staticmethod
    def get_scale() -> Tensor:
        scale = pyro.get_param_store().get_param('net_guide.scale').detach().clone()
        return scale

    @staticmethod
    def set_scale(new_value: Tensor) -> None:
        assert new_value.size() == pyro.get_param_store().get_param('net_guide.scale').size()
        pyro.get_param_store().__setitem__('net_guide.scale', new_value)

    @staticmethod
    def get_tril() -> Tensor:
        scale_tril = pyro.get_param_store().get_param('net_guide.scale_tril').detach().clone()
        return scale_tril

    @staticmethod
    def set_tril(new_value: Tensor) -> None:
        assert new_value.size() == pyro.get_param_store().get_param('net_guide.scale_tril').size()
        pyro.get_param_store().__setitem__('net_guide.scale_tril', new_value)

    @staticmethod
    def get_cov() -> Tensor:
        return scale_and_tril_to_cov(FullCovBNN.get_scale(), FullCovBNN.get_tril())

    @staticmethod
    def set_cov(cov: Tensor) -> None:
        scale, scale_tril = cov_to_scale_and_tril(cov)
        FullCovBNN.set_scale(scale)
        FullCovBNN.set_tril(scale_tril)

    def jacobian(self, inputs: Tensor, truncate: bool = None) -> Tensor:
        """
        Returns the Jacobian of the model with respect to its weights. If multiple inputs are given, the jacobian
        matrices are concatenated along the first dimension.
        """
        if truncate is None:
            truncate = True if self.type == "classifier" else False

        param_vector = self.get_loc()
        nn.utils.vector_to_parameters(param_vector, self.torch_net.parameters())

        return autograd_jacobian(self.torch_net, inputs, truncate)
