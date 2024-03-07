from typing import Callable

import torch
from pyro.distributions import ConditionalTransformedDistribution, transforms
from pyro.distributions.conditional import ConditionalComposeTransformModule
from pyro.nn import PyroModule


class FlowArchitecture(PyroModule):
    def __init__(
        self, num_params: int, context_dim: int, embedding_net: torch.nn.Module
    ):

        super().__init__()
        self.num_params = num_params
        self.context_dim = context_dim
        self.embedding_net = embedding_net

    def transform_block(
        self, *args, **kwargs
    ) -> Callable[[int], transforms.Transform]:
        raise NotImplementedError

    def distribution(self) -> torch.distributions.Distribution:
        raise NotImplementedError

    def build_transforms(self) -> ConditionalComposeTransformModule:
        """Sets the ``transforms`` attribute"""
        raise NotImplementedError

    def flow(self) -> ConditionalTransformedDistribution:
        return ConditionalTransformedDistribution(
            self.distribution(), self.transforms
        )

    def log_prob(self, x, context):
        """Wrapper around :meth:`log_prob` from
        `TransformedDistribution` object.
        """
        if not hasattr(self, "transforms"):
            raise RuntimeError("Flow is not built")
        embedded_context = self.embedding_net(context)
        return self.flow().condition(embedded_context).log_prob(x)

    def sample(self, n, context):
        """Wrapper around :meth:`sample` from
        `TransformedDistribution` object.
        """
        if not hasattr(self, "transforms"):
            raise RuntimeError("Flow is not built")

        embedded_context = self.embedding_net(context)
        n = [n] if isinstance(n, int) else n
        return self.flow.condition(embedded_context).sample(n)
