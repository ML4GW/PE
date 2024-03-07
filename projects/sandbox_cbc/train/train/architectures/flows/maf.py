import torch
import torch.distributions as dist
from pyro.distributions.conditional import ConditionalComposeTransformModule
from pyro.distributions.transforms import ConditionalAffineAutoregressive
from pyro.nn import ConditionalAutoRegressiveNN
from train.architectures.flows import FlowArchitecture


class MaskedAutoRegressiveFlow(FlowArchitecture):
    def __init__(
        self,
        *args,
        hidden_features: int = 50,
        num_transforms: int = 5,
        num_blocks: int = 2,
        activation: torch.nn.modules.activation = torch.nn.Tanh(),
        **kwargs,
    ):

        super().__init__(*args, **kwargs)
        self.hidden_features = hidden_features
        self.num_blocks = num_blocks
        self.num_transforms = num_transforms
        self.activation = activation

        # register these as buffers so the
        # distributions are moved to the correct device
        self.register_buffer("mean", torch.zeros(self.num_params))
        self.register_buffer("std", torch.ones(self.num_params))

        # build the sequence of transforms
        self.transforms = self.build_transforms()

    def transform_block(self):
        """Returns single autoregressive transform"""
        arn = ConditionalAutoRegressiveNN(
            self.num_params,
            self.context_dim,
            self.num_blocks * [self.hidden_features],
            nonlinearity=self.activation,
        )
        return ConditionalAffineAutoregressive(arn)

    def distribution(self):
        """Returns the base distribution for the flow"""
        return dist.Normal(
            torch.zeros(self.num_params),
            torch.ones(self.num_params),
        )

    def build_transforms(self):
        """Build the transform"""
        transforms = []
        for _ in range(self.num_transforms):
            transform = self.transform_block()
            transforms.extend([transform])
        return ConditionalComposeTransformModule(transforms)
