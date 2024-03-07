import pyro.distributions as dist
import torch
from pyro.distributions.conditional import ConditionalComposeTransformModule
from pyro.distributions.transforms import ConditionalAffineCoupling
from pyro.nn import ConditionalDenseNN
from train.architectures.flows import FlowArchitecture


class CouplingFlow(FlowArchitecture):
    def __init__(
        self,
        *args,
        hidden_features: int = 512,
        num_transforms: int = 5,
        num_blocks: int = 2,
        activation: torch.nn.modules.activation = torch.nn.Tanh(),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.split_dim = self.num_params // 2
        self.hidden_features = hidden_features
        self.num_blocks = num_blocks
        self.num_transforms = num_transforms
        self.activation = activation

        # register these as buffers so they are moved to the correct device
        self.register_buffer("mean", torch.zeros(self.num_params))
        self.register_buffer("std", torch.ones(self.num_params))

        # build the sequence of transforms
        self.transforms = self.build_transforms()

    def transform_block(self):
        """Returns single affine coupling transform"""
        arn = ConditionalDenseNN(
            self.split_dim,
            self.context_dim,
            [self.hidden_features],
            param_dims=[
                self.num_params - self.split_dim,
                self.num_params - self.split_dim,
            ],
            nonlinearity=self.activation,
        )
        transform = ConditionalAffineCoupling(self.split_dim, arn)
        return transform

    def distribution(self):
        """Returns the base distribution for the flow"""
        return dist.Normal(
            self.mean,
            self.std,
        )

    def build_transforms(self):
        transforms = []
        for _ in range(self.num_transforms):
            transforms.extend([self.transform_block()])
        return ConditionalComposeTransformModule(transforms)
