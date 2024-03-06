import torch
from typing import Callable
from torch.distributions import Uniform
from ml4gw.distributions import LogUniform
from math import pi
from train.data.utils import ParameterSampler, ParameterTransformer

sine_gaussian = ParameterSampler(
    frequency=Uniform(32, 1024),
    quality=Uniform(2, 100),
    hrss=LogUniform(1e-23, 5e-20),
    phase=Uniform(0, 2 * pi),
    eccentricity=Uniform(0, 1),
)

sine_gaussian_transformer = ParameterTransformer(
    hrss=torch.log
)