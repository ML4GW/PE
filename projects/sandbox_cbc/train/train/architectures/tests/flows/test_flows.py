from unittest.mock import Mock

import pytest
import torch

from mlpe.data.transforms import Preprocessor


@pytest.fixture(params=[2, 10])
def param_dim(request):
    return request.param


@pytest.fixture(params=[512, 1024])
def strain_dim(request):
    return request.param


@pytest.fixture(params=[1, 2, 3])
def n_ifos(request):
    return request.param


@pytest.fixture(params=[1, 2, 4])
def num_transforms(request):
    return request.param


def test_coupling_flow(param_dim, strain_dim, n_ifos, num_transforms):
    data = torch.randn((100, param_dim))
    strain = torch.randn((100, n_ifos, strain_dim))
    from mlpe.architectures.embeddings import ResNet
    from mlpe.architectures.flows import CouplingFlow

    embedding = ResNet((n_ifos, None), layers=[1, 1], context_dim=10)
    preprocessor = Preprocessor(n_ifos, strain_dim, fduration=1.0)

    opt = Mock()
    sched = Mock()
    inference_params = Mock()
    priors = Mock()

    coupling_flow = CouplingFlow(
        (param_dim, n_ifos, strain_dim),
        embedding,
        preprocessor,
        opt,
        sched,
        inference_params,
        priors,
        num_transforms=num_transforms,
    )
    log_likelihoods = coupling_flow.log_prob(data, context=strain)

    assert log_likelihoods.shape == (len(data),)


def test_masked_autoregressive_flow(
    param_dim, strain_dim, n_ifos, num_transforms
):

    data = torch.randn((100, param_dim))
    strain = torch.randn((100, n_ifos, strain_dim))
    from mlpe.architectures.embeddings import ResNet
    from mlpe.architectures.flows import MaskedAutoRegressiveFlow

    embedding = ResNet((n_ifos, None), layers=[1, 1], context_dim=10)
    preprocessor = Preprocessor(n_ifos, strain_dim, fduration=1.0)
    opt = Mock()
    sched = Mock()
    inference_params = Mock()
    priors = Mock()
    maf = MaskedAutoRegressiveFlow(
        (param_dim, n_ifos, strain_dim),
        embedding,
        preprocessor,
        opt,
        sched,
        inference_params,
        priors,
        num_transforms=num_transforms,
    )
    log_likelihoods = maf.log_prob(data, context=strain)
    assert log_likelihoods.shape == (len(data),)
