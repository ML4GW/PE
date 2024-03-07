import pytest
import torch

from mlpe.architectures.embeddings.dense import DenseEmbedding


@pytest.fixture(params=[1, 2, 3])
def n_ifos(request):
    return request.param


@pytest.fixture(params=[2048, 8192])
def length(request):
    return request.param


@pytest.fixture(
    params=[
        10,
        20,
    ]
)
def kernel_size(request):
    return request.param


@pytest.fixture(params=[128, 256])
def out_features(request):
    return request.param


def test_dense_embedding(n_ifos, length):
    embedding = DenseEmbedding(length, 10)
    x = torch.randn(8, n_ifos, length)
    y = embedding(x)

    assert y.shape == (8, n_ifos, 10)
