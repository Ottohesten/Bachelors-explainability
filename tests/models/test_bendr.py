import pytest
import torch

from eegatscale.models.bendr import BendrContextualizer, BendrEncoder


@pytest.mark.parametrize("projection_head", [False, True])
@pytest.mark.parametrize("input_length", [50, 100, 255, 500])
def test_bendr_encoder(projection_head, input_length):
    batch_size = 5
    in_features = 32

    encoder = BendrEncoder(in_features=in_features, projection_head=projection_head)
    output_features = encoder.out_features
    output_length = encoder.downsampling_factor(input_length)
    x = torch.randn(batch_size, in_features, input_length)

    out = encoder(x)
    assert out.shape == (batch_size, output_features, output_length)
    print(out.shape)


@pytest.mark.parametrize("input_length", [50, 100, 255, 500])
def test_bendr_contextualizer(input_length):
    batch_size = 5
    in_features = 32

    contextualizer = BendrContextualizer(in_features=in_features)
    x = torch.randn(batch_size, in_features, input_length)
    out = contextualizer(x)
    assert out.shape == (batch_size, in_features, input_length + 1)
    print(out.shape)
