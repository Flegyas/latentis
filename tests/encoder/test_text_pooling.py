import numpy as np
import pytest
import torch
from latentis.nn.encoders import TextHFEncoder
from latentis.data.encoding.text_pooling import (
    token_pool,
    mean_pool,
    sum_pool,
    cls_pool,
)


@pytest.fixture
def sample_text_data():
    # Create a text encoder for a sample HuggingFace model
    encoder = TextHFEncoder(hf_name="bert-base-uncased")

    # Sample input text data
    samples = [{"text": "Hello world!"}]
    feature = "text"

    # Pre-encode the text data
    pre_encoded = encoder.pre_encode(samples, feature)

    # Generate raw encodings
    raw_encoding = encoder.encode(pre_encoded)

    return encoder, raw_encoding


@pytest.mark.parametrize("pooling_fn", [token_pool, mean_pool, sum_pool, cls_pool])
@pytest.mark.parametrize("layers", [[0], [1], [0, 1], [12]])
def test_text_encoder_with_poolers(sample_text_data, pooling_fn, layers):
    encoder, raw_encoding = sample_text_data

    # Ensure the output of the encode method is in the correct format
    assert "x" in raw_encoding and "mask" in raw_encoding
    encodings = raw_encoding["x"]  # This is a tuple of tensors from different layers
    mask = raw_encoding["mask"]

    # Apply the pooling function
    pooled_encodings = pooling_fn(encodings, mask=mask, layers=layers)

    # Verify output structure
    assert isinstance(pooled_encodings, list)
    for pooled, meta in pooled_encodings:
        assert isinstance(meta, dict)

        # Pooling-specific checks
        if pooling_fn == token_pool:
            assert meta["pool"] == "token"
            # Ensure the pooled outputs are NumPy arrays
            assert all(isinstance(p, np.ndarray) for p in pooled)
        elif pooling_fn == mean_pool:
            assert meta["pool"] == "mean"
            assert pooled.dim() == 2  # [batch_size, hidden_size]
        elif pooling_fn == sum_pool:
            assert meta["pool"] == "sum"
            assert pooled.dim() == 2  # [batch_size, hidden_size]
        elif pooling_fn == cls_pool:
            assert meta["pool"] == "cls"
            assert pooled.dim() == 2  # [batch_size, hidden_size]

    # Additional checks to ensure the mask was applied correctly (no NaNs in mean/sum poolers)
    if pooling_fn in [mean_pool, sum_pool]:
        for pooled, _ in pooled_encodings:
            assert not torch.isnan(pooled).any()

    # Optional: Check shapes depending on the layers
    if layers is not None:
        for _, meta in pooled_encodings:
            assert meta["layer"] in layers


@pytest.mark.parametrize("layers", [[0], [1], [0, 1], [12]])
def test_mean_pool_correctness(sample_text_data, layers):
    encoder, raw_encoding = sample_text_data

    encodings = raw_encoding["x"]  # This is a tuple of tensors from different layers
    mask = raw_encoding["mask"]

    # Apply the mean pooling function
    pooled_encodings = mean_pool(encodings, mask, layers=layers)

    # Manual calculation of the mean pooling only for the specified layers
    for i, layer in enumerate(layers):
        pooled, meta = pooled_encodings[i]
        assert meta["pool"] == "mean"

        # Manual mean calculation for each sample in the specified layer
        for j, sample_encoding in enumerate(encodings[layer]):
            masked_values = sample_encoding[mask[j].bool()]  # Apply the mask manually
            expected_mean = masked_values.mean(dim=0)

            # Compare the result from the mean_pool with the manually computed mean
            assert torch.allclose(expected_mean, pooled[j], atol=1e-6)


@pytest.mark.parametrize("layers", [[0], [1], [0, 1], [12]])
def test_sum_pool_correctness(sample_text_data, layers):
    encoder, raw_encoding = sample_text_data

    encodings = raw_encoding["x"]  # This is a tuple of tensors from different layers
    mask = raw_encoding["mask"]

    # Apply the sum pooling function
    pooled_encodings = sum_pool(encodings, mask, layers=layers)

    # Manual calculation of the sum pooling only for the specified layers
    for i, layer in enumerate(layers):
        pooled, meta = pooled_encodings[i]
        assert meta["pool"] == "sum"

        # Manual mean calculation for each sample in the specified layer
        for j, sample_encoding in enumerate(encodings[layer]):
            masked_values = sample_encoding[mask[j].bool()]  # Apply the mask manually
            expected_mean = masked_values.sum(dim=0)

            # Compare the result from the sum_pool with the manually computed mean
            assert torch.allclose(expected_mean, pooled[j], atol=1e-6)
