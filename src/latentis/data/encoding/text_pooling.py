import logging
from abc import abstractmethod
from typing import Optional, Sequence, Tuple

import torch
from torch import nn

pylogger = logging.getLogger(__name__)


class Pooler(nn.Module):
    def __init__(self, name: str, output_dim: int):
        super().__init__()
        self.name: str = name
        self._output_dim: int = output_dim

    @abstractmethod
    def forward(self, x: torch.Tensor, *args, **kwargs):
        raise NotImplementedError

    @property
    def output_dim(self):
        return self._output_dim


class HFPooler(Pooler):
    def __init__(
        self,
        output_dim: int,
        pooling_fn: callable,
        layers: Optional[Sequence[int]] = None,
    ):
        assert all(isinstance(layer, int) and layer >= 0 for layer in layers)
        super().__init__(name=f"{pooling_fn.__name__}_{layers}", output_dim=output_dim)

        self.pooling_fn = pooling_fn
        self.layers = layers

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        return self.pooling_fn(encodings=x, mask=mask, layers=self.layers)


def token_pool(
    encodings: Tuple[torch.Tensor],
    mask: torch.Tensor,
    layers: Optional[Sequence[int]] = None,
):
    assert all(isinstance(layer, int) and layer >= 0 for layer in layers)
    layers = list(range(len(encodings))) if not layers else set(layers)

    token_encodings = [
        (
            [
                sample_encoding[sample_mask].cpu().numpy()
                for sample_encoding, sample_mask in zip(layer_encoding, mask)
            ],
            {"pool": "token", "layer": i_layer},
        )
        for i_layer, layer_encoding in enumerate(encodings)
        if i_layer in layers
    ]

    return token_encodings


def mean_pool(
    encodings: Tuple[torch.Tensor],
    mask: torch.Tensor,
    layers: Optional[Sequence[int]] = None,
):
    assert all(isinstance(layer, int) and layer >= 0 for layer in layers)
    layers = list(range(len(encodings))) if not layers else set(layers)

    pooled_encodings = []

    # Reshape the mask to match the dimensions of the encodings (batch_size x seq_len x hidden_size)
    expanded_mask = mask.unsqueeze(-1).float()

    for i_layer in layers:
        layer_encoding = encodings[i_layer]

        # Masked encodings - set padded positions to zero
        masked_encodings = layer_encoding * expanded_mask

        # Sum the masked encodings along the sequence dimension
        summed_encodings = masked_encodings.sum(dim=1)

        # Compute the number of valid (non-padded) tokens for each sequence
        valid_counts = expanded_mask.sum(dim=1).clamp(min=1)

        # Calculate the mean by dividing the sum by the number of valid tokens
        pooled_mean = summed_encodings / valid_counts

        pooled_encodings.append((pooled_mean, {"pool": "mean", "layer": i_layer}))

    return pooled_encodings


def sum_pool(
    encodings: Tuple[torch.Tensor],
    mask: torch.Tensor,
    layers: Optional[Sequence[int]] = None,
):
    assert all(isinstance(layer, int) and layer >= 0 for layer in layers)
    layers = list(range(len(encodings))) if not layers else set(layers)

    pooled_encodings = []

    # Reshape the mask to broadcast it over the sequence and hidden dimensions
    expanded_mask = mask.unsqueeze(-1).float()

    for i_layer in layers:
        layer_encoding = encodings[i_layer]

        # Apply the mask to the encodings
        masked_encodings = layer_encoding * expanded_mask

        # Sum over the sequence dimension
        summed_encodings = masked_encodings.sum(dim=1)

        pooled_encodings.append((summed_encodings, {"pool": "sum", "layer": i_layer}))

    return pooled_encodings


def cls_pool(
    encodings: Tuple[torch.Tensor], layers: Optional[Sequence[int]] = None, **kwargs
):
    assert all(isinstance(layer, int) and layer >= 0 for layer in layers)
    layers = list(range(len(encodings))) if not layers else set(layers)

    pooled_encodings = [
        (
            layer_encoding[:, 0, :],
            {"pool": "cls", "layer": i_layer},
        )  # TODO: adapt to encoders without CLS as first token
        for i_layer, layer_encoding in enumerate(encodings)
        if i_layer in layers
    ]

    return pooled_encodings
