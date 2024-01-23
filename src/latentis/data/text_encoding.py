import logging
from abc import abstractmethod
from typing import Optional, Sequence, Tuple

import torch
from torch import nn

pylogger = logging.getLogger(__name__)


class Pooler(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.name: str = name

    @abstractmethod
    def forward(self, x: torch.Tensor, *args, **kwargs):
        raise NotImplementedError


class HFPooler(Pooler):
    def __init__(self, pooling_fn: callable, layers: Optional[Sequence[int]] = None):
        assert all(isinstance(layer, int) and layer >= 0 for layer in layers)
        super().__init__(name=f"{pooling_fn.__name__}_{layers}")
        self.pooling_fn = pooling_fn
        self.layers = layers

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        return self.pooling_fn(encodings=x, mask=mask, layers=self.layers)


def token_pool(encodings: Tuple[torch.Tensor], mask: torch.Tensor, layers: Optional[Sequence[int]] = None):
    assert all(isinstance(layer, int) and layer >= 0 for layer in layers)
    layers = list(range(len(encodings))) if not layers else set(layers)

    token_encodings = [
        (
            [sample_encoding[sample_mask].cpu().numpy() for sample_encoding, sample_mask in zip(layer_encoding, mask)],
            {"pool": "token", "layer": i_layer},
        )
        for i_layer, layer_encoding in enumerate(encodings)
        if i_layer in layers
    ]

    return token_encodings


def mean_pool(encodings: Tuple[torch.Tensor], mask: torch.Tensor, layers: Optional[Sequence[int]] = None):
    assert all(isinstance(layer, int) and layer >= 0 for layer in layers)
    layers = list(range(len(encodings))) if not layers else set(layers)

    pooled_encodings = [
        (
            torch.stack(
                [
                    sample_encoding[sample_mask].mean(dim=0)
                    for sample_encoding, sample_mask in zip(layer_encoding, mask)
                ],
                dim=0,
            ),
            {"pool": "mean", "layer": i_layer},
        )
        for i_layer, layer_encoding in enumerate(encodings)
        if i_layer in layers
    ]

    return pooled_encodings


def sum_pool(encodings: Tuple[torch.Tensor], mask: torch.Tensor, layers: Optional[Sequence[int]] = None):
    assert all(isinstance(layer, int) and layer >= 0 for layer in layers)
    layers = list(range(len(encodings))) if not layers else set(layers)

    pooled_encodings = [
        (
            torch.stack(
                [sample_encoding[sample_mask].sum(dim=0) for sample_encoding, sample_mask in zip(layer_encoding, mask)],
                dim=0,
            ),
            {"pool": "sum", "layer": i_layer},
        )
        for i_layer, layer_encoding in enumerate(encodings)
        if i_layer in layers
    ]

    return pooled_encodings


def cls_pool(encodings: Tuple[torch.Tensor], layers: Optional[Sequence[int]] = None, **kwargs):
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
