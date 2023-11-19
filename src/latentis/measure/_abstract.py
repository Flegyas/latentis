from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

import torch
from torch import nn

import latentis

if TYPE_CHECKING:
    from latentis.types import Space


class PairwiseMetric(nn.Module):
    def __init__(self, name: str) -> None:
        super().__init__()
        self._name: str = name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def forward(self, x: Space, y: Space) -> Sequence[Mapping[str, Any]]:
        raise NotImplementedError


class MetricFn(PairwiseMetric):
    def __init__(self, key: str, fn: Callable([Space, Space], torch.Tensor)) -> None:
        super().__init__(fn.__name__ if hasattr(fn, "__name__") else key)
        self.key = key
        self.fn = fn

    def forward(self, x: Space, y: Space) -> Mapping[str, Any]:
        if isinstance(x, latentis.LatentSpace):
            x = x.vectors

        if isinstance(y, latentis.LatentSpace):
            y = y.vectors

        return {self.key: self.fn(x, y)}
