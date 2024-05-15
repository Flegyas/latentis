from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Mapping, Tuple

import torch
from torch import nn

from latentis.serialize.io_utils import IndexableMixin
from latentis.space import LatentSpace

if TYPE_CHECKING:
    from latentis.types import Space


class Metric(nn.Module, IndexableMixin):
    def __init__(self, name: str) -> None:
        super().__init__()
        self._name: str = name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def forward(self, *spaces: Space) -> Mapping[Tuple[str], Mapping[str, Any]]:
        raise NotImplementedError


class PairwiseMetric(Metric):
    def __init__(self, name: str) -> None:
        super().__init__(name=name)

    @abstractmethod
    def forward(self, x: Space, y: Space) -> Mapping[str, Any]:
        raise NotImplementedError


class MetricFn(PairwiseMetric):
    def __init__(self, key: str, fn: Callable([Space, Space], torch.Tensor)) -> None:
        super().__init__(fn.__name__ if hasattr(fn, "__name__") else key)
        self.key = key
        self.fn = fn

    def forward(self, x: Space, y: Space) -> Mapping[str, Any]:
        if isinstance(x, LatentSpace):
            x = x.vectors

        if isinstance(y, LatentSpace):
            y = y.vectors

        return {self.key: self.fn(x, y)}
