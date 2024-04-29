from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Mapping, Tuple

import torch
from torch import nn

from latentis.serialize.io_utils import SerializableMixin
from latentis.space import Space

if TYPE_CHECKING:
    from latentis.types import LatentisSpace


class Metric(nn.Module, SerializableMixin):
    def __init__(self, name: str) -> None:
        super().__init__()
        self._name: str = name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def forward(self, *spaces: LatentisSpace) -> Mapping[Tuple[str], Mapping[str, Any]]:
        raise NotImplementedError


class PairwiseMetric(Metric):
    def __init__(self, name: str) -> None:
        super().__init__(name=name)

    @abstractmethod
    def forward(self, x: LatentisSpace, y: LatentisSpace) -> Mapping[str, Any]:
        raise NotImplementedError


class MetricFn(PairwiseMetric):
    def __init__(self, key: str, fn: Callable[[LatentisSpace, LatentisSpace], torch.Tensor]) -> None:
        super().__init__(fn.__name__ if hasattr(fn, "__name__") else key)
        self.key = key
        self.fn = fn

    def forward(self, x: LatentisSpace, y: LatentisSpace) -> Mapping[str, Any]:
        if isinstance(x, Space):
            x = x.vectors

        if isinstance(y, Space):
            y = y.vectors

        return {self.key: self.fn(x, y)}
