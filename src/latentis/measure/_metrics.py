from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

import torch
from torch import nn

import latentis

if TYPE_CHECKING:
    from latentis.types import Space


class Metric(nn.Module):
    def __init__(self, name: str) -> None:
        super().__init__()
        self._name: str = name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def _forward(self, space1: Space, space2: Space) -> Mapping[str, Any]:
        raise NotImplementedError

    def forward(self, space1: Space, *others: Space) -> Sequence[Mapping[str, Any]]:
        result = [self._forward(space1, other) for other in others]

        return result[0] if len(result) == 1 else result


# TODO
class MetricFn(Metric):
    def __init__(self, key: str, fn: Callable([Space, Space], torch.Tensor)) -> None:
        super().__init__(fn.__name__ if hasattr(fn, "__name__") else key)
        self.key = key
        self.fn = fn

    def _forward(self, space1: Space, space2: Space) -> Mapping[str, Any]:
        if isinstance(space1, latentis.LatentSpace):
            space1 = space1.vectors

        if isinstance(space2, latentis.LatentSpace):
            space2 = space2.vectors

        return {self.key: self.fn(space1, space2)}
