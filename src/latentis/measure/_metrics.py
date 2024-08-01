from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable, Mapping, Sequence, Union

import numpy as np
import torch
from torch import nn

from latentis.space import Space


class Metric(nn.Module):
    def __init__(self, name: str, device: Union[str, torch.device] = None) -> None:
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


class MetricFn(Metric):
    def __init__(self, key: str, fn: Callable([Space, Space], torch.Tensor)) -> None:
        super().__init__(fn.__name__ if hasattr(fn, "__name__") else key)
        self.key = key
        self.fn = fn

    def _forward(self, space1: Space, space2: Space) -> Mapping[str, Any]:
        if isinstance(space1, Space):
            space1 = space1.as_tensor()

        if isinstance(space2, Space):
            space2 = space2.as_tensor()

        return {self.key: self.fn(space1, space2)}


def preprocess_latent_space_args(func):
    def wrapper(*args, **kwargs):
        if "space1" in kwargs and "space2" in kwargs:
            if isinstance(kwargs["space1"], Space):
                kwargs["space1"] = kwargs["space1"].as_tensor()
            if isinstance(kwargs["space2"], Space):
                kwargs["space2"] = kwargs["space2"].as_tensor()

            if isinstance(kwargs["space1"], np.ndarray):
                kwargs["space1"] = torch.tensor(kwargs["space1"])
            if isinstance(kwargs["space2"], np.ndarray):
                kwargs["space2"] = torch.tensor(kwargs["space2"])

        args = list(args)
        if len(args) >= 2:
            if isinstance(args[0], Space):
                args[0] = args[0].as_tensor()
            if isinstance(args[1], Space):
                args[1] = args[1].as_tensor()
            if isinstance(args[0], np.ndarray):
                args[0] = torch.tensor(args[0])
            if isinstance(args[1], np.ndarray):
                args[1] = torch.tensor(args[1])

        return func(*args, **kwargs)

    return wrapper
