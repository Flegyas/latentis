from abc import abstractmethod
from typing import Any, Mapping

import torch
from torch import nn


class Estimator(nn.Module):
    def __init__(self, name: str) -> None:
        super().__init__()
        self._name: str = name

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def fit(self, source_data: torch.Tensor, target_data: torch.Tensor) -> Mapping[str, Any]:
        raise NotImplementedError


class IdentityEstimator(Estimator):
    def __init__(self) -> None:
        super().__init__("identity")

    def fit(self, *args, **kwargs) -> Mapping[str, Any]:
        return {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
