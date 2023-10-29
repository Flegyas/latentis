from abc import abstractmethod
from typing import Any, Mapping, Optional

import torch
from torch import nn

from latentis.estimate.dim_matcher import DimMatcher, IdentityMatcher


class Estimator(nn.Module):
    def __init__(self, name: str, dim_matcher: Optional[DimMatcher]) -> None:
        super().__init__()
        self._name: str = name
        self.dim_matcher: Optional[DimMatcher] = dim_matcher if dim_matcher is not None else IdentityMatcher()

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def fit(self, source_data: torch.Tensor, target_data: torch.Tensor) -> Mapping[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class IdentityEstimator(Estimator):
    def __init__(self) -> None:
        super().__init__("identity")

    def fit(self, *args, **kwargs) -> Mapping[str, Any]:
        return {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
