from abc import abstractmethod
from typing import Mapping, Optional

import torch
from torch import nn

_PREFIX: str = "latentis_stat_"


class Transform(nn.Module):
    def __init__(self, name: str) -> None:
        super().__init__()
        self._name: str = name
        self.fitted: bool = False

    @property
    def name(self) -> str:
        return self._name

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"

    def get_stats(self) -> Mapping[str, torch.Tensor]:
        return {k[len(_PREFIX) :]: v for k, v in self.state_dict().items() if k.startswith(_PREFIX)}

    @abstractmethod
    def compute_stats(self, reference: torch.Tensor) -> Mapping[str, torch.Tensor]:
        raise NotImplementedError

    def fit(self, reference: torch.Tensor, *args, **kwargs) -> None:
        for key, value in self.compute_stats(reference=reference, *args, **kwargs).items():
            self.register_buffer(f"{_PREFIX}{key}", value)
        self.fitted: bool = True

    @abstractmethod
    def _forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor, reference: Optional[torch.Tensor] = None, *args, **kwargs) -> torch.Tensor:
        assert self.fitted or reference is not None, "The transform must be fit first or reference must be not None."

        stats: Mapping[str, torch.Tensor] = (
            self.compute_stats(reference=reference) if reference is not None else self.get_stats()
        )

        return self._forward(x=x, **stats)

    @abstractmethod
    def _reverse(x: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def reverse(self, x: torch.Tensor, reference: Optional[torch.Tensor] = None, *args, **kwargs) -> torch.Tensor:
        assert self.fitted or reference is not None, "The transform must be fit first or reference must be not None."

        stats: Mapping[str, torch.Tensor] = (
            self.compute_stats(reference=reference) if reference is not None else self.get_stats()
        )

        return self._reverse(x=x, **stats)
