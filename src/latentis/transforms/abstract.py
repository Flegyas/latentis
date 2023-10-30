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


class Independent(Transform):
    @staticmethod
    @abstractmethod
    def compute_stats(reference: torch.Tensor) -> Mapping[str, torch.Tensor]:
        raise NotImplementedError

    def get_stats(self) -> Mapping[str, torch.Tensor]:
        return {k.removeprefix(_PREFIX): v for k, v in self.state_dict().items() if k.startswith(_PREFIX)}

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


class Joint(Transform):
    @abstractmethod
    def _fit(self, source_data: torch.Tensor, target_data: torch.Tensor, *args, **kwargs) -> Mapping[str, torch.Tensor]:
        raise NotImplementedError

    def fit(self, source_data: torch.Tensor, target_data: torch.Tensor, *args, **kwargs) -> None:
        for key, value in self._fit(source_data=source_data, target_data=target_data, *args, **kwargs).items():
            self.register_buffer(key, value)
        self.fitted: bool = True

    def forward(
        self, source_x: Optional[torch.Tensor], target_x: Optional[torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        raise NotImplementedError

    def reverse(
        self, source_x: Optional[torch.Tensor], target_x: Optional[torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        raise NotImplementedError
