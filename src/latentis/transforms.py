from abc import abstractmethod
from typing import Mapping, Optional

import torch
import torch.nn.functional as F
from torch import nn


class Transform(nn.Module):
    def __init__(self, name: str) -> None:
        super().__init__()
        self._name: str = name
        self.fitted: bool = False

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def _fit(self, x: torch.Tensor, *args, **kwargs) -> Mapping[str, torch.Tensor]:
        raise NotImplementedError

    def fit(self, x: torch.Tensor, *args, **kwargs) -> None:
        raise NotImplementedError
        # TODO: Implement fit method
        for key, value in self._fit(x=x, *args, **kwargs).items():
            self.register_buffer(key, value)
        self.fitted: bool = True

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def reverse(self, x: torch.Tensor, anchors: Optional[torch.Tensor] = None, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"


class Transforms:
    class Centering(Transform):
        def __init__(self) -> None:
            super().__init__(name="centering")

        def _fit(self, anchors: torch.Tensor, *args, **kwargs) -> Mapping[str, torch.Tensor]:
            return {"shift": anchors.mean(dim=0)}

        def forward(self, x: torch.Tensor, anchors: Optional[torch.Tensor] = None, *args, **kwargs) -> torch.Tensor:
            assert (
                anchors is not None or self.fitted
            ), "Either anchors must be provided or the transform must be fit first."

            shift = anchors.mean(dim=0) if anchors is not None else self.shift
            return x - shift

        def reverse(self, x: torch.Tensor, anchors: Optional[torch.Tensor] = None, *args, **kwargs) -> torch.Tensor:
            assert (
                anchors is not None or self.fitted
            ), "Either anchors must be provided or the transform must be fit first."

            shift = anchors.mean(dim=0) if anchors is not None else self.shift
            return x + shift

    class STDScaling(Transform):
        def __init__(self) -> None:
            super().__init__(name="std_scaling")

        def _fit(self, anchors: torch.Tensor, *args, **kwargs) -> Mapping[str, torch.Tensor]:
            return {"scale": anchors.std(dim=0)}

        def forward(self, x: torch.Tensor, anchors: Optional[torch.Tensor] = None, *args, **kwargs) -> torch.Tensor:
            assert (
                anchors is not None or self.fitted
            ), "Either anchors must be provided or the transform must be fit first."

            scale = anchors.std(dim=0) if anchors is not None else self.scale
            return x / scale

        def reverse(self, x: torch.Tensor, anchors: Optional[torch.Tensor] = None, *args, **kwargs) -> torch.Tensor:
            assert (
                anchors is not None or self.fitted
            ), "Either anchors must be provided or the transform must be fit first."

            scale = anchors.std(dim=0) if anchors is not None else self.scale
            return x * scale

    class StandardScaling(Transform):
        def __init__(self) -> None:
            super().__init__(name="standard_scaling")

        def _fit(self, anchors: torch.Tensor, *args, **kwargs) -> Mapping[str, torch.Tensor]:
            return {"shift": anchors.mean(dim=0), "scale": anchors.std(dim=0)}

        def forward(self, x: torch.Tensor, anchors: Optional[torch.Tensor] = None, *args, **kwargs) -> torch.Tensor:
            assert (
                anchors is not None or self.fitted
            ), "Either anchors must be provided or the transform must be fit first."

            shift = anchors.mean(dim=0) if anchors is not None else self.shift
            scale = anchors.std(dim=0) if anchors is not None else self.scale

            return (x - shift) / scale

        def reverse(self, x: torch.Tensor, anchors: Optional[torch.Tensor] = None, *args, **kwargs) -> torch.Tensor:
            assert (
                anchors is not None or self.fitted
            ), "Either anchors must be provided or the transform must be fit first."

            shift = anchors.mean(dim=0) if anchors is not None else self.shift
            scale = anchors.std(dim=0) if anchors is not None else self.scale

            return (x * scale) + shift

    class L2(Transform):
        def __init__(self) -> None:
            super().__init__(name="l2")

        def _fit(self, anchors: torch.Tensor, *args, **kwargs) -> Mapping[str, torch.Tensor]:
            return {"mean_norm": anchors.norm(dim=1).mean()}

        def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            return F.normalize(x, p=2, dim=-1)

        def reverse(self, x: torch.Tensor, anchors: Optional[torch.Tensor] = None, *args, **kwargs) -> torch.Tensor:
            assert (
                anchors is not None or self.fitted
            ), "Either anchors must be provided or the transform must be fit first."

            mean_norm = anchors.norm(dim=1).mean() if anchors is not None else self.mean_norm

            return x * mean_norm
