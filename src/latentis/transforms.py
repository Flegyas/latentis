from abc import abstractmethod
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import nn


# https://github.com/scikit-learn/scikit-learn/blob/7f9bad99d6e0a3e8ddf92a7e5561245224dab102/sklearn/preprocessing/_data.py#L87
def _handle_zeros(scale: torch.Tensor, copy=True, constant_mask=None):
    """Set scales of near constant features to 1.

    The goal is to avoid division by very small or zero values.

    Near constant features are detected automatically by identifying
    scales close to machine precision unless they are precomputed by
    the caller and passed with the `constant_mask` kwarg.

    Typically for standard scaling, the scales are the standard
    deviation while near constant features are better detected on the
    computed variances which are closer to machine precision by
    construction.
    """
    eps = torch.finfo(scale.dtype).eps

    # if we are fitting on 1D tensors, scale might be a scalar
    if scale.ndim == 0:
        return 1 if scale == 0 else scale
    elif isinstance(scale, torch.Tensor):
        if constant_mask is None:
            # Detect near constant values to avoid dividing by a very small
            # value that could lead to surprising results and numerical
            # stability issues.
            constant_mask = scale < 10 * eps

        if copy:
            # New tensor to avoid side-effects
            scale = scale.clone()
        scale[constant_mask] = 1.0
        scale[scale == 0.0] = 1.0
        return scale


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
        for key, value in self._fit(x=x, *args, **kwargs).items():
            self.register_buffer(key, value)
        self.fitted: bool = True

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def reverse(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"


class Centering(Transform):
    def __init__(self) -> None:
        super().__init__(name="centering")

    def _fit(self, data: torch.Tensor, *args, **kwargs) -> Mapping[str, torch.Tensor]:
        return {"shift": data.mean(dim=0)}

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        assert self.fitted, "The transform must be fit first."
        return x - self.shift

    def reverse(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        assert self.fitted, "The transform must be fit first."
        return x + self.shift


class STDScaling(Transform):
    def __init__(self) -> None:
        super().__init__(name="std_scaling")

    def _fit(self, data: torch.Tensor, *args, **kwargs) -> Mapping[str, torch.Tensor]:
        return {"scale": _handle_zeros(data.std(dim=0))}

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        assert self.fitted, "The transform must be fit first."

        # scale = _handle_zeros(data.std(dim=0)) if data is not None else self.scale
        return x / self.scale

    def reverse(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        assert self.fitted, "The transform must be fit first."

        # scale = _handle_zeros(data.std(dim=0)) if data is not None else self.scale
        return x * self.scale


class StandardScaling(Transform):
    def __init__(self) -> None:
        super().__init__(name="standard_scaling")

    def _fit(self, data: torch.Tensor, *args, **kwargs) -> Mapping[str, torch.Tensor]:
        return {"shift": data.mean(dim=0), "scale": _handle_zeros(data.std(dim=0))}

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        assert self.fitted, "The transform must be fit first."

        return (x - self.shift) / self.scale

    def reverse(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        assert self.fitted, "The transform must be fit first."

        return (x * self.scale) + self.shift


class L2(Transform):
    def __init__(self) -> None:
        super().__init__(name="l2")

    def _fit(self, data: torch.Tensor, *args, **kwargs) -> Mapping[str, torch.Tensor]:
        return {"mean_norm": data.norm(dim=1).mean()}

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return F.normalize(x, p=2, dim=-1)

    def reverse(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        assert self.fitted, "The transform must be fit first."

        return x * self.mean_norm


class ZeroPadding(Transform):
    def __init__(self, pad: int) -> None:
        super().__init__(name="zero_padding")
        self.register_buffer("pad", pad)

    def _fit(self, data: torch.Tensor, *args, **kwargs) -> Mapping[str, torch.Tensor]:
        return {}

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        assert x.ndim == 2, "The input tensor must be 2D."

        return torch.nn.functional.pad(x, (0, self.pad))

    def reverse(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        assert self.fitted, "The transform must be fit first."
        return x[..., : -self.pad]


class PCATruncation(Transform):
    def __init__(self, n_components: int) -> None:
        super().__init__(name="pca_truncation")
        self.n_components = n_components

    def _fit(self, data: torch.Tensor, *args, **kwargs) -> Mapping[str, torch.Tensor]:
        raise NotImplementedError

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def reverse(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError
