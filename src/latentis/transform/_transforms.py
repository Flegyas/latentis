from typing import Mapping

import torch
import torch.nn.functional as F
from scipy.stats import ortho_group

from latentis.transform.abstract import Transform


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


class Centering(Transform):
    def compute_stats(self, reference: torch.Tensor) -> Mapping[str, torch.Tensor]:
        return {"shift": reference.mean(dim=0)}

    def __init__(self) -> None:
        super().__init__(name="centering")

    def _forward(self, x: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        return x - shift

    def _reverse(self, x: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        return x + shift


class STDScaling(Transform):
    def compute_stats(self, reference: torch.Tensor) -> Mapping[str, torch.Tensor]:
        return {"scale": _handle_zeros(reference.std(dim=0))}

    def __init__(self) -> None:
        super().__init__(name="std_scaling")

    def _forward(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return x / scale

    def _reverse(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return x * scale


class StandardScaling(Transform):
    def compute_stats(self, reference: torch.Tensor) -> Mapping[str, torch.Tensor]:
        return {"shift": reference.mean(dim=0), "scale": _handle_zeros(reference.std(dim=0))}

    def __init__(self) -> None:
        super().__init__(name="standard_scaling")

    def _forward(self, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return (x - shift) / scale

    def _reverse(self, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return (x * scale) + shift


class L2(Transform):
    def compute_stats(self, reference: torch.Tensor) -> Mapping[str, torch.Tensor]:
        return {"mean_norm": reference.norm(dim=1).mean()}

    def __init__(self) -> None:
        super().__init__(name="l2")

    def _forward(self, x: torch.Tensor, mean_norm: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2, dim=-1)

    def _reverse(self, x: torch.Tensor, mean_norm: torch.Tensor) -> torch.Tensor:
        return x * mean_norm


class IsotropicScaling(Transform):
    def compute_stats(self, reference: torch.Tensor) -> Mapping[str, torch.Tensor]:
        return {"scale": torch.as_tensor(self.scale, dtype=reference.dtype, device=reference.device)}

    def __init__(self, scale: float) -> None:
        super().__init__(name="isotropic_scaling")
        self.scale: float = scale

    def _forward(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return x * scale

    def _reverse(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return x / scale


class RandomIsotropicScaling(IsotropicScaling):
    def __init__(self, low: float, high: float, random_seed: int) -> None:
        scale = (
            torch.rand(size=(1,), dtype=torch.double, generator=torch.Generator().manual_seed(random_seed))
            * (high - low)
            + low
        )
        self.random_seed: int = random_seed
        super().__init__(scale=scale)


class DimensionPermutation(Transform):
    def compute_stats(self, reference: torch.Tensor) -> Mapping[str, torch.Tensor]:
        return {"permutation": self.permutation}

    def __init__(self, permutation: torch.Tensor) -> None:
        super().__init__(name="dimension_permutation")
        self.permutation: torch.Tensor = permutation

    def _forward(self, x: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
        return x[:, permutation]

    def _reverse(self, x: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
        inverse_permutation = torch.zeros_like(permutation, dtype=torch.long, device=permutation.device)
        inverse_permutation[permutation] = torch.arange(len(permutation), dtype=torch.long, device=permutation.device)
        return x[:, inverse_permutation]


class RandomDimensionPermutation(Transform):
    def __init__(self, random_seed: int) -> None:
        super().__init__(name="random_dimension_permutation")
        self.random_seed: int = random_seed

    def compute_stats(self, reference: torch.Tensor) -> Mapping[str, torch.Tensor]:
        d = reference.shape[1]
        permutation = torch.as_tensor(
            torch.randperm(d, generator=torch.Generator().manual_seed(self.random_seed)), dtype=torch.long
        )
        return {"permutation": permutation}

    def _forward(self, x: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
        return x[:, permutation]

    def _reverse(self, x: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
        inverse_permutation = torch.zeros_like(permutation, dtype=torch.long, device=permutation.device)
        inverse_permutation[permutation] = torch.arange(len(permutation), dtype=torch.long, device=permutation.device)
        return x[:, inverse_permutation]


class Isometry(Transform):
    def compute_stats(self, reference: torch.Tensor) -> Mapping[str, torch.Tensor]:
        return {"matrix": self.matrix}

    def __init__(self, matrix: torch.Tensor) -> None:
        super().__init__(name="isometry")
        self.matrix: torch.Tensor = matrix

    def _forward(self, x: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        return x @ matrix

    def _reverse(self, x: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        return x @ matrix.T


class RandomIsometry(Transform):
    def compute_stats(self, reference: torch.Tensor) -> Mapping[str, torch.Tensor]:
        d = reference.shape[1]
        matrix = torch.as_tensor(ortho_group.rvs(d, random_state=self.random_seed), dtype=torch.double)

        return {"matrix": matrix}

    def __init__(self, random_seed: int) -> None:
        super().__init__(name="random_isometry")
        self.random_seed: int = random_seed

    def _forward(self, x: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        return x @ matrix

    def _reverse(self, x: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        return x @ matrix.T
