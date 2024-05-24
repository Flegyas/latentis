from typing import Any, Mapping

import torch
import torch.nn.functional as F

import latentis.transform.functional as FL
from latentis.transform import FuncXTransform, Transform


class Centering(FuncXTransform):
    def __init__(self):
        super().__init__(
            transform_fn=FL.centering_transform,
            state_fn=FL.centering_state,
            inverse_fn=FL.centering_inverse,
            name="centering",
        )


class STDScaling(FuncXTransform):
    def __init__(self):
        super().__init__(
            transform_fn=FL.std_scaling_transform,
            state_fn=FL.std_scaling_state,
            inverse_fn=FL.std_scaling_inverse,
            name="std_scaling",
        )


class StandardScaling(FuncXTransform):
    def __init__(self):
        super().__init__(
            transform_fn=FL.standard_scaling_transform,
            state_fn=FL.standard_scaling_state,
            inverse_fn=FL.standard_scaling_inverse,
            name="standard_scaling",
        )


class LPNorm(Transform):
    def __init__(self, p: int = 2):
        super().__init__(
            name=f"l{p}_norm",
        )
        self.p = p

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return FL.lp_normalize_transform(x=x, p=self.p)


class MeanLPNorm(Transform):
    def __init__(self, p: int = 2, dim: int = -1):
        super().__init__(name=f"mean_l{p}_norm", invertible=True)
        self.p = p
        self.dim = dim

    def fit(self, x: torch.Tensor) -> "MeanLPNorm":
        mean_norm: torch.Tensor = x.norm(p=self.p, dim=-1).mean()
        self._register_state(dict(mean_norm=mean_norm))
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: decide if we want to use the state or not
        return F.normalize(x, p=self.p, dim=-1)

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.get_state("mean_norm")


class IsotropicScaling(Transform):
    def __init__(self, scale: float):
        super().__init__(name="isotropic_scaling", invertible=True)

        self.scale: float = scale

    def fit(self, x: torch.Tensor) -> "IsotropicScaling":
        self._register_state({"scale": torch.tensor(self.scale, dtype=x.dtype, device=x.device)})
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return FL.isotropic_scaling_transform(x=x, **self.get_state())

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        return FL.isotropic_scaling_inverse(x=x, **self.get_state())


class RandomIsometry(Transform):
    def __init__(self, random_seed: int):
        super().__init__(name="random_isometry", invertible=True)
        self.random_seed: int = random_seed

    def fit(self, x: torch.Tensor) -> "RandomIsometry":
        self._register_state(FL.random_isometry_state(x=x, random_seed=self.random_seed))
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return FL.isometry_transform(x=x, **self.get_state())

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        return FL.isometry_inverse(x=x, **self.get_state())


class RandomDimensionPermutation(Transform):
    def __init__(self, random_seed: int):
        super().__init__(name="random_dimension_permutation", invertible=True)
        self.random_seed: int = random_seed

    def fit(self, x: torch.Tensor, y=None) -> "RandomDimensionPermutation":
        self._register_state(FL.random_dimension_permutation_state(x=x, random_seed=self.random_seed))

        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return FL.dimension_permutation_transform(x=x, **self.get_state())

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        return FL.dimension_permutation_inverse(x=x, **self.get_state())


class InverseTransform(Transform):
    def __init__(self, transform: Transform) -> None:
        super().__init__(name=f"reversed_{transform.name}", invertible=True)
        if not transform.invertible:
            raise ValueError(f"Transform {transform.name} is not invertible.")

        self._transform = transform

    def fit(self, x: torch.Tensor) -> Mapping[str, Any]:
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return self._transform.inverse_transform(x=x)

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        return self._transform.transform(x=x)
