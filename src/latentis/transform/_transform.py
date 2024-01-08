import torch

import latentis.transform.functional as FL
from latentis.transform._abstract import SimpleTransform, Transform


class Centering(SimpleTransform):
    def __init__(self):
        super().__init__(
            transform_fn=FL.centering_transform,
            state_fn=FL.centering_state,
            inverse_fn=FL.centering_inverse,
            name="centering",
        )


class STDScaling(SimpleTransform):
    def __init__(self):
        super().__init__(
            transform_fn=FL.std_scaling_transform,
            state_fn=FL.std_scaling_state,
            inverse_fn=FL.std_scaling_inverse,
            name="std_scaling",
        )


class StandardScaling(SimpleTransform):
    def __init__(self):
        super().__init__(
            transform_fn=FL.standard_scaling_transform,
            state_fn=FL.standard_scaling_state,
            inverse_fn=FL.standard_scaling_inverse,
            name="centering",
        )


class LPNorm(Transform):
    def __init__(self, p: int = 2):
        super().__init__(
            name=f"l{p}_norm",
        )
        self.p = p

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return FL.lp_normalize_transform(x=x, p=self.p)


class IsotropicScaling(Transform):
    def __init__(self, scale: float):
        super().__init__(name="isotropic_scaling")

        self.scale: float = scale

    def fit(self, x: torch.Tensor) -> "IsotropicScaling":
        self._register_state({"scale": torch.tensor(self.scale, dtype=x.dtype, device=x.device)})
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return FL.isotropic_scaling_transform(x=x, **self.get_state())

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return FL.isotropic_scaling_inverse(x=x, **self.get_state())


class RandomIsometry(Transform):
    def __init__(self, random_seed: int):
        super().__init__(name="random_isometry")
        self.random_seed: int = random_seed

    def fit(self, x: torch.Tensor, y=None) -> "RandomIsometry":
        self._register_state(FL.random_isometry_state(x=x, random_seed=self.random_seed))
        return self

    def transform(self, x: torch.Tensor, y=None) -> torch.Tensor:
        return FL.isometry_transform(x=x, **self.get_state())

    def inverse(self, x: torch.Tensor, y=None) -> torch.Tensor:
        return FL.isometry_inverse(x=x, **self.get_state())


class RandomDimensionPermutation(Transform):
    def __init__(self, random_seed: int):
        super().__init__(name="random_dimension_permutation")
        self.random_seed: int = random_seed

    def fit(self, x: torch.Tensor, y=None) -> "RandomDimensionPermutation":
        self._register_state(FL.random_dimension_permutation_state(x=x, random_seed=self.random_seed))

        return self

    def transform(self, x: torch.Tensor, y=None) -> torch.Tensor:
        return FL.dimension_permutation_transform(x=x, **self.get_state())

    def inverse(self, x: torch.Tensor, y=None) -> torch.Tensor:
        return FL.dimension_permutation_inverse(x=x, **self.get_state())
