import logging
from typing import Optional

import torch
import torch.nn.functional as F
from scipy.stats import ortho_group

from latentis.brick.abstract import Brick, BrickState

pylogger = logging.getLogger(__name__)


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


class Centering(Brick):
    def __init__(self) -> None:
        super().__init__(name="centering")

    def fit(self, reference: torch.Tensor, save: bool = True) -> BrickState:
        state = {"shift": reference.mean(dim=0)}
        if save:
            self.save_state(state)
        return state

    def forward(self, x: torch.Tensor, state: Optional[BrickState] = None) -> torch.Tensor:
        state = self._get_state(state)
        return x - state["shift"]

    def reverse(self, x: torch.Tensor, state: Optional[BrickState] = None) -> torch.Tensor:
        state = self._get_state(state)
        return x + state["shift"]


class STDScaling(Brick):
    def __init__(self) -> None:
        super().__init__(name="std_scaling")

    def fit(self, reference: torch.Tensor, save: bool = True) -> BrickState:
        state = {"scale": _handle_zeros(reference.std(dim=0))}
        if save:
            self.save_state(state)
        return state

    def forward(self, x: torch.Tensor, state: Optional[BrickState] = None) -> torch.Tensor:
        state = self._get_state(state)
        return x / state["scale"]

    def reverse(self, x: torch.Tensor, state: Optional[BrickState] = None) -> torch.Tensor:
        state = self._get_state(state)
        return x * state["scale"]


class StandardScaling(Brick):
    def __init__(self) -> None:
        super().__init__(name="standard_scaling")

    def fit(self, reference: torch.Tensor, save: bool = True) -> BrickState:
        state = {"shift": reference.mean(dim=0), "scale": _handle_zeros(reference.std(dim=0))}
        if save:
            self.save_state(state)
        return state

    def forward(self, x: torch.Tensor, state: Optional[BrickState] = None) -> torch.Tensor:
        state = self._get_state(state)
        return (x - state["shift"]) / state["scale"]

    def reverse(self, x: torch.Tensor, state: Optional[BrickState] = None) -> torch.Tensor:
        state = self._get_state(state)
        return (x * state["scale"]) + state["shift"]


class L2(Brick):
    def __init__(self) -> None:
        super().__init__(name="l2")

    def fit(self, reference: torch.Tensor, save: bool = True) -> BrickState:
        state = {"mean_norm": reference.norm(dim=1).mean()}
        if save:
            self.save_state(state)
        return state

    def forward(self, x: torch.Tensor, state: Optional[BrickState] = None) -> torch.Tensor:
        state = self._get_state(state)
        return F.normalize(x, p=2, dim=-1)

    def reverse(self, x: torch.Tensor, state: Optional[BrickState] = None) -> torch.Tensor:
        state = self._get_state(state)
        return x * state["mean_norm"]


class IsotropicScaling(Brick):
    def __init__(self, scale: float) -> None:
        super().__init__(name="isotropic_scaling")
        self.scale: float = scale

    def fit(self, reference: torch.Tensor, save: bool = True) -> BrickState:
        state = {"scale": torch.as_tensor(self.scale, dtype=reference.dtype, device=reference.device)}
        if save:
            self.save_state(state)
        return state

    def forward(self, x: torch.Tensor, state: Optional[BrickState] = None) -> torch.Tensor:
        state = self._get_state(state)
        return x * state["scale"]

    def reverse(self, x: torch.Tensor, state: Optional[BrickState] = None) -> torch.Tensor:
        state = self._get_state(state)
        return x / state["scale"]


class RandomIsotropicScaling(IsotropicScaling):
    def __init__(self, low: float, high: float, random_seed: int) -> None:
        scale = (
            torch.rand(size=(1,), dtype=torch.double, generator=torch.Generator().manual_seed(random_seed))
            * (high - low)
            + low
        )
        self.random_seed: int = random_seed
        super().__init__(scale=scale)


class DimensionPermutation(Brick):
    def __init__(self, permutation: torch.Tensor) -> None:
        super().__init__(name="dimension_permutation")
        self.permutation: torch.Tensor = permutation

    def fit(self, reference: torch.Tensor, save: bool = True) -> BrickState:
        state = {"permutation": self.permutation}
        if save:
            self.save_state(state)
        return state

    def forward(self, x: torch.Tensor, state: Optional[BrickState] = None) -> torch.Tensor:
        state = self._get_state(state)
        permutation = state["permutation"]
        return x[:, permutation]

    def reverse(self, x: torch.Tensor, state: Optional[BrickState] = None) -> torch.Tensor:
        state = self._get_state(state)
        permutation = state["permutation"]
        inverse_permutation = torch.zeros_like(permutation, dtype=torch.long, device=permutation.device)
        inverse_permutation[permutation] = torch.arange(len(permutation), dtype=torch.long, device=permutation.device)
        return x[:, inverse_permutation]


class RandomDimensionPermutation(Brick):
    def __init__(self, random_seed: int) -> None:
        super().__init__(name="random_dimension_permutation")
        self.random_seed: int = random_seed

    def fit(self, reference: torch.Tensor, save: bool = True) -> BrickState:
        d = reference.shape[1]
        permutation = torch.as_tensor(
            torch.randperm(d, generator=torch.Generator().manual_seed(self.random_seed)), dtype=torch.long
        )
        state = {"permutation": permutation}
        if save:
            self.save_state(state)
        return state

    forward = DimensionPermutation.forward
    reverse = DimensionPermutation.reverse


class Isometry(Brick):
    def __init__(self, matrix: torch.Tensor) -> None:
        super().__init__(name="isometry")
        self.matrix: torch.Tensor = matrix

    def fit(self, reference: torch.Tensor, save: bool = True) -> BrickState:
        state = {"matrix": self.matrix}
        if save:
            self.save_state(state)
        return state

    def forward(self, x: torch.Tensor, state: Optional[BrickState] = None) -> torch.Tensor:
        state = self._get_state(state)
        matrix = state["matrix"]
        return x @ matrix

    def reverse(self, x: torch.Tensor, state: Optional[BrickState] = None) -> torch.Tensor:
        state = self._get_state(state)
        matrix = state["matrix"]
        return x @ matrix.T


class RandomIsometry(Brick):
    def fit(self, reference: torch.Tensor, save: bool = True) -> BrickState:
        d = reference.shape[1]
        matrix = torch.as_tensor(ortho_group.rvs(d, random_state=self.random_seed), dtype=torch.double)

        state = {"matrix": matrix}
        if save:
            self.save_state(state)
        return state

    def __init__(self, random_seed: int) -> None:
        super().__init__(name="random_isometry")
        self.random_seed: int = random_seed

    forward = Isometry.forward
    reverse = Isometry.reverse
