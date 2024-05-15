from typing import Any, Callable, Mapping, Optional

import torch
from torch import nn

from latentis.transform import Estimator, Identity, Transform
from latentis.transform.base import StandardScaling
from latentis.transform.dim_matcher import DimMatcher, ZeroPadding
from latentis.transform.translate.functional import sgd_affine_align_state, svd_align_state


class Translator(Estimator):
    def __init__(
        self,
        aligner: Estimator,
        name: Optional[str] = None,
        x_transform: Optional[Transform] = None,
        y_transform: Optional[Transform] = None,
        dim_matcher: Optional[DimMatcher] = None,
    ) -> None:
        super().__init__(name=name)
        self.x_transform = x_transform or Identity()
        self.y_transform = y_transform or Identity()
        self.aligner = aligner
        self.dim_matcher: DimMatcher = dim_matcher or Identity()

        self._fitted = False

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> Mapping[str, Any]:
        self.x_transform.fit(x)
        x, _ = self.x_transform.transform(x=x, y=None)

        self.y_transform.fit(y)
        y, _ = self.y_transform.transform(x=y, y=None)

        x, y = self.dim_matcher.fit_transform(x=x, y=y)

        self.aligner.fit(x, y)

        self._fitted = True

        return self

    def transform(self, x: torch.Tensor, y=None) -> torch.Tensor:
        assert self._fitted, "The transform should be fitted before being applied."

        x, _ = self.x_transform.transform(x=x, y=None)
        x, y = self.dim_matcher.transform(x=x, y=y)
        x, y = self.aligner.transform(x=x, y=y)
        _, x = self.dim_matcher.inverse_transform(x=None, y=x)
        x, _ = self.y_transform.inverse_transform(x=x, y=None)

        return x, y

    # def inverse_transform(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
    #     # assert x is None, "The inverse transform should be applied on the target space (y)"
    #     assert self._fitted, "The transform should be fitted before being applied."

    #     y = self.y_transform.transform(y)
    #     y = self.aligner.inverse_transform(y)

    #     return self.x_transform.inverse_transform(y), y


class MatrixAligner(Estimator):
    def __init__(
        self,
        name: str,
        align_fn_state: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> None:
        super().__init__(name=name)
        self.align_fn_state = align_fn_state

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> Mapping[str, Any]:
        state = self.align_fn_state(x=x, y=y)
        self._register_state(state=state)

        return self

    def transform(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        x = x @ self.get_state("matrix")

        return x, y

    def inverse_transform(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        raise NotImplementedError


class SGDAffineAligner(Estimator):
    def __init__(
        self,
        num_steps: int,
        lr: float,
        random_seed: int,
    ):
        super().__init__(name="sgd_affine_aligner")
        self.num_steps = num_steps
        self.lr = lr
        self.random_seed = random_seed

        self.translation: nn.Linear = None

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> Mapping[str, Any]:
        translation: nn.Module = sgd_affine_align_state(
            x=x, y=y, num_steps=self.num_steps, lr=self.lr, random_seed=self.random_seed
        )["translation"]
        self.translation = translation

        return self

    def transform(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        return self.translation(x), y


class Procrustes(Translator):
    def __init__(self) -> None:
        super().__init__(
            name="procrustes",
            aligner=MatrixAligner(name="svd_aligner", align_fn_state=svd_align_state),
            x_transform=StandardScaling(),
            y_transform=StandardScaling(),
            dim_matcher=ZeroPadding(),
        )
