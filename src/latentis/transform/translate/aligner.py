from typing import Any, Callable, Mapping, Optional

import torch
from torch import nn

from latentis.transform import Estimator, Identity, Transform, TransformSequence
from latentis.transform._abstract import Estimator
from latentis.transform.base import StandardScaling
from latentis.transform.dim_matcher import DimMatcher, ZeroPadding
from latentis.transform.translate.functional import sgd_affine_align_state, svd_align_state


class Aligner(Estimator):
    def __init__(self, name: Optional[str] = None, dim_matcher: Optional[DimMatcher] = None) -> None:
        super().__init__(name=name)
        self.dim_matcher = dim_matcher


class Translator(Estimator):
    def __init__(
        self,
        aligner: Estimator,
        name: Optional[str] = None,
        x_transform: Optional[Transform] = None,
        y_transform: Optional[Transform] = None,
    ) -> None:
        super().__init__(name=name)
        self.x_transform = x_transform or Identity()
        self.y_transform = y_transform or Identity()
        self.aligner = aligner

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> Mapping[str, Any]:
        self.x_transform.fit(x)
        x = self.x_transform.transform(x)

        self.y_transform.fit(y)
        y = self.y_transform.transform(y)

        self.aligner.fit(x, y)

        return self

    def transform(self, x: torch.Tensor, y=None) -> torch.Tensor:
        x = self.x_transform.transform(x)
        x = self.aligner.transform(x=x)

        return self.y_transform.inverse_transform(x=x)

    def inverse_transform(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x is None, "The inverse transform should be applied on the target space (y)"

        y = self.y_transform.transform(y)
        y = self.aligner.inverse_transform(y)

        return self.x_transform.inverse_transform(y)


class MatrixAligner(Aligner):
    def __init__(
        self,
        name: str,
        align_fn_state: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        dim_matcher: Optional[DimMatcher] = None,
    ) -> None:
        super().__init__(name=name, dim_matcher=dim_matcher)
        self.align_fn_state = align_fn_state

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> Mapping[str, Any]:
        if self.dim_matcher is not None:
            self.dim_matcher.fit(x=x, y=y)
            x = self.dim_matcher.transform(x=x, y=None)
            y = self.dim_matcher.transform(x=None, y=y)

        state = self.align_fn_state(x=x, y=y)
        self._register_state(state=state)

        return self

    def transform(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        if self.dim_matcher is not None:
            x = self.dim_matcher.transform(x=x, y=None)

        x = x @ self.get_state("matrix")

        if self.dim_matcher is not None:
            x = self.dim_matcher.inverse_transform(x=None, y=x)

        return x

    def inverse_transform(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        raise NotImplementedError


class SGDAffineAligner(Aligner):
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
        return self.translation(x)


class Procrustes(Translator):
    def __init__(self) -> None:
        super().__init__(
            name="procrustes",
            aligner=TransformSequence(
                [
                    ZeroPadding(),
                    MatrixAligner(name="svd_aligner", align_fn_state=svd_align_state),
                ]
            ),
            x_transform=StandardScaling(),
            y_transform=StandardScaling(),
        )
