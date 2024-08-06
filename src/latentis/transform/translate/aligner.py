from typing import Any, Callable, Mapping, Optional

import torch
from torch import nn

from latentis.correspondence._base import PI
from latentis.transform import Estimator, Identity, Transform
from latentis.transform.base import StandardScaling
from latentis.transform.dim_matcher import DimMatcher, ZeroPadding
from latentis.transform.translate.functional import sgd_affine_align_state, svd_align_state


class Translator(Estimator):
    @property
    def metadata(self) -> Mapping[str, Any]:
        super_metadata = super().metadata
        super_metadata.update({f"aligner.{k}": v for k, v in self.aligner.metadata.items()})
        if self.dim_matcher is not None:
            super_metadata.update({f"dim_matcher.{k}": v for k, v in self.dim_matcher.metadata.items()})

        return super_metadata

    def __init__(
        self,
        aligner: Estimator,
        name: Optional[str] = None,
        x_transform: Optional[Transform] = None,
        y_transform: Optional[Transform] = None,
        dim_matcher: Optional[DimMatcher] = None,
    ) -> None:
        super().__init__(name=name or "translator")
        self.x_transform = x_transform or Identity()
        self.y_transform = y_transform or Identity()
        self.aligner = aligner
        self.dim_matcher: DimMatcher = dim_matcher

        self._fitted = False

    def fit(self, x: torch.Tensor, y: torch.Tensor, pi: PI = None, **kwargs) -> Mapping[str, Any]:
        x_anchors = x[pi.x_indices] if pi is not None else x
        y_anchors = y[pi.y_indices] if pi is not None else y

        x_anchors = self.x_transform.fit_transform(x=x_anchors)
        y_anchors = self.y_transform.fit_transform(x=y_anchors)

        if self.dim_matcher is not None:
            x_anchors, y_anchors = self.dim_matcher.fit_transform(x=x_anchors, y=y_anchors)

        self.aligner.fit(x=x_anchors, y=y_anchors)

        self._fitted = True

        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        assert self._fitted, "The transform should be fitted before being applied."

        x = self.x_transform.transform(x=x)
        if self.dim_matcher is not None:
            x = self.dim_matcher.transform(x=x)[0]
        x = self.aligner.transform(x=x)
        if self.dim_matcher is not None:
            x = self.dim_matcher.inverse_transform(x=None, y=x)
        x = self.y_transform.inverse_transform(x=x)

        return x

    # def inverse_transform(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
    #     # assert x is None, "The inverse transform should be applied on the target space (y)"
    #     assert self._fitted, "The transform should be fitted before being applied."

    #     y = self.y_transform.transform(y)
    #     y = self.aligner.inverse_transform(y)

    #     return self.x_transform.inverse_transform(y), y


class MatrixAligner(Estimator):
    @property
    def metadata(self) -> Mapping[str, Any]:
        super_metadata = super().metadata
        super_metadata["align_fn_state"] = self.align_fn_state.__name__
        return super_metadata

    def __init__(
        self,
        name: str,
        align_fn_state: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> None:
        super().__init__(name=name)
        self.align_fn_state = align_fn_state

    def fit(self, x: torch.Tensor, y: torch.Tensor, pi: PI = None) -> Mapping[str, Any]:
        x_anchors = x[pi.x_indices] if pi is not None else x
        y_anchors = y[pi.y_indices] if pi is not None else y

        state = self.align_fn_state(x=x_anchors, y=y_anchors)
        self._register_state(state=state)

        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        x = x @ self.get_state("matrix")

        return x

    def inverse_transform(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        raise NotImplementedError


class SGDAffineAligner(Estimator):
    @property
    def metadata(self) -> Mapping[str, Any]:
        super_metadata = super().metadata
        super_metadata["num_steps"] = self.num_steps
        super_metadata["lr"] = self.lr
        super_metadata["random_seed"] = self.random_seed

        return super_metadata

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

    # TODO: add a parameter to control the gradients on the output
    def fit(self, x: torch.Tensor, y: torch.Tensor) -> Mapping[str, Any]:
        translation: nn.Module = sgd_affine_align_state(
            x=x, y=y, num_steps=self.num_steps, lr=self.lr, random_seed=self.random_seed
        )["translation"]
        self.translation = translation

        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return self.translation(x)


class Procrustes(Translator):
    def __init__(self) -> None:
        super().__init__(
            name="procrustes",
            aligner=MatrixAligner(name="svd_aligner", align_fn_state=svd_align_state),
            x_transform=StandardScaling(),
            y_transform=StandardScaling(),
            dim_matcher=ZeroPadding(),
        )
