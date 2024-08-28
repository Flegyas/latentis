from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch
import torch.nn.functional as F

from latentis.space import Space
from latentis.transform._abstract import Transform
from latentis.transform.functional import TransformFn

if TYPE_CHECKING:
    from latentis.types import LatentisSpace

_PROJECTIONS = {}


def projection_fn(
    name: str,
):
    """Decorator to register a projection function.

    Args:
        name: The name of the projection function.
        reverse_fn: The reverse function of the projection function.
        state: The state of the projection function.

    Returns:
        A decorator that registers the projection function.
    """
    assert name not in _PROJECTIONS, f"Projection function {name} already registered."

    def decorator(fn: TransformFn):
        assert callable(fn), f"projection_fn must be callable. projection_fn: {fn}"

        # result = transform_fn(name=name, reverse_fn=reverse_fn, state=state)(fn)
        fn.name = name

        _PROJECTIONS[name] = fn

        return fn

    return decorator


@projection_fn(name="CoB")
def change_of_basis_proj(x: torch.Tensor, *, anchors: torch.Tensor) -> torch.Tensor:
    return torch.linalg.lstsq(anchors.T, x.T)[0].T


@projection_fn(name="cosine")
def cosine_proj(
    x: torch.Tensor,
    *,
    anchors: torch.Tensor,
) -> torch.Tensor:
    x = F.normalize(x, p=2, dim=-1)
    anchors = F.normalize(anchors, p=2, dim=-1)

    x = x @ anchors.mT

    return x


@projection_fn(name="angular")
def angular_proj(
    x: torch.Tensor,
    *,
    anchors: torch.Tensor,
) -> torch.Tensor:
    x = F.normalize(x, p=2, dim=-1)
    anchors = F.normalize(anchors, p=2, dim=-1)

    x = (x @ anchors.mT).clamp(-1.0, 1.0)
    x = torch.arccos(x) / torch.pi

    return x


@projection_fn(name="lp")
def lp_proj(
    x: torch.Tensor,
    *,
    anchors: torch.Tensor,
    p: int,
) -> torch.Tensor:
    x = torch.cdist(x, anchors, p=p)

    return x


@projection_fn(name="euclidean")
def euclidean_proj(
    x: torch.Tensor,
    *,
    anchors: torch.Tensor,
) -> torch.Tensor:
    return lp_proj(x, anchors=anchors, p=2)


@projection_fn(name="l1")
def l1_proj(
    x: torch.Tensor,
    *,
    anchors: torch.Tensor,
) -> torch.Tensor:
    return lp_proj(x, anchors=anchors, p=1)


def pointwise_wrapper(func, unsqueeze: bool = False) -> Callable[..., torch.Tensor]:
    """This wrapper allows to apply a projection function pointwise to a batch of points and anchors.

    It is useful when the projection function does not support batched inputs.

    Args:
        func: The projection function to be wrapped.
        unsqueeze: If True, the first dimension of the inputs will be unsqueezed before applying the projection function.

    Returns:
        A wrapper function that applies the projection function pointwise.
    """
    unsqueeze = None if unsqueeze else ...

    def wrapper(x, anchors):
        rel_x = []
        for point in x:
            partial_rel_data = [
                func(point[unsqueeze], anchors=anchor[unsqueeze]) for anchor in anchors
            ]
            rel_x.append(partial_rel_data)
        rel_x = torch.as_tensor(rel_x, dtype=anchors.dtype, device=anchors.device)
        return rel_x

    return wrapper


def relative_projection(
    x: LatentisSpace,
    anchors: LatentisSpace,
    projection_fn: TransformFn,
):
    x_vectors = x.as_tensor() if isinstance(x, Space) else x
    anchor_vectors = anchors.as_tensor() if isinstance(anchors, Space) else anchors

    # absolute normalization/transformation
    transformed_x = x_vectors
    transformed_anchors = anchor_vectors

    # relative projection of x with respect to the anchors
    rel_x = projection_fn(x=transformed_x, anchors=transformed_anchors)

    if isinstance(x, Space):
        return Space.like(space=x, vector_source=rel_x)
    else:
        return rel_x


class RelativeProjection(Transform):
    def __init__(
        self,
        projection_fn: TransformFn,
        # abs_transform: Optional[Transform] = None,
        # rel_transform: Optional[Transform] = None,
    ):
        super().__init__()
        self.projection_fn = projection_fn

    def fit(self, anchors: LatentisSpace) -> "RelativeProjection":
        self._register_state(dict(anchors=anchors))
        return self

    def transform(self, x: LatentisSpace) -> torch.Tensor:
        rel_x = relative_projection(
            x, **self.get_state(), projection_fn=self.projection_fn
        )

        return rel_x
