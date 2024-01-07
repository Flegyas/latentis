from typing import Callable, Optional, Sequence

import torch
import torch.nn.functional as F

from latentis.space import LatentSpace
from latentis.transform import Transform
from latentis.transform.functional import ReverseFn, TransformFn, TransformResult, transform_fn
from latentis.types import Space

_PROJECTIONS = {}


def projection_fn(
    name: str,
    reverse_fn: Optional[ReverseFn] = None,
    state: Optional[Sequence[str]] = ["anchors"],
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

        result = transform_fn(name=name, reverse_fn=reverse_fn, state=state)(fn)
        result.name = name

        _PROJECTIONS[name] = result

        return result

    return decorator


@projection_fn(name="CoB")
def change_of_basis_proj(x: torch.Tensor, *, anchors: torch.Tensor) -> torch.Tensor:
    x = torch.linalg.lstsq(anchors.T, x.T)[0].T
    return TransformResult(x=x, state=dict(anchors=anchors))


@projection_fn(name="cosine")
def cosine_proj(
    x: torch.Tensor,
    *,
    anchors: torch.Tensor,
) -> torch.Tensor:
    x = F.normalize(x, p=2, dim=-1)
    anchors = F.normalize(anchors, p=2, dim=-1)

    x = x @ anchors.mT

    return TransformResult(x=x, state=dict(anchors=anchors))


@projection_fn(name="angular")
def angular_proj(
    x: torch.Tensor,
    *,
    anchors: torch.Tensor,
) -> torch.Tensor:
    x = F.normalize(x, p=2, dim=-1)
    anchors = F.normalize(anchors, p=2, dim=-1)

    x = (x @ anchors.mT).clamp(-1.0, 1.0)
    x = 1 - torch.arccos(x)

    return TransformResult(x=x, state=dict(anchors=anchors))


@projection_fn(name="lp")
def lp_proj(
    x: torch.Tensor,
    *,
    anchors: torch.Tensor,
    p: int,
) -> torch.Tensor:
    x = torch.cdist(x, anchors, p=p)

    return TransformResult(x=x, state=dict(anchors=anchors))


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
            partial_rel_data = [func(point[unsqueeze], anchors=anchor[unsqueeze]).x for anchor in anchors]
            rel_x.append(partial_rel_data)
        rel_x = torch.as_tensor(rel_x, dtype=anchors.dtype, device=anchors.device)
        return rel_x

    return wrapper


def relative_projection(
    x: Space,
    anchors: Space,
    projection_fn: TransformFn,
):
    if isinstance(x, LatentSpace):
        x = x.vectors
    if isinstance(anchors, LatentSpace):
        anchors = anchors.vectors

    return _PROJECTIONS[projection_fn.name](x, anchors=anchors).x


class RelativeProjection(Transform):
    def fit(self, x: torch.Tensor, y=None) -> None:
        self.register_buffer(f"{Transform._PREFIX}anchors", x)
        self._fitted: bool = True
        return self
