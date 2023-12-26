import functools
from typing import Callable, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from latentis.space import LatentSpace
from latentis.transform import Transform
from latentis.types import ProjectionFunc, Space


def change_of_basis_proj(x: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    return torch.linalg.lstsq(anchors.T, x.T)[0].T


def cosine_proj(
    x: torch.Tensor,
    anchors: torch.Tensor,
) -> torch.Tensor:
    x = F.normalize(x, p=2, dim=-1)
    anchors = F.normalize(anchors, p=2, dim=-1)

    return x @ anchors.mT


def angular_proj(
    x: torch.Tensor,
    anchors: torch.Tensor,
) -> torch.Tensor:
    x = F.normalize(x, p=2, dim=-1)
    anchors = F.normalize(anchors, p=2, dim=-1)

    return 1 - torch.arccos(x @ anchors.mT)


def lp_proj(
    x: torch.Tensor,
    anchors: torch.Tensor,
    p: int,
) -> torch.Tensor:
    return torch.cdist(x, anchors, p=p)


def euclidean_proj(
    x: torch.Tensor,
    anchors: torch.Tensor,
) -> torch.Tensor:
    return lp_proj(x=x, anchors=anchors, p=2)


l1_proj = functools.partial(lp_proj, p=1)


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
            partial_rel_data = [func(x=point[unsqueeze], anchors=anchor[unsqueeze]) for anchor in anchors]
            rel_x.append(partial_rel_data)
        rel_x = torch.as_tensor(rel_x, dtype=anchors.dtype, device=anchors.device)
        return rel_x

    return wrapper


def relative_projection(
    x: Space,
    anchors: Space,
    projection_fn: ProjectionFunc,
    abs_transforms: Optional[Sequence[Transform]] = None,
    rel_transforms: Optional[Sequence[Transform]] = None,
):
    abs_transforms = nn.ModuleList(
        abs_transforms if isinstance(abs_transforms, Sequence) else [] if abs_transforms is None else [abs_transforms]
    )
    rel_transforms = nn.ModuleList(
        rel_transforms if isinstance(rel_transforms, Sequence) else [] if rel_transforms is None else [rel_transforms]
    )

    x_vectors = x.vectors if isinstance(x, LatentSpace) else x
    anchor_vectors = anchors.vectors if isinstance(anchors, LatentSpace) else anchors

    # absolute normalization/transformation
    transformed_x = x_vectors
    transformed_anchors = anchor_vectors
    for abs_transform in abs_transforms:
        transformed_x = abs_transform(x=transformed_x, reference=transformed_anchors)
        transformed_anchors = abs_transform(x=transformed_anchors, reference=transformed_anchors)

    # relative projection of x with respect to the anchors
    rel_x = projection_fn(x=transformed_x, anchors=transformed_anchors)

    if len(rel_transforms) != 0:
        rel_anchors = projection_fn(x=transformed_anchors, anchors=transformed_anchors)
        # relative normalization/transformation
        for rel_transform in rel_transforms:
            rel_x = rel_transform(x=rel_x, reference=rel_anchors)

    if isinstance(x, LatentSpace):
        return LatentSpace.like(space=x, vector_source=rel_x)
    else:
        return rel_x


class RelativeProjection(nn.Module):
    def __init__(
        self,
        projection_fn: ProjectionFunc,
        name: Optional[str] = None,
        anchors: Optional[Space] = None,
        abs_transforms: Optional[Sequence[Transform]] = None,
        rel_transforms: Optional[Sequence[Transform]] = None,
    ) -> None:
        super().__init__()
        self.projection: ProjectionFunc = projection_fn
        self._name: str = (
            name
            if name is not None
            else f"relative_{projection_fn.__name__}"
            if hasattr(projection_fn, "__name__")
            else "relative_projection"
        )
        self.register_buffer("anchors", anchors.vectors if anchors is not None else None)

        self.abs_transforms = nn.ModuleList(
            abs_transforms
            if isinstance(abs_transforms, Sequence)
            else []
            if abs_transforms is None
            else [abs_transforms]
        )
        self.rel_transforms = nn.ModuleList(
            rel_transforms
            if isinstance(rel_transforms, Sequence)
            else []
            if rel_transforms is None
            else [rel_transforms]
        )

    @property
    def name(self) -> str:
        return self._name

    def forward(self, x: Space) -> Space:
        return relative_projection(
            x=x,
            anchors=self.anchors,
            projection_fn=self.projection,
            abs_transforms=self.abs_transforms,
            rel_transforms=self.rel_transforms,
        )
