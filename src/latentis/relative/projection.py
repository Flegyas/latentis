import functools
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from latentis.types import TransformType


class RelativeProjection(nn.Module):
    def __init__(self, name: str, func: TransformType) -> None:
        super().__init__()
        self._name: str = name
        self.func: TransformType = func

    @property
    def name(self) -> str:
        return self._name

    def forward(self, x: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        return self.func(x=x, anchors=anchors)


def change_of_basis(x: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    return torch.linalg.lstsq(anchors.T, x.T)[0].T


def cosine_dist(
    x: torch.Tensor,
    anchors: torch.Tensor,
) -> torch.Tensor:
    x = F.normalize(x, p=2, dim=-1)
    anchors = F.normalize(anchors, p=2, dim=-1)

    return x @ anchors.T


def geodesic_dist(
    x: torch.Tensor,
    anchors: torch.Tensor,
) -> torch.Tensor:
    x = F.normalize(x, p=2, dim=-1)
    anchors = F.normalize(anchors, p=2, dim=-1)

    return 1 - torch.arccos(x @ anchors.T)


def lp_dist(
    x: torch.Tensor,
    anchors: torch.Tensor,
    p: int,
) -> torch.Tensor:
    return torch.cdist(x, anchors, p=p)


def euclidean_dist(
    x: torch.Tensor,
    anchors: torch.Tensor,
) -> torch.Tensor:
    return lp_dist(x=x, anchors=anchors, p=2)


class Projections:
    COSINE = RelativeProjection(name="cosine", func=cosine_dist)
    GEODESIC = RelativeProjection(name="geodesic", func=geodesic_dist)
    COB = RelativeProjection(name="change_of_basis", func=change_of_basis)
    EUCLIDEAN = RelativeProjection(name="euclidean", func=euclidean_dist)
    L1 = RelativeProjection(name="l1", func=functools.partial(lp_dist, p=1))


class RelativeProjector(nn.Module):
    # TODO: Add support for anchor optimization
    def __init__(
        self,
        projection: RelativeProjection,
        anchors: Optional[torch.Tensor] = None,
        abs_transforms: Optional[Sequence[TransformType]] = None,
        rel_transforms: Optional[Sequence[TransformType]] = None,
    ) -> None:
        super().__init__()
        self.projection: TransformType = projection

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

        if anchors is not None:
            self.register_buffer("anchors", anchors)

    def set_anchors(self, anchors: torch.Tensor) -> "RelativeProjector":
        self.register_buffer("anchors", anchors)
        return self

    @property
    def name(self) -> str:
        return self._name

    def forward(self, x: torch.Tensor, anchors: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert anchors is not None or self.anchors is not None, "anchors must be provided"
        anchors = anchors if anchors is not None else self.anchors

        # absolute normalization/transformation
        transformed_x = x
        transformed_anchors = anchors
        for abs_transform in self.abs_transforms:
            transformed_x = abs_transform(x=transformed_x, anchors=anchors)
            transformed_anchors = abs_transform(x=transformed_anchors, anchors=anchors)

        # relative projection of x with respect to the anchors
        rel_x = self.projection(x=transformed_x, anchors=transformed_anchors)

        # relative normalization/transformation
        for rel_transform in self.rel_transforms:
            # TODO: handle the case where the rel_transform needs additional arguments (e.g. rel_anchors, x, ...)
            # maybe with a custom Compose object with kwargs
            rel_x = rel_transform(x=rel_x, anchors=anchors)

        return rel_x


class PointWiseProjector(RelativeProjector):
    def __init__(
        self,
        name: str,
        func: TransformType,
        anchors: Optional[torch.Tensor] = None,
        abs_transforms: Optional[Sequence[TransformType]] = None,
        rel_transforms: Optional[Sequence[TransformType]] = None,
    ) -> None:
        super().__init__(
            name=name, projection=func, anchors=anchors, abs_transforms=abs_transforms, rel_transforms=rel_transforms
        )

    def forward(self, x: torch.Tensor, anchors: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert anchors is not None or self.anchors is not None, "anchors must be provided"
        anchors = anchors if anchors is not None else self.anchors

        # absolute normalization/transformation
        transformed_x = x
        transformed_anchors = anchors
        for abs_transform in self.abs_transforms:
            transformed_x = abs_transform(x=transformed_x, anchors=anchors)
            transformed_anchors = abs_transform(x=transformed_anchors, anchors=anchors)

        # relative projection of x with respect to the anchors
        rel_x = []
        for point in x:
            partial_rel_data = []
            for anchor in transformed_anchors:
                partial_rel_data.append(self.projection(x=point, anchors=anchor))
            rel_x.append(partial_rel_data)
        rel_x = torch.as_tensor(rel_x, dtype=anchors.dtype, device=anchors.device)

        # relative normalization/transformation
        for rel_transform in self.rel_transforms:
            # TODO: handle the case where the rel_transform needs additional arguments (e.g. rel_anchors, x, ...)
            # maybe with a custom Compose object with kwargs
            rel_x = rel_transform(x=rel_x, anchors=anchors)

        return rel_x
