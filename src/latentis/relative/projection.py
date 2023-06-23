import functools
from typing import Callable, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

TransformType = Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]


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
    x_norm = torch.norm(x, dim=-1)
    anchors_norm = torch.norm(anchors, dim=-1)

    # if the norm is already 1, then the cosine distance is just the dot product
    if not torch.allclose(x_norm, torch.ones_like(x_norm)) or not torch.allclose(
        anchors_norm, torch.ones_like(anchors_norm)
    ):
        x = x / x_norm.unsqueeze(-1)
        anchors = anchors / anchors_norm.unsqueeze(-1)

    return x @ anchors.T


def geodesic_dist(
    x: torch.Tensor,
    anchors: torch.Tensor,
) -> torch.Tensor:
    x_norm = torch.norm(x, dim=-1)
    anchors_norm = torch.norm(anchors, dim=-1)

    # if the norm is already 1, then the cosine distance is just the dot product
    if not torch.allclose(x_norm, torch.ones_like(x_norm)) or not torch.allclose(
        anchors_norm, torch.ones_like(anchors_norm)
    ):
        x = x / x_norm.unsqueeze(-1)
        anchors = anchors / anchors_norm.unsqueeze(-1)

    cos_dists = x @ anchors.T

    return torch.arccos(cos_dists)


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


def centering(x: torch.Tensor, anchors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    anchor_mean: torch.Tensor = anchors.mean(dim=0)
    return x - anchor_mean


def std_scaling(x: torch.Tensor, anchors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    anchor_std: torch.Tensor = anchors.std(dim=0)
    return x / anchor_std


def standard_scaling(x: torch.Tensor, anchors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return std_scaling(*centering(x=x, anchors=anchors))


class Transform(nn.Module):
    def __init__(self, name: str) -> None:
        super().__init__()
        self._name: str = name
        self.fit_data = {"std_anchors": ..., "mean_anchors": ...}

    @property
    def name(self) -> str:
        return self._name

    def fit(self, x: torch.Tensor, *args, **kwargs) -> None:
        raise NotImplementedError

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"


# TODO: Convert to Transform modules for efficiency
class Transforms:
    CENTERING = centering
    STD_SCALING = std_scaling
    STANDARD_SCALING = standard_scaling
    L2 = lambda x, anchors: F.normalize(x, p=2, dim=-1)  # noqa E731


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
        abs_transforms: Sequence[TransformType] | None = None,
        rel_transforms: Sequence[TransformType] | None = None,
    ) -> None:
        super().__init__()
        self.projection: TransformType = projection

        self.abs_transforms = (
            abs_transforms
            if isinstance(abs_transforms, Sequence)
            else []
            if abs_transforms is None
            else [abs_transforms]
        )
        self.rel_transforms = (
            rel_transforms
            if isinstance(rel_transforms, Sequence)
            else []
            if rel_transforms is None
            else [rel_transforms]
        )

    def set_anchors(self, anchors: torch.Tensor) -> None:
        orig_anchors = anchors.clone()
        self.register_buffer("orig_anchors", orig_anchors)

        for abs_transform in self.abs_transforms:
            anchors = abs_transform(x=anchors, anchors=anchors)

        self.register_buffer("anchors", anchors)
        return self

    @property
    def name(self) -> str:
        return self._name

    def forward(self, x: torch.Tensor, anchors: Optional[torch.Tensor]) -> torch.Tensor:
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
        abs_transforms: Sequence[TransformType] | None = None,
        rel_transforms: Sequence[TransformType] | None = None,
    ) -> None:
        super().__init__(name=name, projection=func, abs_transforms=abs_transforms, rel_transforms=rel_transforms)

    def forward(self, x: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        # absolute normalization/transformation
        transformed_x = x
        for abs_transform in self.abs_transforms:
            transformed_x = abs_transform(x=transformed_x, anchors=anchors)

        transformed_anchors = abs_transform(x=anchors, anchors=anchors)

        rel_x = []
        for point in x:
            partial_rel_data = []
            for anchor in transformed_anchors:
                partial_rel_data.append(self.projection(x=point, anchors=anchor))
            rel_x.append(partial_rel_data)
        rel_x = torch.as_tensor(rel_x, dtype=anchors.dtype, device=anchors.device)

        # relative normalization/transformation
        for rel_transform in self.rel_transforms:
            # todo: handle the case where the rel_transform needs additional arguments (e.g. rel_anchors, x, ...)
            rel_x = rel_transform(x=rel_x, anchors=anchors)

        return rel_x
