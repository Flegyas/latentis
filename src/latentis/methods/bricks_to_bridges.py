from __future__ import annotations

from typing import Sequence, Optional, Type, TYPE_CHECKING
from torch import nn
import torch
import logging
from latentis.transform.projection import relative_projection
from latentis.transform._abstract import Transform, TransformSequence
from latentis.transform._abstract import Identity

if TYPE_CHECKING:
    from latentis.types import Space

log = logging.getLogger(__name__)


class BricksToBridges(Transform):
    def __init__(
        self,
        projection_fns: Sequence[TransformSequence],
        aggregation_module: Type[nn.Module],
        aggregate_transforms: Optional[TransformSequence] = None,
        abs_transforms: Optional[Sequence[TransformSequence]] = None,
        rel_transforms: Optional[Sequence[TransformSequence]] = None,
    ):
        super().__init__()
        self.projection_fns = projection_fns
        self.aggregation_module = aggregation_module

        default_length = len(projection_fns)
        self.aggregate_transforms = (
            aggregate_transforms if aggregate_transforms is not None else [Identity()] * default_length
        )
        self.abs_transforms = abs_transforms if abs_transforms is not None else [Identity()] * default_length
        self.rel_transforms = rel_transforms if rel_transforms is not None else [Identity()] * default_length

        assert len(self.abs_transforms) == len(
            self.projection_fns
        ), "abs_transforms must have the same length as projection_fns"

    def fit(self, x: Space, **kwargs) -> "BricksToBridges":
        relative_spaces = []
        for abs_transform, rel_transform, projection_fn in zip(
            self.abs_transforms,
            self.rel_transforms,
            self.projection_fns,
        ):
            abs_transform.fit(x)
            tx = abs_transform.transform(x)

            self._register_state({f"abs_transform_{abs_transform}": tx})

            rel_x = relative_projection(tx, anchors=tx, projection_fn=projection_fn)
            rel_transform.fit(rel_x)
            t_rel_x = rel_transform.transform(rel_x)

            relative_spaces.append(t_rel_x)

        self.aggregation_module.fit(torch.stack(relative_spaces, dim=1))

        return self

    def transform(self, x: Space) -> Sequence[torch.Tensor]:
        relative_spaces = []

        for abs_transform, rel_transform, projection_fn in zip(
            self.abs_transforms,
            self.rel_transforms,
            self.projection_fns,
        ):
            tx = abs_transform.transform(x)
            rel_x = relative_projection(tx, anchors=tx, projection_fn=projection_fn)
            t_rel_x = rel_transform.transform(rel_x)
            relative_spaces.append(t_rel_x)

        return relative_spaces

    def aggregate(self, relative_spaces: Sequence[torch.Tensor]) -> torch.Tensor:
        return self.aggregation_module(torch.stack(relative_spaces, dim=1))

    def get_num_subspaces(self):
        return len(self.transforms)

    def get_subspace_dim(self):
        return self.transforms[0].out_dim

    def get_out_dim(self):
        return self.aggregation_module.out_dim


class SumAggregation(nn.Module):
    def __init__(self, subspace_dim: int, num_subspaces: int):
        super().__init__()

        self.subspace_dim = subspace_dim
        self.num_subspaces = num_subspaces

        log.info(f"{__class__.__name__}: subspace_dim={self.subspace_dim}, num_subspaces={self.num_subspaces}")

    @property
    def out_dim(self):
        return self.subspace_dim

    def forward(self, concat_subspaces: Sequence[torch.Tensor]) -> torch.Tensor:
        concat_subspaces = concat_subspaces.split(self.subspace_dim, dim=1)

        out = [norm_layer(x) for norm_layer, x in zip(self.norm_layers, concat_subspaces)]

        return torch.stack(out, dim=1).sum(dim=1)


class LinearSumAggregation(SumAggregation):
    def __init__(self, subspace_dim: int, num_subspaces: int):
        super().__init__(subspace_dim, num_subspaces)

        self.norm_layers = nn.ModuleList([nn.LayerNorm(subspace_dim) for _ in range(num_subspaces)])


class NonLinearSumAggregation(SumAggregation):
    def __init__(self, subspace_dim: int, num_subspaces: int):
        super().__init__(subspace_dim, num_subspaces)

        self.norm_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(subspace_dim),
                    nn.Linear(subspace_dim, subspace_dim),
                    nn.Tanh(),
                )
                for _ in range(num_subspaces)
            ]
        )
