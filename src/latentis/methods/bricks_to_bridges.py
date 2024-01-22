from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Sequence, Type

import torch
from torch import nn
from torch.nn import functional as F

from latentis.transform._abstract import Identity, Transform, TransformSequence
from latentis.transform.projection import RelativeProjection

if TYPE_CHECKING:
    from latentis.types import Space

log = logging.getLogger(__name__)


class Aggregation(nn.Module):
    def __init__(self, subspace_dim: int, num_subspaces: int):
        super().__init__()
        self.subspace_dim = subspace_dim
        self.num_subspaces = num_subspaces

    def fit(self, x: torch.Tensor, **kwargs) -> "Aggregation":
        raise NotImplementedError("fit method must be implemented in the specific aggregator")


class ConcatAggregation(Aggregation):
    def __init__(self, subspace_dim: int, num_subspaces: int):
        super().__init__(
            subspace_dim=subspace_dim,
            num_subspaces=num_subspaces,
        )

        log.info(f"ConcatAggregation: subspace_dim={self.subspace_dim}, num_subspaces={self.num_subspaces}")

        self.norm_layers = nn.ModuleList([nn.LayerNorm(subspace_dim) for _ in range(num_subspaces)])

    @property
    def out_dim(self):
        return self.subspace_dim * self.num_subspaces

    def fit(self, x: torch.Tensor, **kwargs) -> "ConcatAggregation":
        return self

    def forward(self, subspaces: Sequence[torch.Tensor]) -> torch.Tensor:
        out = [norm_layer(x) for norm_layer, x in zip(self.norm_layers, subspaces)]

        return torch.cat(out, dim=1)


class SumAggregation(Aggregation):
    def __init__(self, subspace_dim: int, num_subspaces: int):
        super().__init__(
            subspace_dim=subspace_dim,
            num_subspaces=num_subspaces,
        )

    @property
    def out_dim(self):
        return self.subspace_dim

    def fit(self, x: torch.Tensor, **kwargs) -> "SumAggregation":
        return self

    def forward(self, subspaces: Sequence[torch.Tensor]) -> torch.Tensor:
        out = [norm_layer(x) for norm_layer, x in zip(self.norm_layers, subspaces)]

        return torch.stack(out, dim=1).sum(dim=1)


class LinearSumAggregation(SumAggregation):
    def __init__(self, subspace_dim: int, num_subspaces: int):
        super().__init__(subspace_dim, num_subspaces)

        log.info(f"{__class__.__name__}: subspace_dim={self.subspace_dim}, num_subspaces={self.num_subspaces}")

        self.norm_layers = nn.ModuleList([nn.LayerNorm(subspace_dim) for _ in range(num_subspaces)])


class NonLinearSumAggregation(SumAggregation):
    def __init__(self, subspace_dim: int, num_subspaces: int):
        super().__init__(subspace_dim, num_subspaces)

        log.info(f"{__class__.__name__}: subspace_dim={self.subspace_dim}, num_subspaces={self.num_subspaces}")

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


class WeightedAvgAggregation(Aggregation):
    def __init__(self, subspace_dim: int, num_subspaces: int):
        super().__init__(subspace_dim, num_subspaces)

        log.info(f"WeightedAvgAggregation: subspace_dim={self.subspace_dim}, num_subspaces={self.num_subspaces}")

        self.weight = nn.Parameter(torch.ones(num_subspaces))

        self.norm_layers = nn.ModuleList([nn.LayerNorm(subspace_dim) for _ in range(num_subspaces)])

    @property
    def out_dim(self):
        return self.subspace_dim

    def fit(self, x: torch.Tensor, **kwargs) -> "WeightedAvgAggregation":
        return self

    def forward(self, subspaces: Sequence[torch.Tensor]) -> torch.Tensor:
        out = [norm_layer(x) for norm_layer, x in zip(self.norm_layers, subspaces)]

        softmax_weights = F.softmax(self.weight, dim=0)
        concat_subspaces = torch.stack(subspaces, dim=1)
        out = torch.einsum("bns,n -> bs", concat_subspaces, softmax_weights)

        return out


class SelfAttentionLayer(Aggregation):
    def __init__(self, subspace_dim: int, num_subspaces: int):
        super().__init__(subspace_dim, num_subspaces)

        log.info(f"SelfAttentionLayer: subspace_dim={self.subspace_dim}, num_subspaces={self.num_subspaces}")

        self.attention = nn.MultiheadAttention(embed_dim=self.subspace_dim, num_heads=1, batch_first=True)
        self.norm_layers = nn.ModuleList([nn.LayerNorm(self.subspace_dim) for _ in range(self.num_subspaces)])

    @property
    def out_dim(self):
        return self.subspace_dim

    def fit(self, x: torch.Tensor, **kwargs) -> "WeightedAvgAggregation":
        return self

    def forward(self, subspaces: Sequence[torch.Tensor]):
        query = [norm_layer(x) for norm_layer, x in zip(self.norm_layers, subspaces)]
        query = torch.stack(query, dim=1)

        out, _ = self.attention(query=query, key=query, value=query)

        return torch.sum(out, dim=1)


class BricksToBridges(Transform):
    def __init__(
        self,
        projection_fns: Sequence[TransformSequence],
        aggregation_type: Type[Aggregation],
        num_anchors: int,
        aggregate_transforms: Optional[TransformSequence] = None,
        abs_transforms: Optional[Sequence[TransformSequence]] = None,
        rel_transforms: Optional[Sequence[TransformSequence]] = None,
    ):
        super().__init__()
        self.projection_fns = projection_fns
        self.subspace_dim = num_anchors
        self.num_subspaces = len(projection_fns)
        self.aggregation_module = aggregation_type(subspace_dim=self.subspace_dim, num_subspaces=self.num_subspaces)

        self.aggregate_transforms = aggregate_transforms or Identity()
        self.abs_transforms = abs_transforms if abs_transforms is not None else [Identity()] * self.num_subspaces
        self.rel_transforms = rel_transforms if rel_transforms is not None else [Identity()] * self.num_subspaces

        self.rel_projs = nn.ModuleList(
            [
                RelativeProjection(
                    projection_fn=projection_fn, abs_transform=abs_transform, rel_transform=rel_transform
                )
                for abs_transform, rel_transform, projection_fn in zip(
                    self.abs_transforms, self.rel_transforms, self.projection_fns
                )
            ]
        )

        assert len(self.projection_fns) > 1, "projection_fns must be at least 2"

        assert (
            len(self.abs_transforms) == len(self.projection_fns) == len(self.rel_transforms)
        ), "abs_transforms must have the same length as projection_fns"

        duplicates = [
            item.__name__ + " = " + str(projection_fns.count(item))
            for item in set(projection_fns)
            if projection_fns.count(item) > 1
        ]
        if duplicates:
            log.warning(f"There are some projection functions that are duplicated: {duplicates}")

    def fit(self, x: Space, **kwargs) -> "BricksToBridges":
        relative_spaces = []

        for rel_proj in self.rel_projs:
            rel_proj.fit(x)
            tx = rel_proj.transform(x)
            relative_spaces.append(tx)

        self.aggregation_module.fit(relative_spaces)
        rel_aggregation = self.aggregation_module(relative_spaces)
        self.aggregate_transforms.fit(rel_aggregation)

        return self

    def transform(self, x: Space) -> torch.Tensor:
        relative_spaces = []

        for rel_proj in self.rel_projs:
            tx = rel_proj.transform(x)
            relative_spaces.append(tx.vectors)

        rel_aggregation = self.aggregation_module(relative_spaces)
        t_rel_agg = self.aggregate_transforms.transform(rel_aggregation)

        return t_rel_agg
