from __future__ import annotations

from typing import TYPE_CHECKING

from latentis.sample import Uniform
from latentis.space import LatentSpace

if TYPE_CHECKING:
    from latentis.types import Space


def test_space_len(space1: Space):
    if isinstance(space1, LatentSpace):
        assert len(space1) == space1.vectors.shape[0]


def test_space_repr(space1: Space):
    assert repr(space1)


def test_space_get(space1: Space):
    item = space1[0]
    if isinstance(space1, LatentSpace):
        assert item.shape[0] == space1.vectors.shape[-1]


def test_space_sample_hook(space1: Space):
    if isinstance(space1, LatentSpace):
        subspace = space1.sample(Uniform(), n=50)
        assert subspace.name == space1.name + "_sampled"
        assert len(subspace) == 50
