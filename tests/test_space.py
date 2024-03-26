from __future__ import annotations

from typing import TYPE_CHECKING

from latentis.sample import Uniform
from latentis.space import Space

if TYPE_CHECKING:
    from latentis.types import LatentisSpace


def test_space_len(space1: LatentisSpace):
    if isinstance(space1, Space):
        assert len(space1) == space1.vectors.shape[0]


def test_space_repr(space1: LatentisSpace):
    assert repr(space1)


def test_space_get(space1: LatentisSpace):
    item = space1[0]
    if isinstance(space1, Space):
        assert item.shape[0] == space1.vectors.shape[-1]


def test_space_sample_hook(space1: LatentisSpace):
    if isinstance(space1, Space):
        subspace = space1.sample(Uniform(), n=50)
        # assert subspace._name == space1._name + "_sampled"
        assert len(subspace) == 50
