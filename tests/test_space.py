from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy
import numpy as np
import pytest
import torch

from latentis.sample import Uniform
from latentis.space import Space
from latentis.space.vector_source import HDF5Source

if TYPE_CHECKING:
    from latentis.types import LatentisSpace
import tempfile


def test_space_len(space1: LatentisSpace):
    if isinstance(space1, Space):
        assert len(space1) == space1.as_tensor().shape[0]


def test_space_repr(space1: LatentisSpace):
    assert repr(space1)


def test_space_get(space1: LatentisSpace):
    item = space1[0]
    if isinstance(space1, Space):
        assert item.shape[0] == space1.as_tensor().shape[-1]


def test_space_sample_hook(space1: LatentisSpace):
    if isinstance(space1, Space):
        subspace = space1.sample(Uniform(), n=50)
        # assert subspace._name == space1._name + "_sampled"
        assert len(subspace) == 50


@pytest.mark.parametrize(
    "data, selection",
    [
        (
            torch.arange(0, 3 * 3 * 3 * 3).reshape(3, 3, 3, 3).float(),
            (slice(None), np.array([2, 0, 0]), slice(None), 0),
        ),
        (
            torch.arange(0, 3 * 3 * 3 * 3).reshape(3, 3, 3, 3).float(),
            (slice(None), slice(None), slice(None), 0),
        ),
        (
            torch.arange(0, 3 * 3 * 3 * 3).reshape(3, 3, 3, 3).float(),
            (slice(None), slice(None), slice(None), slice(None)),
        ),
        (
            torch.arange(0, 3 * 3 * 3 * 3).reshape(3, 3, 3, 3).float(),
            (slice(None), [1, 1, 1, 1], slice(None), slice(None)),
        ),
        (
            torch.arange(0, 3 * 3 * 3 * 3).reshape(3, 3, 3, 3).float(),
            [1, 1, 1, 1],
        ),
        (
            torch.arange(0, 3 * 3 * 3 * 3).reshape(3, 3, 3, 3).float(),
            [0, 2],
        ),
        (
            torch.arange(0, 3 * 3 * 3 * 3).reshape(3, 3, 3, 3).float(),
            np.array([0, 0, 2, 1]),
        ),
        (
            torch.arange(0, 3 * 3 * 3 * 3).reshape(3, 3, 3, 3).float(),
            torch.tensor([0, 0, 2, 1]).long(),
        ),
        (
            torch.arange(0, 5**5).reshape(*([5] * 5)).float(),
            (
                slice(None),
                numpy.random.choice(5, 3),
                slice(None),
                1,
                3,
            ),
        ),
    ],
)
def test_hdf5_vector_source(data: torch.Tensor, selection: Sequence):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        space = Space(vector_source=HDF5Source(shape=data.shape, root_dir=tmp_dir))
        space.add_vectors(data)

        assert len(space) == data.shape[0]
        assert torch.allclose(space.as_tensor(), data)
        assert space._vector_source.root_dir == tmp_dir
        assert torch.allclose(space[selection], data[selection])
