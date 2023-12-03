from typing import Tuple

import pytest
import torch

from latentis import LatentSpace
from latentis.measure.cka import CKAMode
from latentis.types import Space


class SameShapeSpaces(object):
    instances = [
        (
            LatentSpace(
                vectors=torch.randn(space1_n, space_1_dim, dtype=torch.double),
                name="space1",
            ),
            LatentSpace(
                vectors=torch.randn(space2_n, space2_dim, dtype=torch.double),
                name="space2",
            ),
        )
        for (space1_n, space_1_dim), (space2_n, space2_dim) in [
            ((50, 250), (50, 250)),
            ((20, 300), (20, 300)),
        ]
    ] + [
        (
            torch.randn(space1_n, space_1_dim, dtype=torch.double),
            torch.randn(space2_n, space2_dim, dtype=torch.double),
        )
        for (space1_n, space_1_dim), (space2_n, space2_dim) in [
            ((50, 250), (50, 250)),
            ((20, 300), (20, 300)),
        ]
    ]


class DifferentDimSpaces(object):
    instances = [
        (
            LatentSpace(
                vectors=torch.randn(space1_n, space_1_dim, dtype=torch.double),
                name="space1",
            ),
            LatentSpace(
                vectors=torch.randn(space2_n, space2_dim, dtype=torch.double),
                name="space2",
            ),
        )
        for (space1_n, space_1_dim), (space2_n, space2_dim) in [
            ((50, 250), (50, 300)),  # Same number of samples, different dimensions
            ((20, 300), (20, 150)),
        ]
    ] + [
        (
            torch.randn(space1_n, space_1_dim, dtype=torch.double),
            torch.randn(space2_n, space2_dim, dtype=torch.double),
        )
        for (space1_n, space_1_dim), (space2_n, space2_dim) in [
            ((50, 250), (50, 300)),  # Same number of samples, different dimensions
            ((20, 300), (20, 150)),
        ]
    ]


@pytest.fixture(params=SameShapeSpaces().instances)
def same_shape_spaces(request) -> Tuple[Space, Space]:
    return request.param


@pytest.fixture(params=DifferentDimSpaces().instances)
def different_dim_spaces(request) -> Tuple[Space, Space]:
    return request.param


class PrecomputedCKA(object):

    stored_space1 = torch.tensor(
        [
            [0.0, 1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0, 7.0],
            [8.0, 9.0, 10.0, 11.0],
            [12.0, 13.0, 14.0, 15.0],
            [16.0, 17.0, 18.0, 19.0],
            [20.0, 21.0, 22.0, 23.0],
            [24.0, 25.0, 26.0, 27.0],
            [28.0, 29.0, 30.0, 31.0],
        ]
    )

    stored_space2 = torch.tensor(
        [
            [1.0, 2.0, 27.0, 22.0],
            [0.0, 28.0, 15.0, 18.0],
            [12.0, 9.0, 14.0, 8.0],
            [11.0, 19.0, 16.0, 13.0],
            [31.0, 20.0, 10.0, 3.0],
            [24.0, 5.0, 7.0, 29.0],
            [4.0, 30.0, 21.0, 6.0],
            [26.0, 17.0, 23.0, 25.0],
        ]
    )

    stored_linear_cka_res = 0.2815
    stored_rbf_cka_res = 0.3945

    params = {
        "stored_space1": stored_space1,
        "stored_space2": stored_space2,
        CKAMode.LINEAR: stored_linear_cka_res,
        CKAMode.RBF: stored_rbf_cka_res,
    }


@pytest.fixture
def precomputed_cka():
    cka_instance = PrecomputedCKA()

    return cka_instance.params
