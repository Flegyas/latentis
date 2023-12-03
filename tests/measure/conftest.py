from typing import Tuple

import pytest
import torch

from latentis import LatentSpace
from latentis.types import Space
from latentis.measure.cka import CKAMode

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
            ((1000, 100), (1000, 100)),
        ]
    ] + [
        (
            torch.randn(space1_n, space_1_dim, dtype=torch.double),
            torch.randn(space2_n, space2_dim, dtype=torch.double),
        )
        for (space1_n, space_1_dim), (space2_n, space2_dim) in [
            ((50, 250), (50, 250)),
            ((20, 300), (20, 300)),
            ((1000, 100), (1000, 100)),
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
            ((50, 250), (50, 300)),   # Same number of samples, different dimensions
            ((20, 300), (20, 150)),
        ]
    ] + [
        (
            torch.randn(space1_n, space_1_dim, dtype=torch.double),
            torch.randn(space2_n, space2_dim, dtype=torch.double),
        )
        for (space1_n, space_1_dim), (space2_n, space2_dim) in [
            ((50, 250), (50, 300)),   # Same number of samples, different dimensions
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

    stored_space1 = torch.tensor([[ 0.,  1.,  2.,  3.],
        [ 4.,  5.,  6.,  7.],
        [ 8.,  9., 10., 11.],
        [12., 13., 14., 15.],
        [16., 17., 18., 19.],
        [20., 21., 22., 23.],
        [24., 25., 26., 27.],
        [28., 29., 30., 31.]])

    stored_space2 = torch.tensor([[ 1.,  2., 27., 22.],
            [ 0., 28., 15., 18.],
            [12.,  9., 14.,  8.],
            [11., 19., 16., 13.],
            [31., 20., 10.,  3.],
            [24.,  5.,  7., 29.],
            [ 4., 30., 21.,  6.],
            [26., 17., 23., 25.]])

    stored_linear_cka_res = 0.2815
    stored_rbf_cka_res = 0.3945

    params = {
        'stored_space1': stored_space1,
        'stored_space2': stored_space2,
        CKAMode.LINEAR: stored_linear_cka_res,
        CKAMode.RBF: stored_rbf_cka_res
    }

@pytest.fixture
def precomputed_cka():
    cka_instance = PrecomputedCKA()

    return cka_instance.params