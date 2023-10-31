from typing import Tuple

import pytest
import torch

from latentis import LatentSpace
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


@pytest.fixture(params=SameShapeSpaces().instances)
def same_shape_spaces(request) -> Tuple[Space, Space]:
    return request.param
