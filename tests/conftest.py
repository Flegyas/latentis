import pytest
import torch

from latentis.space import LatentSpace
from latentis.utils import seed_everything


@pytest.fixture(scope="session")
def space1() -> LatentSpace:
    seed_everything(42)
    return LatentSpace(
        vectors=torch.randn(1000, 128, dtype=torch.double),
        name="space1",
        properties={
            "label": torch.rand(1000) > 0.5,
        },
    )


@pytest.fixture(scope="session")
def space2() -> LatentSpace:
    seed_everything(0)
    return LatentSpace(
        vectors=torch.randn(53, 250, dtype=torch.double),
        name="space2",
        properties={
            "label": torch.rand(53) > 0.5,
        },
    )


class ParallelSpaces(object):
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
            ((50, 250), (50, 300)),
            ((50, 300), (50, 250)),
            ((1000, 100), (1000, 500)),
            ((1000, 500), (1000, 100)),
        ]
    ]


@pytest.fixture(params=ParallelSpaces().instances)
def parallel_spaces(request):
    return request.param
