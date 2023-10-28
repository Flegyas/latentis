import pytest
import torch

from latentis.space import LatentSpace
from latentis.utils import seed_everything


@pytest.fixture(scope="session")
def space1() -> LatentSpace:
    seed_everything(42)
    return LatentSpace(
        vectors=torch.randn(1000, 128),
        name="space1",
        properties={
            "label": torch.rand(1000) > 0.5,
        },
    )


@pytest.fixture(scope="session")
def space2() -> LatentSpace:
    seed_everything(0)
    return LatentSpace(
        vectors=torch.randn(53, 250),
        name="space2",
        properties={
            "label": torch.rand(53) > 0.5,
        },
    )
