from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from latentis.space import LatentSpace
from latentis.utils import seed_everything

if TYPE_CHECKING:
    from latentis.types import Space
BATCH_DIM = 4
LATENT_DIM = 8
N_CLASSES = 10
NUM_ANCHORS = 20


class Anchor1Params(object):
    seed_everything(42)
    instances = [
        LatentSpace(
            vector_source=torch.randn(NUM_ANCHORS, LATENT_DIM, dtype=torch.double),
            name="space1",
        ),
        torch.randn(NUM_ANCHORS, LATENT_DIM, dtype=torch.double),
    ]


@pytest.fixture(params=Anchor1Params().instances, scope="session")
def x_anchors(request) -> Space:
    return request.param


# @pytest.fixture
# def anchor_targets() -> torch.Tensor:
#     return torch.cat(
#         (
#             torch.arange(N_CLASSES, dtype=torch.double),
#             torch.randint(N_CLASSES, size=(NUM_ANCHORS - N_CLASSES,), dtype=torch.double),
#         )
#     )


class X1Params(object):
    seed_everything(42)
    instances = [
        LatentSpace(
            vector_source=torch.randn(BATCH_DIM, LATENT_DIM, dtype=torch.double),
            name="space1",
        ),
        torch.randn(BATCH_DIM, LATENT_DIM, dtype=torch.double),
    ]


@pytest.fixture(params=X1Params().instances, scope="session")
def x() -> torch.Tensor:
    return torch.randn(BATCH_DIM, LATENT_DIM, dtype=torch.double)
