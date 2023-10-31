import pytest
import torch

from latentis import LatentSpace
from latentis.types import Space
from latentis.utils import seed_everything

BATCH_DIM = 4
LATENT_DIM = 8
N_CLASSES = 10
NUM_ANCHORS = 20


class Anchor1Params(object):
    seed_everything(42)
    instances = [
        LatentSpace(
            vectors=torch.randn(NUM_ANCHORS, LATENT_DIM, dtype=torch.double),
            name="space1",
            features={
                "label": torch.rand(NUM_ANCHORS) > 0.5,
            },
        ),
        torch.randn(NUM_ANCHORS, LATENT_DIM, dtype=torch.double),
    ]


@pytest.fixture(params=Anchor1Params().instances, scope="session")
def anchor_latents(request) -> Space:
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
            vectors=torch.randn(BATCH_DIM, LATENT_DIM, dtype=torch.double),
            name="space1",
            features={
                "label": torch.rand(BATCH_DIM) > 0.5,
            },
        ),
        torch.randn(BATCH_DIM, LATENT_DIM, dtype=torch.double),
    ]


@pytest.fixture(params=X1Params().instances, scope="session")
def x_latents() -> torch.Tensor:
    return torch.randn(BATCH_DIM, LATENT_DIM, dtype=torch.double)
