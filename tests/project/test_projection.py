import functools

import pytest
import torch
from scipy.stats import ortho_group

from tests.project.conftest import LATENT_DIM

from latentis import LatentSpace, transform
from latentis.project import (
    RelativeProjector,
    angular_proj,
    change_of_basis_proj,
    cosine_proj,
    euclidean_proj,
    l1_proj,
    lp_proj,
    pointwise_wrapper,
)
from latentis.types import ProjectionFunc, Space
from latentis.utils import seed_everything


def random_ortho_matrix(random_seed: int) -> torch.Tensor:
    return torch.as_tensor(ortho_group.rvs(LATENT_DIM, random_state=random_seed), dtype=torch.double)


def random_perm_matrix(random_seed: int) -> torch.Tensor:
    seed_everything(random_seed)
    return torch.as_tensor(torch.randperm(LATENT_DIM), dtype=torch.long)


def random_isotropic_scaling(random_seed: int) -> torch.Tensor:
    seed_everything(random_seed)
    return torch.randint(1, 100, (1,), dtype=torch.double)


@pytest.mark.parametrize(
    "projection_fn,unsqueeze",
    [
        (cosine_proj, True),
        (angular_proj, True),
        (euclidean_proj, True),
        (l1_proj, True),
        (functools.partial(lp_proj, p=10), True),
    ],
)
def test_pointwise_wrapper(projection_fn, unsqueeze: bool, tensor_space_with_ref):
    x, anchors = tensor_space_with_ref

    vectorized = projection_fn(x, anchors)
    pointwise = pointwise_wrapper(projection_fn, unsqueeze=unsqueeze)(x, anchors)

    assert torch.allclose(vectorized, pointwise)


@pytest.mark.parametrize(
    "projector,invariance,invariant",
    [
        (
            RelativeProjector(projection_fn=angular_proj),
            lambda x: x @ random_ortho_matrix(random_seed=42),
            True,
        ),
        (
            RelativeProjector(projection_fn=angular_proj),
            lambda x: x @ random_ortho_matrix(random_seed=42) + 100,
            False,
        ),
        (
            RelativeProjector(projection_fn=angular_proj, abs_transforms=[transform.Centering()]),
            lambda x: x @ random_ortho_matrix(random_seed=42) + 100,
            True,
        ),
        (
            RelativeProjector(projection_fn=cosine_proj),
            lambda x: x @ random_ortho_matrix(random_seed=42),
            True,
        ),
        (
            RelativeProjector(projection_fn=cosine_proj, abs_transforms=[transform.Centering()]),
            lambda x: (x + 20) @ random_ortho_matrix(42),
            True,
        ),
        (
            RelativeProjector(projection_fn=cosine_proj, abs_transforms=[transform.Centering()]),
            lambda x: (x) @ random_ortho_matrix(42) + 20,
            True,
        ),
        (
            RelativeProjector(projection_fn=cosine_proj),
            lambda x: (x) @ random_ortho_matrix(42) * 100,
            True,
        ),
        (
            RelativeProjector(projection_fn=cosine_proj),
            lambda x: (x) @ random_ortho_matrix(42) + 20,
            False,
        ),
        (
            RelativeProjector(projection_fn=cosine_proj),
            lambda x: x @ random_ortho_matrix(random_seed=42),
            True,
        ),
        (
            RelativeProjector(projection_fn=euclidean_proj),
            lambda x: (x) @ random_ortho_matrix(42) + 100,
            True,
        ),
        (
            RelativeProjector(projection_fn=l1_proj),
            lambda x: (x) @ random_ortho_matrix(42) + 100,
            False,
        ),
        (
            RelativeProjector(projection_fn=l1_proj),
            lambda x: x[:, random_perm_matrix(42)] + 100,
            True,
        ),
        (
            RelativeProjector(projection_fn=l1_proj),
            lambda x: (x + 100)[:, random_perm_matrix(42)],
            True,
        ),
        (
            RelativeProjector(projection_fn=cosine_proj),
            lambda x: (x + 100) * random_isotropic_scaling(42),
            False,
        ),
        (
            RelativeProjector(projection_fn=cosine_proj, abs_transforms=transform.Centering()),
            lambda x: (x + 100) * random_isotropic_scaling(42),
            True,
        ),
        (
            RelativeProjector(projection_fn=cosine_proj, abs_transforms=transform.Centering()),
            lambda x: (x + 100) * random_isotropic_scaling(42) + 100,
            True,
        ),
        (
            RelativeProjector(projection_fn=change_of_basis_proj, abs_transforms=transform.Centering()),
            lambda x: (x + 100) * random_isotropic_scaling(42),
            True,
        ),
        (
            RelativeProjector(projection_fn=change_of_basis_proj),
            lambda x: (x) * random_isotropic_scaling(42),
            True,
        ),
        (
            RelativeProjector(projection_fn=change_of_basis_proj),
            lambda x: (x) * random_isotropic_scaling(42) + 53,
            False,
        ),
    ],
)
def test_invariances(projector: ProjectionFunc, x_latents: Space, anchor_latents, invariance, invariant):
    y = invariance(x_latents if isinstance(x_latents, torch.Tensor) else x_latents.vectors)
    y_anchors = invariance(anchor_latents if isinstance(anchor_latents, torch.Tensor) else anchor_latents.vectors)

    if isinstance(x_latents, LatentSpace):
        y = LatentSpace.like(x_latents, vectors=y)

    if isinstance(anchor_latents, LatentSpace):
        y_anchors = LatentSpace.like(anchor_latents, vectors=y_anchors)

    x_projected = projector(x_latents, anchor_latents)
    y_projected = projector(y, y_anchors)

    assert not invariant or torch.allclose(x_projected, y_projected)

    if isinstance(x_latents, LatentSpace):
        space_relative = x_latents.to_relative(projector=projector, anchors=anchor_latents)
        assert torch.allclose(space_relative.vectors, x_projected)
