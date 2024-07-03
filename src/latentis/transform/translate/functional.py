from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from latentis.transform.dim_matcher import DimMatcher
from latentis.utils import seed_everything


def svd_align_state(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.size(1) == y.size(
        1
    ), f"Dimension mismatch between {x.size(1)} and {y.size(1)}. Forgot some padding/truncation transforms?"

    #  Compute the translation vector that aligns A to B using SVD.
    u, sigma, vt = torch.svd((y.T @ x).T)
    translation_matrix = u @ vt.T

    translation_matrix = torch.as_tensor(translation_matrix, dtype=x.dtype, device=x.device)
    return dict(matrix=translation_matrix)


def svd_align(x: torch.Tensor, y: torch.Tensor, dim_matcher: Optional[DimMatcher] = None) -> torch.Tensor:
    if dim_matcher is not None:
        x, y = dim_matcher.fit_transform(x=x, y=y)

    state = svd_align_state(x, y)

    x = x @ state["matrix"]

    if dim_matcher is not None:
        _, x = dim_matcher.inverse_transform(x=None, y=x)

    return x


@torch.enable_grad()
def sgd_affine_align_state(
    x: torch.Tensor, y: torch.Tensor, num_steps: int = 300, lr: float = 1e-3, random_seed: int = None
) -> nn.Module:
    device = None if x.device.type == "cpu" else x.device.index
    with torch.random.fork_rng(devices=[device]):
        seed_everything(random_seed)
        translation = nn.Linear(x.size(1), y.size(1), device=x.device, dtype=x.dtype, bias=True)

        x = x.detach()
        y = y.detach()
        optimizer = torch.optim.Adam(translation.parameters(), lr=lr)
        for _ in range(num_steps):
            optimizer.zero_grad()
            pred = translation(x)
            loss = F.mse_loss(pred, y)
            loss.backward()
            optimizer.step()

        return dict(translation=translation)


def sgd_affine_align(
    x: torch.Tensor, y: torch.Tensor, *, num_steps: int = 300, lr: float = 1e-3, random_seed: int = None
) -> nn.Module:
    state = sgd_affine_align_state(x, y, num_steps=num_steps, lr=lr, random_seed=random_seed)
    return state["translation"](x)


def lstsq_align_state(x: torch.Tensor, y: torch.Tensor):
    translation_matrix = torch.linalg.lstsq(x, y).solution
    translation_matrix = torch.as_tensor(translation_matrix, dtype=x.dtype, device=x.device)
    return dict(matrix=translation_matrix)


def lstsq_align(x: torch.Tensor, y: torch.Tensor):
    state = lstsq_align_state(x, y)
    return x @ state["matrix"]


def lstsq_ortho_align_state(x: torch.Tensor, y: torch.Tensor):
    translation_matrix = torch.linalg.lstsq(x, y).solution
    translation_matrix = torch.as_tensor(translation_matrix, dtype=x.dtype, device=x.device)
    U, _, Vt = torch.svd(translation_matrix)
    translation_matrix = U @ Vt.T
    return dict(matrix=translation_matrix)


def lstsq_ortho_align(x: torch.Tensor, y: torch.Tensor):
    state = lstsq_ortho_align_state(x, y)
    return x @ state["matrix"]
