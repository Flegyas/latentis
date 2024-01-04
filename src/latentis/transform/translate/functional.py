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
        dim_matcher.fit(x, y)
        x = dim_matcher.transform(x=x, y=None)
        y = dim_matcher.transform(x=None, y=y)

    state = svd_align_state(x, y)

    x = x @ state["matrix"]
    x = dim_matcher.inverse_transform(x=None, y=x)

    return x


def sgd_affine_align_state(
    x: torch.Tensor, y: torch.Tensor, num_steps: int = 300, lr: float = 1e-3, random_seed: int = None
) -> nn.Module:
    with torch.random.fork_rng():
        seed_everything(random_seed)
        with torch.enable_grad():
            translation = nn.Linear(x.size(1), y.size(1), device=x.device, dtype=x.dtype, bias=True)

            optimizer = torch.optim.Adam(translation.parameters(), lr=lr)

            for _ in range(num_steps):
                optimizer.zero_grad()
                loss = F.mse_loss(translation(x), y)
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
