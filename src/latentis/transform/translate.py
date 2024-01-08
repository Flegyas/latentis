from typing import Any, Mapping, Optional

import torch
import torch.nn.functional as F
from torch import nn

from latentis.transform import Identity, StandardScaling, Transform
from latentis.transform._abstract import Estimator
from latentis.transform.dim_matcher import DimMatcher, ZeroPadding
from latentis.utils import seed_everything


class Aligner(Estimator):
    def __init__(self, name: Optional[str] = None, dim_matcher: Optional[DimMatcher] = None) -> None:
        super().__init__(name=name)

        self.dim_matcher = dim_matcher


class Translator(Estimator):
    def __init__(
        self,
        aligner: Estimator,
        name: Optional[str] = None,
        x_transform: Optional[Transform] = None,
        y_transform: Optional[Transform] = None,
    ) -> None:
        super().__init__(name=name)
        self.x_transform = x_transform or Identity()
        self.y_transform = y_transform or Identity()
        self.aligner = aligner

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> Mapping[str, Any]:
        self.x_transform.fit(x)
        x = self.x_transform.transform(x)

        self.y_transform.fit(y)
        y = self.y_transform.transform(y)

        self.aligner.fit(x, y)

        return self

    def transform(self, x: torch.Tensor, y=None) -> torch.Tensor:
        x = self.x_transform.transform(x)
        x = self.aligner.transform(x)

        return self.y_transform.inverse_transform(x)

    def inverse_transform(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y = self.y_transform.transform(y)
        y = self.aligner.inverse_transform(y)

        return self.x_transform.inverse_transform(y)


class SVDAligner(Aligner):
    def __init__(self, dim_matcher: Optional[DimMatcher]) -> None:
        super().__init__("svd", dim_matcher=dim_matcher)

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> Mapping[str, Any]:
        self.dim_matcher.fit(x, y)
        dim_matcher_out = self.dim_matcher(x, y)
        x, y = dim_matcher_out["source"], dim_matcher_out["target"]

        assert x.size(1) == y.size(
            1
        ), f"Dimension mismatch between {x.size(1)} and {y.size(1)}. Forgot some padding/truncation transforms?"

        #  Compute the translation vector that aligns A to B using SVD.
        u, sigma, vt = torch.svd((y.T @ x).T)
        translation_matrix = u @ vt.T

        translation_matrix = torch.as_tensor(translation_matrix, dtype=x.dtype, device=x.device)
        self.register_buffer("translation_matrix", translation_matrix)

        sigma_rank = (~sigma.isclose(torch.zeros_like(sigma), atol=1e-1).bool()).sum().item()

        return {"sigma_rank": sigma_rank}

    def transform(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        x = self.dim_matcher(source_x=x, target_x=None)["source"]
        x = x @ self.translation_matrix
        x = self.dim_matcher.reverse(source_x=None, target_x=x)["target"]
        return x


class Procrustes(Translator):
    def __init__(self) -> None:
        super().__init__(
            name="procrustes",
            aligner=SVDAligner(dim_matcher=ZeroPadding()),
            x_transform=StandardScaling(),
            y_transform=StandardScaling(),
        )


class SGDAffineAligner(Aligner):
    def __init__(self, num_steps: int = 300, lr: float = 1e-3, random_seed: int = None) -> None:
        """Estimator that uses SGD to estimate an affine transformation between two spaces.

        Args:
            num_steps (int): Number of optimization steps to take.
            lr (float): Learning rate for the optimizer.
            random_seed (int): Random seed for reproducibility.
        """
        super().__init__("sgd_affine", dim_matcher=None)
        self.num_steps: int = num_steps
        self.lr: float = lr
        self.random_seed: int = random_seed

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> Mapping[str, Any]:
        with torch.random.fork_rng():
            seed_everything(self.random_seed)
            with torch.enable_grad():
                translation = nn.Linear(x.size(1), y.size(1), device=x.device, dtype=x.dtype, bias=True)

                optimizer = torch.optim.Adam(translation.parameters(), lr=self.lr)

                for _ in range(self.num_steps):
                    optimizer.zero_grad()
                    loss = F.mse_loss(translation(x), y)
                    loss.backward()
                    optimizer.step()

                self.translation = translation

    def transform(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        return self.translation(x)


class LSTSQAligner(Aligner):
    def __init__(self) -> None:
        super().__init__("lstsq", dim_matcher=None)

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> Mapping[str, Any]:
        translation_matrix = torch.linalg.lstsq(x, y).solution
        translation_matrix = torch.as_tensor(translation_matrix, dtype=x.dtype, device=x.device)
        self.register_buffer("translation_matrix", translation_matrix)

        return {}

    def transform(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        return x @ self.translation_matrix


class LSTSQOrthoAligner(Aligner):
    def __init__(self) -> None:
        super().__init__("lstsq_ortho", dim_matcher=None)

    def fit(self, source_data: torch.Tensor, target_data: torch.Tensor) -> Mapping[str, Any]:
        translation_matrix = torch.linalg.lstsq(source_data, target_data).solution
        translation_matrix = torch.as_tensor(translation_matrix, dtype=source_data.dtype, device=source_data.device)
        U, _, Vt = torch.svd(translation_matrix)
        translation_matrix = U @ Vt.T
        self.register_buffer("translation_matrix", translation_matrix)
        return {}

    def transform(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        return x @ self.translation_matrix
