from typing import Any, Mapping, Optional

import torch

from latentis.estimate.dim_matcher import DimMatcher
from latentis.estimate.estimator import Estimator


class LSTSQOrthoEstimator(Estimator):
    def __init__(self) -> None:
        super().__init__("lstsq_ortho", dim_matcher=None)

    def fit(self, source_data: torch.Tensor, target_data: torch.Tensor) -> Mapping[str, Any]:
        translation_matrix = torch.linalg.lstsq(source_data, target_data).solution
        translation_matrix = torch.as_tensor(translation_matrix, dtype=source_data.dtype, device=source_data.device)
        U, _, Vt = torch.svd(translation_matrix)
        translation_matrix = U @ Vt.T
        self.register_buffer("translation_matrix", translation_matrix)
        return {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.translation_matrix


class SVDEstimator(Estimator):
    def __init__(self, dim_matcher: Optional[DimMatcher]) -> None:
        super().__init__("svd", dim_matcher=dim_matcher)

    def fit(self, source_data: torch.Tensor, target_data: torch.Tensor) -> Mapping[str, Any]:
        self.dim_matcher.fit(source_data, target_data)
        dim_matcher_out = self.dim_matcher(source_data, target_data)
        source_data, target_data = dim_matcher_out["source"], dim_matcher_out["target"]

        assert source_data.size(1) == target_data.size(
            1
        ), f"Dimension mismatch between {source_data.size(1)} and {target_data.size(1)}. Forgot some padding/truncation transforms?"

        #  Compute the translation vector that aligns A to B using SVD.
        u, sigma, vt = torch.svd((target_data.T @ source_data).T)
        translation_matrix = u @ vt.T

        translation_matrix = torch.as_tensor(translation_matrix, dtype=source_data.dtype, device=source_data.device)
        self.register_buffer("translation_matrix", translation_matrix)

        sigma_rank = (~sigma.isclose(torch.zeros_like(sigma), atol=1e-1).bool()).sum().item()

        return {"sigma_rank": sigma_rank}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dim_matcher(source_x=x, target_x=None)["source"]
        x = x @ self.translation_matrix
        x = self.dim_matcher.reverse(source_x=None, target_x=x)["target"]
        return x
