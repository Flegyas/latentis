from typing import Any, Mapping

import torch

from latentis.estimate.estimator import Estimator


class LSTSQOrthoTranslator(Estimator):
    def __init__(self) -> None:
        super().__init__("lstsq_ortho")

    def fit(self, source_data: torch.Tensor, target_data: torch.Tensor) -> Mapping[str, Any]:
        translation_matrix = torch.linalg.lstsq(source_data, target_data).solution
        U, _, Vt = torch.svd(translation_matrix)
        self.translation_matrix = U @ Vt.T

        return {}


class SVDEstimator(Estimator):
    def __init__(self) -> None:
        super().__init__("svd")
        # self.register_buffer("translation_matrix", None)

    def fit(self, source_data: torch.Tensor, target_data: torch.Tensor) -> Mapping[str, Any]:
        #  Compute the translation vector that aligns A to B using SVD.
        assert source_data.size(1) == target_data.size(
            1
        ), f"Dimension mismatch between {source_data.size(1)} and {target_data.size(1)}. Forgot some padding/truncation transforms?"
        u, sigma, vt = torch.svd((target_data.T @ source_data).T)
        translation_matrix = u @ vt.T

        translation_matrix = torch.as_tensor(translation_matrix, dtype=source_data.dtype, device=source_data.device)
        self.register_buffer("translation_matrix", translation_matrix)

        sigma_rank = (~sigma.isclose(torch.zeros_like(sigma), atol=1e-1).bool()).sum().item()

        return {"sigma_rank": sigma_rank}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.translation_matrix
