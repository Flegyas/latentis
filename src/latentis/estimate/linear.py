from typing import Any, Mapping

import torch

from latentis.estimate.estimator import Estimator


class LSTSQEstimator(Estimator):
    def __init__(self) -> None:
        super().__init__("lstsq", dim_matcher=None)

    def fit(self, source_data: torch.Tensor, target_data: torch.Tensor) -> Mapping[str, Any]:
        translation_matrix = torch.linalg.lstsq(source_data, target_data).solution
        translation_matrix = torch.as_tensor(translation_matrix, dtype=source_data.dtype, device=source_data.device)
        self.register_buffer("translation_matrix", translation_matrix)

        return {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.translation_matrix
