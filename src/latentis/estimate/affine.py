from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from latentis.estimate.estimator import Estimator


class SGDAffineTranslator(Estimator):
    def __init__(self, num_steps: int = 300, lr: float = 1e-3) -> None:
        """Estimator that uses SGD to estimate an affine transformation between two spaces.

        Args:
            num_steps (int): Number of optimization steps to take.
            lr (float): Learning rate for the optimizer.
        """
        super().__init__("sgd_affine", dim_matcher=None)
        self.num_steps: int = num_steps
        self.lr: float = lr

    def fit(self, source_data: torch.Tensor, target_data: torch.Tensor) -> Mapping[str, Any]:
        with torch.enable_grad():
            translation = nn.Linear(
                source_data.size(1), target_data.size(1), device=source_data.device, dtype=source_data.dtype, bias=True
            )
            optimizer = torch.optim.Adam(translation.parameters(), lr=self.lr)

            for _ in range(self.num_steps):
                optimizer.zero_grad()
                loss = F.mse_loss(translation(source_data), target_data)
                loss.backward()
                optimizer.step()

            self.translation = translation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.translation(x)
