from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from latentis.estimate.estimator import Estimator


class AffineTranslator(Estimator):
    def __init__(self) -> None:
        super().__init__("affine")

    def fit(self, source_data: torch.Tensor, target_data: torch.Tensor) -> Mapping[str, Any]:
        with torch.enable_grad():
            translation = nn.Linear(source_data.size(1), target_data.size(1), device=source_data.device)
            optimizer = torch.optim.Adam(translation.parameters(), lr=1e-3)

            for _ in range(100):
                optimizer.zero_grad()
                loss = F.mse_loss(translation(source_data), target_data)
                loss.backward()
                optimizer.step()
            self.translation = translation.cpu()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.translation(x)
