# from typing import Any, Mapping

# import torch

# from latentis.estimate.estimator import Estimator


# class LSTSQTranslator(Estimator):
#     def __init__(self) -> None:
#         super().__init__("lstsq")
#         self.translation_matrix = None

#     def fit(self, source_data: torch.Tensor, target_data: torch.Tensor) -> Mapping[str, Any]:
#         translation_matrix = torch.linalg.lstsq(source_data, target_data).solution
#         self.translation_matrix = torch.as_tensor(translation_matrix, dtype=torch.float32, device=source_data.device)

#         return {}

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return x @ self.translation_matrix
