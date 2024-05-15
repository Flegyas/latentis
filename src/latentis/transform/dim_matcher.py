from typing import Dict, Mapping, Optional

import torch

from latentis.transform import Estimator


class DimMatcher(Estimator):
    def __init__(self, name: str, invertible: bool) -> None:
        super().__init__(name=name, invertible=invertible)


class ZeroPadding(DimMatcher):
    def __init__(self) -> None:
        super().__init__(name="zero_padding", invertible=True)

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> Mapping[str, torch.Tensor]:
        x_pad = y.size(1) - x.size(1)
        y_pad = x.size(1) - y.size(1)
        transform_x = x_pad > 0
        transform_y = y_pad > 0

        self._register_state(
            {
                "x_pad": torch.as_tensor(max(0, x_pad), dtype=torch.long, device=x.device),
                "y_pad": torch.as_tensor(max(0, y_pad), dtype=torch.long, device=x.device),
                "transform_x": torch.as_tensor(transform_x, dtype=torch.bool, device=x.device),
                "transform_y": torch.as_tensor(transform_y, dtype=torch.bool, device=x.device),
            }
        )

        return self

    def transform(
        self,
        x: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if x is not None and self.get_state("transform_x"):
            assert x.ndim == 2, "The source tensor must be 2D."
            x = torch.nn.functional.pad(
                x,
                (0, self.get_state("x_pad")),
                mode="constant",
                value=0,
            )

        if y is not None and self.get_state("transform_y"):
            assert y.ndim == 2, "The target tensor must be 2D."
            y = torch.nn.functional.pad(
                y,
                (0, self.get_state("y_pad")),
                mode="constant",
                value=0,
            )

        return x, y

    def inverse_transform(
        self,
        x: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if x is not None and self.get_state("transform_x"):
            x = x[..., : -self.get_state("x_pad")]

        if y is not None and self.get_state("transform_y"):
            y = y[..., : -self.get_state("y_pad")]

        return x, y


# class PCATruncation(DimMatcher):
#     def __init__(self, n_components: int) -> None:
#         super().__init__(name="pca_truncation")
#         self.n_components = n_components

#     def _fit(self, data: torch.Tensor, *args, **kwargs) -> Mapping[str, torch.Tensor]:
#         raise NotImplementedError

#     def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
#         raise NotImplementedError

#     def reverse(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
#         raise NotImplementedError
