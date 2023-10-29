from typing import Mapping, Optional

import torch

from latentis.transforms.abstract import Joint


class ZeroPadding(Joint):
    def __init__(self) -> None:
        super().__init__(name="zero_padding")

    def _fit(self, source_data: torch.Tensor, target_data: torch.Tensor, *args, **kwargs) -> Mapping[str, torch.Tensor]:
        source_pad = target_data.size(1) - source_data.size(1)
        target_pad = source_data.size(1) - target_data.size(1)
        transform_source = source_pad > 0
        transform_target = target_pad > 0
        return {
            "source_pad": torch.as_tensor(max(0, source_pad)),
            "target_pad": torch.as_tensor(max(0, target_pad)),
            "transform_source": torch.as_tensor(transform_source),
            "transform_target": torch.as_tensor(transform_target),
        }

    def forward(
        self, source_x: Optional[torch.Tensor], target_x: Optional[torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        if self.transform_source and source_x is not None:
            assert source_x.ndim == 2, "The source tensor must be 2D."
            source_x = torch.nn.functional.pad(
                source_x,
                (0, self.source_pad),
                mode="constant",
                value=0,
            )
        if self.transform_target and target_x is not None:
            assert target_x.ndim == 2, "The target tensor must be 2D."
            target_x = torch.nn.functional.pad(
                target_x,
                (0, self.target_pad),
                mode="constant",
                value=0,
            )
        return (source_x, target_x)

    def reverse(
        self, source_x: Optional[torch.Tensor], target_x: Optional[torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        assert self.fitted, "The transform must be fit first."
        if self.transform_source and source_x is not None:
            source_x = source_x[..., : -self.source_pad]
        if self.transform_target and target_x is not None:
            target_x = target_x[..., : -self.target_pad]
        return (source_x, target_x)


# class PCATruncation(Joint):
#     def __init__(self, n_components: int) -> None:
#         super().__init__(name="pca_truncation")
#         self.n_components = n_components

#     def _fit(self, data: torch.Tensor, *args, **kwargs) -> Mapping[str, torch.Tensor]:
#         raise NotImplementedError

#     def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
#         raise NotImplementedError

#     def reverse(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
#         raise NotImplementedError
