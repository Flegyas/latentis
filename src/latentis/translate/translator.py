from typing import Any, Mapping, Optional, Sequence

import torch
from torch import nn

from latentis.estimate.estimator import Estimator
from latentis.transforms import Transform
from latentis.utils import seed_everything


class LatentTranslator(nn.Module):
    def __init__(
        self,
        random_seed: int,
        estimator: Estimator,
        source_transforms: Optional[Sequence[Transform]] = None,
        target_transforms: Optional[Sequence[Transform]] = None,
    ) -> None:
        super().__init__()
        self.random_seed: int = random_seed
        self.estimator: Estimator = estimator

        self.source_transforms: Sequence[Transform] = nn.ModuleList(
            source_transforms
            if isinstance(source_transforms, Sequence)
            else []
            if source_transforms is None
            else [source_transforms]
        )
        self.target_transforms: Sequence[Transform] = nn.ModuleList(
            target_transforms
            if isinstance(target_transforms, Sequence)
            else []
            if target_transforms is None
            else [target_transforms]
        )

    def fit(self, source_data: torch.Tensor, target_data: torch.Tensor) -> Mapping[str, Any]:
        seed_everything(self.random_seed)

        # for transform in self.transforms:
        #     transform.fit(source_data=source_data, target_data=target_data)

        # for transform in self.transforms:
        #     transform.fit(source_data=source_data, target_data=target_data)
        self.register_buffer("source_data", source_data)
        self.register_buffer("target_data", target_data)

        transformed_source_data = source_data
        transformed_target_data = target_data

        for transform in self.source_transforms:
            transform.fit(transformed_source_data)
            transformed_source_data = transform(transformed_source_data)

        for transform in self.target_trasnforms:
            transform.fit(transformed_target_data)
            transformed_target_data = transform(transformed_target_data)

        translator_info = self.estimator.fit(source_data=transformed_source_data, target_data=transformed_target_data)

        self.register_buffer("transformed_source_data", transformed_source_data)
        self.register_buffer("transformed_target_data", transformed_target_data)

        return translator_info

    def forward(self, x: torch.Tensor, compute_info: bool = True) -> torch.Tensor:
        source_x = x
        for transform in self.source_transforms:
            source_x = transform(x=source_x)

        target_x = self.estimator(source_x)

        for transform in reversed(self.target_transforms):
            target_x = transform.reverse(x=target_x)

        return {"source": source_x, "target": target_x, "info": {}}
