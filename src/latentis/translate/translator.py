from typing import Any, Mapping, Optional, Sequence

from torch import nn

from latentis import transforms
from latentis.estimate.estimator import Estimator
from latentis.space import LatentSpace
from latentis.transforms import Transform
from latentis.utils import seed_everything


class LatentTranslator(nn.Module):
    def __init__(
        self,
        random_seed: int,
        estimator: Estimator,
        source_transforms: Optional[Sequence[Transform]] = None,
        target_transforms: Optional[Sequence[Transform]] = None,
        joint_transforms: Optional[Sequence[Transform]] = None,
    ) -> None:
        super().__init__()
        self.random_seed: int = random_seed
        self.estimator: Estimator = estimator
        self.fitted: bool = False

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
        self.joint_transforms: Sequence[Transform] = nn.ModuleList(
            joint_transforms
            if isinstance(joint_transforms, Sequence)
            else [transforms.ZeroPadding()]
            if joint_transforms is None
            else [joint_transforms]
        )

    def fit(self, source_data: LatentSpace, target_data: LatentSpace) -> Mapping[str, Any]:
        assert not self.fitted, "Translator is already fitted."
        self.fitted = True

        seed_everything(self.random_seed)

        source_data = source_data.vectors
        target_data = target_data.vectors

        self.register_buffer("source_data", source_data)
        self.register_buffer("target_data", target_data)

        transformed_source_data = source_data
        transformed_target_data = target_data

        for transform in self.source_transforms:
            transform.fit(transformed_source_data)
            transformed_source_data = transform(transformed_source_data)

        for transform in self.target_transforms:
            transform.fit(transformed_target_data)
            transformed_target_data = transform(transformed_target_data)

        for transform in self.joint_transforms:
            transform.fit(transformed_source_data, transformed_target_data)
            transformed_source_data, transformed_target_data = transform(
                transformed_source_data, transformed_target_data
            )

        self.translator_info = self.estimator.fit(
            source_data=transformed_source_data,
            target_data=transformed_target_data,
        )

        self.register_buffer("transformed_source_data", transformed_source_data)
        self.register_buffer("transformed_target_data", transformed_target_data)

        return self.translator_info

    def forward(self, x: LatentSpace, name: Optional[str] = None) -> LatentSpace:
        source_x = x.vectors
        for transform in self.source_transforms:
            source_x = transform(x=source_x)

        for transform in self.joint_transforms:
            source_x, _ = transform(source_x=source_x, target_x=None)

        target_x = self.estimator(source_x)

        for transform in reversed(self.joint_transforms):
            _, target_x = transform.reverse(source_x=None, target_x=target_x)

        for transform in reversed(self.target_transforms):
            target_x = transform.reverse(x=target_x)

        return LatentSpace.like(space=x, vectors=target_x, name=name)
