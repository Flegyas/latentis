from typing import Any, Mapping, Optional, Sequence

from torch import nn

from latentis.estimate.estimator import Estimator
from latentis.space import LatentSpace
from latentis.transform import Transform
from latentis.types import Space
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

    def fit(self, source_data: Space, target_data: Space) -> Mapping[str, Any]:
        assert not self.fitted, "Translator is already fitted."
        self.fitted = True

        seed_everything(self.random_seed)

        source_vectors = source_data.vectors if isinstance(source_data, LatentSpace) else source_data
        target_vectors = target_data.vectors if isinstance(target_data, LatentSpace) else target_data

        self.register_buffer("source_vectors", source_vectors)
        self.register_buffer("target_vectors", target_vectors)

        transformed_source_data = source_vectors
        transformed_target_data = target_vectors

        for transform in self.source_transforms:
            transform.fit(transformed_source_data)
            transformed_source_data = transform(transformed_source_data)

        for transform in self.target_transforms:
            transform.fit(transformed_target_data)
            transformed_target_data = transform(transformed_target_data)

        self.translator_info = self.estimator.fit(
            source_data=transformed_source_data,
            target_data=transformed_target_data,
        )

        self.register_buffer("transformed_source_data", transformed_source_data)
        self.register_buffer("transformed_target_data", transformed_target_data)

        return self.translator_info

    def forward(self, x: Space, name: Optional[str] = None) -> LatentSpace:
        assert self.fitted, "Translator must be fitted before it can be used."

        source_x = x.vectors if isinstance(x, LatentSpace) else x

        for transform in self.source_transforms:
            source_x = transform(x=source_x)

        target_x = self.estimator(source_x)

        for transform in reversed(self.target_transforms):
            target_x = transform.reverse(x=target_x)

        if isinstance(x, LatentSpace):
            return LatentSpace.like(space=x, vectors=target_x, name=name if name is not None else x.name)
        else:
            return target_x
