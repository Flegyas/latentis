from __future__ import annotations

import copy
from enum import auto
from typing import TYPE_CHECKING, Any, Callable, Dict, Mapping, Optional, Sequence, Union

from latentis.measure import Metric, MetricFn

if TYPE_CHECKING:
    from latentis.sample import Sampler
    from latentis.project import RelativeProjector
    from latentis.types import Space
    from latentis.translate import LatentTranslator

import torch
from torch.utils.data import Dataset as TorchDataset

try:
    # be ready for 3.10 when it drops
    from enum import StrEnum
except ImportError:
    from backports.strenum import StrEnum


class SpaceProperty(StrEnum):
    SAMPLING_IDS = auto()


class LatentSpace(TorchDataset):
    def __init__(
        self,
        vectors: torch.Tensor,
        name: str,
        features: Optional[Dict[str, Sequence[Any]]] = None,
    ):
        assert isinstance(vectors, torch.Tensor)
        assert features is None or all(len(values) == vectors.size(0) for values in features.values())
        if features is None:
            features = {}
        self.vectors: torch.Tensor = vectors
        self.name: str = name
        self.features = features

    @classmethod
    def like(
        cls,
        space: LatentSpace,
        name: Optional[str] = None,
        vectors: Optional[torch.Tensor] = None,
        features: Optional[Dict[str, Sequence[Any]]] = None,
        deepcopy: bool = False,
    ):
        """Create a new space with the arguments not provided taken from the given space.

        There is no copy of the vectors, so changes to the vectors of the new space will also affect the vectors of the given space.

        Args:
            space (LatentSpace): The space to copy.
            name (Optional[str]): The name of the new space.
            vectors (Optional[torch.Tensor], optional): The vectors of the new space.
            features (Optional[Dict[str, Sequence[Any]]], optional): The features of the new space.

        Returns:
            LatentSpace: The new space, with the arguments not provided taken from the given space.
        """
        if name is None:
            name = space.name
        if vectors is None:
            vectors = space.vectors if not deepcopy else copy.deepcopy(space.vectors)
        if features is None:
            features = space.features if not deepcopy else copy.deepcopy(space.features)
        # TODO: test deepcopy
        return LatentSpace(name=name, vectors=vectors, features=features)

    @property
    def shape(self) -> torch.Size:
        return self.vectors.shape

    def __getitem__(self, index: int) -> Mapping[str, torch.Tensor]:
        return {"x": self.vectors[index], **{key: values[index] for key, values in self.features.items()}}

    def __len__(self) -> int:
        return self.vectors.size(0)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, vectors={self.vectors.shape}, features={self.features.keys()})"

    def __eq__(self, __value: object) -> bool:
        return (
            self.name == __value.name
            and self.features == __value.features
            and torch.allclose(self.vectors, __value.vectors)
        )

    def sample(self, sampler: Sampler, n: int) -> "LatentSpace":
        """Sample n vectors from this space using the given sampler.

        Args:
            sampler (Sampler): The sampler to use.
            n (int): The number of vectors to sample.

        Returns:
            LatentSpace: The sampled space.

        """
        return sampler(self, n=n)

    def compare(self, *others: Space, metrics: Mapping[str, Union[Metric, Callable[[Space, Space], torch.Tensor]]]):
        """Compare this space with another space using the given metrics.

        Args:
            other (Space): The space to compare with.
            metrics (Mapping[str, Metric]): The metrics to use.

        Returns:
            Dict[str, Any]: The results of the comparison.
        """
        metrics = {
            key: metric if isinstance(metric, Metric) else MetricFn(key, metric) for key, metric in (metrics.items())
        }
        metrics_results = {metric_name: metric(self, *others) for metric_name, metric in metrics.items()}
        return metrics_results

    def translate(
        self,
        translator: LatentTranslator,
    ):
        return translator(x=self)

    # @lru_cache
    # def to_faiss(self, normalize: bool, keys: Sequence[str]) -> FaissIndex:
    #     index: FaissIndex = FaissIndex(d=self.vectors.size(1))

    #     index.add_vectors(
    #         embeddings=list(zip(keys, self.vectors.cpu().numpy())),
    #         normalize=normalize,
    #     )

    #     return index
    def to_relative(
        self,
        projector: RelativeProjector,
        anchors: Space,
        # sampler: Optional[Sampler] = None,
        # n: Optional[int] = None,
        # random_seed: int = None,
    ) -> "LatentSpace":
        relative_vectors = projector(
            x=self.vectors,
            anchors=anchors,
        )

        return RelativeSpace.like(
            space=self,
            name=f"{self.name}/{projector.name}",
            vectors=relative_vectors,
            anchors=anchors,
        )


class RelativeSpace(LatentSpace):
    @classmethod
    def like(
        cls,
        space: RelativeSpace,
        name: Optional[str] = None,
        vectors: Optional[torch.Tensor] = None,
        features: Optional[Dict[str, Sequence[Any]]] = None,
        anchors: Optional[Space] = None,
        deepcopy: bool = False,
    ):
        if name is None:
            name = space.name

        if vectors is None:
            vectors = space.vectors if not deepcopy else space.vectors.clone()

        if features is None:
            features = (
                space.features if not deepcopy else {key: values.clone() for key, values in space.features.items()}
            )

        if anchors is None:
            anchors = space.anchors if not deepcopy else copy.deepcopy(space.anchors)

        return cls(name=name, vectors=vectors, features=features, anchors=anchors)

    def __init__(
        self,
        vectors: torch.Tensor,
        name: str,
        features: Dict[str, Sequence[Any]] | None = None,
        anchors: Space | None = None,
    ):
        super().__init__(vectors=vectors, name=name, features=features)
        self.anchors: Space = anchors
