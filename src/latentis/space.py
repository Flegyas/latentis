from __future__ import annotations

import copy
import logging
from abc import abstractmethod
from enum import auto
from typing import TYPE_CHECKING, Any, Callable, Dict, Mapping, Optional, Sequence, Union

from latentis.measure import MetricFn
from latentis.search import SearchIndex, SearchMetric

if TYPE_CHECKING:
    from latentis.sample import Sampler
    from latentis.project import RelativeProjection
    from latentis.types import Space
    from latentis.translate import LatentTranslator

import torch
from torch.utils.data import Dataset as TorchDataset

try:
    # be ready for 3.10 when it drops
    from enum import StrEnum
except ImportError:
    from backports.strenum import StrEnum

pylogger = logging.getLogger(__name__)


class SpaceProperty(StrEnum):
    SAMPLING_IDS = auto()


class VectorSource:
    @abstractmethod
    def shape(self) -> torch.Size:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: int) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def as_tensor(self) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def __eq__(self, __value: VectorSource) -> bool:
        raise NotImplementedError


class TorchVectorSource(VectorSource):
    def __init__(self, vectors: torch.Tensor):
        self._vectors = vectors

    def shape(self) -> torch.Size:
        return self._vectors.shape

    def __getitem__(self, index: int) -> torch.Tensor:
        return self._vectors[index]

    def __len__(self) -> int:
        return self._vectors.size(0)

    def __eq__(self, __value: TorchVectorSource) -> bool:
        assert isinstance(__value, TorchVectorSource), f"Expected {TorchVectorSource}, got {type(__value)}"
        return torch.allclose(self._vectors, __value.as_tensor())

    def as_tensor(self) -> torch.Tensor:
        return self._vectors


class LatentSpace(TorchDataset):
    def __init__(
        self,
        name: str,
        vector_source: Union[torch.Tensor, VectorSource],
        features: Optional[Dict[str, Sequence[Any]]] = None,
    ):
        assert isinstance(vector_source, torch.Tensor) or isinstance(
            vector_source, VectorSource
        ), f"Expected {torch.Tensor} or {VectorSource}, got {type(vector_source)}"
        assert features is None or all(len(values) == vector_source.size(0) for values in features.values())
        self.name: str = name
        if features is None:
            features = {}
        self._vector_source: torch.Tensor = (
            TorchVectorSource(vector_source) if torch.is_tensor(vector_source) else vector_source
        )
        self.features = features

    @property
    def vectors(self) -> torch.Tensor:
        return self._vector_source.as_tensor()

    def to_memory(self) -> "LatentSpace":
        if isinstance(self._vector_source, torch.Tensor):
            pylogger.info("Already in memory, skipping.")
            return self

        return LatentSpace.like(
            space=self,
            vector_source=self.as_tensor,
        )

    @classmethod
    def like(
        cls,
        space: LatentSpace,
        name: Optional[str] = None,
        vector_source: Optional[VectorSource] = None,
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
        if vector_source is None:
            vector_source = space.vector_source if not deepcopy else copy.deepcopy(space.vector_source)
        if features is None:
            features = space.features if not deepcopy else copy.deepcopy(space.features)
        # TODO: test deepcopy
        return LatentSpace(name=name, vector_source=vector_source, features=features)

    @property
    def shape(self) -> torch.Size:
        return self.vectors.shape

    def __getitem__(self, index: int) -> Mapping[str, torch.Tensor]:
        return {"x": self._vector_source[index], **{key: values[index] for key, values in self.features.items()}}

    def __len__(self) -> int:
        return len(self._vector_source)

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

    def compare(
        self, *others: Space, metrics: Mapping[str, Union[SearchMetric, Callable[[Space, Space], torch.Tensor]]]
    ):
        """Compare this space with another space using the given metrics.

        Args:
            other (Space): The space to compare with.
            metrics (Mapping[str, Metric]): The metrics to use.

        Returns:
            Dict[str, Any]: The results of the comparison.
        """
        metrics = {
            key: metric if isinstance(metric, SearchMetric) else MetricFn(key, metric)
            for key, metric in (metrics.items())
        }
        metrics_results = {metric_name: metric(self, *others) for metric_name, metric in metrics.items()}
        return metrics_results

    def translate(
        self,
        translator: LatentTranslator,
    ):
        return translator(x=self)

    def to_index(
        self,
        metric_fn: SearchMetric,
        keys: Optional[Sequence[str]] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> SearchIndex:
        index: SearchIndex = SearchIndex.create(
            metric_fn=metric_fn,
            num_dimensions=self.vectors.size(dim=1),
            name=self.name,
            transform=transform,
        )

        index.add_items(
            vectors=self.vectors.cpu(),
            keys=keys,
        )

        return index

    def to_relative(
        self,
        projection: RelativeProjection,
        anchors: Space,
        # sampler: Optional[Sampler] = None,
        # n: Optional[int] = None,
        # random_seed: int = None,
    ) -> "LatentSpace":
        relative_vectors = projection(
            x=self.vectors,
            anchors=anchors,
        )

        return RelativeSpace.like(
            space=self,
            name=f"{self.name}/{projection.name}",
            vector_source=relative_vectors,
            anchors=anchors,
        )


class RelativeSpace(LatentSpace):
    @classmethod
    def like(
        cls,
        space: RelativeSpace,
        name: Optional[str] = None,
        vector_source: Optional[Union[torch.Tensor, VectorSource]] = None,
        features: Optional[Dict[str, Sequence[Any]]] = None,
        anchors: Optional[Space] = None,
        deepcopy: bool = False,
    ):
        if name is None:
            name = space.name

        if vector_source is None:
            vector_source = space._vector_source if not deepcopy else space._vector_source.clone()

        if features is None:
            features = (
                space.features if not deepcopy else {key: values.clone() for key, values in space.features.items()}
            )

        if anchors is None:
            anchors = space.anchors if not deepcopy else copy.deepcopy(space.anchors)

        return cls(name=name, vectors=vector_source, features=features, anchors=anchors)

    def __init__(
        self,
        vector_source: Union[torch.Tensor, VectorSource],
        name: str,
        features: Dict[str, Sequence[Any]] | None = None,
        anchors: Space | None = None,
    ):
        super().__init__(vector_source=vector_source, name=name, features=features)
        self.anchors: Space = anchors
