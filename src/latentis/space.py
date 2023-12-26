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
        vector_source: Union[torch.Tensor, VectorSource],
        metadata: Optional[Dict[str, Any]] = None,
        name: str = "",
    ):
        assert isinstance(
            vector_source, (torch.Tensor, VectorSource)
        ), f"Expected {torch.Tensor} or {VectorSource}, got {type(vector_source)}"
        self.name: str = name
        if metadata is None:
            metadata = {}
        self._vector_source: torch.Tensor = (
            TorchVectorSource(vector_source) if torch.is_tensor(vector_source) else vector_source
        )
        self.metadata = metadata

    @property
    def vectors(self) -> torch.Tensor:
        return self._vector_source.as_tensor()

    def to_memory(self) -> "LatentSpace":
        if isinstance(self._vector_source, torch.Tensor):
            pylogger.info("Already in memory, skipping.")
            return self

        return LatentSpace.like(
            space=self,
            vector_source=self.as_tensor(),
        )

    @classmethod
    def like(
        cls,
        space: LatentSpace,
        name: Optional[str] = None,
        vector_source: Optional[VectorSource] = None,
        metadata: Optional[Dict[str, Any]] = None,
        deepcopy: bool = False,
    ):
        """Create a new space with the arguments not provided taken from the given space.

        There is no copy of the vectors, so changes to the vectors of the new space will also affect the vectors of the given space.

        Args:
            space (LatentSpace): The space to copy.
            name (Optional[str]): The name of the new space.
            vector_source (Optional[torch.Tensor], optional): The vectors of the new space.
            metadata (Optional[Dict[str, Any]], optional): The metadata of the new space.

        Returns:
            LatentSpace: The new space, with the arguments not provided taken from the given space.
        """
        if name is None:
            name = space.name
        if vector_source is None:
            vector_source = space.vector_source if not deepcopy else copy.deepcopy(space.vector_source)
        if metadata is None:
            metadata = space.metadata if not deepcopy else copy.deepcopy(space.metadata)
        # TODO: test deepcopy
        return LatentSpace(name=name, vector_source=vector_source, metadata=metadata)

    @property
    def shape(self) -> torch.Size:
        return self.vectors.shape

    def __getitem__(self, index: int) -> Mapping[str, torch.Tensor]:
        return self._vector_source[index]

    def __len__(self) -> int:
        return len(self._vector_source)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, vectors={self.vectors.shape}, metadata={self.metadata})"

    def __eq__(self, __value: object) -> bool:
        return (
            self.name == __value.name
            and self.metadata == __value.metadata
            and self._vector_source == __value._vector_source
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
