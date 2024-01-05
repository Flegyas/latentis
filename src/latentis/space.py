from __future__ import annotations

import copy
import json
import logging
from enum import auto
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Mapping, Optional, Sequence, Union

from latentis.search import SearchIndex, SearchMetric
from latentis.vector_source import InMemorySource, VectorSource

if TYPE_CHECKING:
    from latentis.sample import Sampler
    from latentis.types import Space
    from latentis.translate import LatentTranslator
    from latentis.measure import MetricFn

import torch

try:
    # be ready for 3.10 when it drops
    from enum import StrEnum
except ImportError:
    from backports.strenum import StrEnum

pylogger = logging.getLogger(__name__)


class SpaceProperty(StrEnum):
    VECTOR_SOURCE = auto()


class LatentSpace:
    def __init__(
        self,
        vector_source: Union[torch.Tensor, VectorSource],
        metadata: Optional[Dict[str, Any]] = None,
        name: str = "space",
    ):
        assert isinstance(
            vector_source, (torch.Tensor, VectorSource)
        ), f"Expected {torch.Tensor} or {VectorSource}, got {type(vector_source)}"
        assert name is not None and len(name.strip()) > 0, "Name must be a non-empty string."

        self.name: str = name
        if metadata is None:
            metadata = {}
        self._vector_source: torch.Tensor = (
            InMemorySource(vector_source) if torch.is_tensor(vector_source) else vector_source
        )
        if SpaceProperty.VECTOR_SOURCE not in metadata:
            metadata[SpaceProperty.VECTOR_SOURCE] = type(self._vector_source).__name__

        self.metadata = metadata

    def save_to_disk(self, parent_dir: Path, name: Optional[str] = None):
        if name is None:
            name = self.name

        path = parent_dir / name
        path.mkdir(parents=True, exist_ok=False)

        # save VectorSource
        self._vector_source.save_to_disk(path / "vectors")

        # save metadata
        with open(parent_dir / "metadata.json", "w") as fw:
            json.dump(self.metadata, fw, indent=4, default=lambda o: o.__dict__)

    @classmethod
    def load_from_disk(cls, path: Path) -> LatentSpace:
        # load VectorSource
        vector_source = InMemorySource.load_from_disk(path / "vectors")

        # load metadata
        with open(path / "metadata.json", "r") as fr:
            metadata = json.load(fr)

        return LatentSpace(vector_source=vector_source, metadata=metadata)

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
