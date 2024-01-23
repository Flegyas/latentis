from __future__ import annotations

import copy
import logging
from enum import auto
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

from latentis.nn import LatentisModule
from latentis.serialize.disk_index import DiskIndex
from latentis.serialize.io_utils import IndexableMixin, load_json, load_model, save_json, save_model
from latentis.space.search import SearchIndex, SearchMetric
from latentis.space.vector_source import TensorSource, VectorSource
from latentis.transform import Transform
from latentis.types import Properties, StrEnum

if TYPE_CHECKING:
    from latentis.sample import Sampler
    from latentis.types import Space
    from latentis.measure import MetricFn
    from latentis.transform.translate import Translator

import torch

pylogger = logging.getLogger(__name__)


class _SpaceMetadata(StrEnum):
    _VERSION = auto()
    _VECTOR_SOURCE = auto()
    _TYPE = auto()


_PROPERTIES_FILE_NAME = "properties.json"


class LatentSpace(IndexableMixin):
    def __init__(
        self,
        vector_source: Optional[Union[torch.Tensor, Tuple[torch.Tensor, Sequence[str]], VectorSource]],
        source_model: Optional[LatentisModule] = None,
        properties: Optional[Properties] = None,
        root_path: Optional[Path] = None,
    ):
        super().__init__()
        self.root_path = root_path
        if vector_source is None:
            vector_source = torch.empty(0)

        # add "keys" as second element of tuple if not present
        if isinstance(vector_source, torch.Tensor):
            vector_source = (vector_source, None)

        assert isinstance(
            vector_source, (torch.Tensor, Tuple, VectorSource)
        ), f"Expected {torch.Tensor} or {VectorSource} or {Tuple[torch.Tensor, Sequence[str]]}, got {type(vector_source)}"

        self._vector_source: torch.Tensor = (
            TensorSource(vectors=vector_source[0], keys=vector_source[1])
            if isinstance(vector_source, tuple)
            else vector_source
        )
        self._source_model = source_model

        self._decoders = None
        if root_path is not None:
            self._decoders: Dict[str, LatentisModule] = DiskIndex(
                root_path=root_path / "decoders", item_class=LatentisModule
            )
            self._decoders.save_to_disk()

        properties = properties or {}
        # metadata[SpaceMetadata._NAME] = self._name
        properties[_SpaceMetadata._VERSION] = self.version

        # TODO: store also the module to use for deserialization,
        # removing this properties from the index
        properties[_SpaceMetadata._TYPE] = LatentSpace.__name__
        properties[_SpaceMetadata._VECTOR_SOURCE] = type(self._vector_source).__name__

        self._properties = properties.copy()

    @property
    def properties(self) -> Properties:
        return copy.deepcopy(self._properties)

    @property
    def split(self):
        return self.properties["split"]

    @property
    def name(self) -> str:
        return self.properties.get("name", "space")

    @property
    def decoders(self) -> Dict[str, LatentisModule]:
        if self.root_path is None:
            raise ValueError("Cannot store or access decoders if root_path is None")
        return self._decoders

    @property
    def version(cls) -> str:
        return -42

    @property
    def source_model(self) -> Optional[LatentisModule]:
        return self._source_model

    def get_vector_by_key(self, key: str) -> torch.Tensor:
        return self._vector_source.get_vector_by_key(key=key)

    def save_to_disk(
        self,
        target_path: Path,
        save_vector_source=True,
        save_properties=True,
        save_source_model=True,
    ):
        target_path.mkdir(parents=True, exist_ok=True)

        # save VectorSource
        if save_vector_source:
            vector_path = target_path / "vectors"
            vector_path.mkdir(parents=True, exist_ok=True)
            self._vector_source.save_to_disk(vector_path)

        # save metadata
        if save_properties:
            save_json(self.properties, target_path / _PROPERTIES_FILE_NAME)

        # save model
        if save_source_model:
            if self.source_model is not None:
                # TODO: save model to disk index
                save_model(model=self.source_model, target_path=target_path / "model.pt", version=self.version)

        # TODO: remove save to disk from disk index
        self.decoders.save_to_disk()

    @classmethod
    def load_properties(cls, space_path: Path) -> Dict[str, Any]:
        metadata = load_json(space_path / _PROPERTIES_FILE_NAME)

        return metadata

    @classmethod
    def load_from_disk(cls, path: Path, load_source_model: bool = False) -> LatentSpace:
        # load VectorSource
        vector_source = TensorSource.load_from_disk(path / "vectors")

        # load metadata
        properties = cls.load_properties(path)

        # load model
        model_path = path
        model = (
            load_model(model_path, version=properties[_SpaceMetadata._VERSION])
            if (model_path.exists() and load_source_model)
            else None
        )

        space = LatentSpace.__new__(cls)
        space._properties = properties
        space._vector_source = vector_source
        space._source_model = model
        space.root_path: Path = path

        try:
            space._decoders = DiskIndex.load_from_disk(path=path / "decoders")
        except FileNotFoundError:
            space._decoders = DiskIndex(root_path=path / "decoders", item_class=LatentisModule)
        return space

    @property
    def keys(self) -> Sequence[str]:
        return self._vector_source.keys

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
        #
        space: LatentSpace,
        vector_source: Optional[Union[torch.Tensor, Tuple[torch.Tensor, Sequence[str]], VectorSource]] = None,
        # decoders: Optional[Dict[str, LatentisModule]] = None,
        properties: Optional[Mapping[str, Any]] = None,
        #
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
        if vector_source is None:
            vector_source = space.vector_source if not deepcopy else copy.deepcopy(space.vector_source)

        # if decoders is None:
        #     decoders = space.decoders if not deepcopy else copy.deepcopy(space.decoders)
        # if source_model is None:
        #     source_model = space.source_model if not deepcopy else copy.deepcopy(space.source_model)

        if properties is None:
            properties = space.properties if not deepcopy else copy.deepcopy(space.properties)

        # TODO: test deepcopy
        return LatentSpace(
            vector_source=vector_source,
            # decoders=decoders,
            properties=properties,
        )

    @property
    def shape(self) -> torch.Size:
        return self._vector_source.shape

    def __getitem__(self, index: Union[int, Sequence[int], slice]) -> Mapping[str, torch.Tensor]:
        return self._vector_source[index]

    def __iter__(self):
        return iter(self._vector_source)

    def __len__(self) -> int:
        return len(self._vector_source)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(vectors={self.vectors.shape}, metadata={self.properties})"

    def __eq__(self, __value: object) -> bool:
        return self.properties == __value.properties and self.vector_source == __value.vector_source

    def add_vectors(self, vectors: torch.Tensor, keys: Optional[Sequence[str]] = None) -> None:
        """Add vectors to this space.

        Args:
            vectors (torch.Tensor): The vectors to add.
            keys (Optional[Sequence[str]], optional): The keys of the vectors. Defaults to None.

        Returns:
            LatentSpace: The new space.
        """
        self._vector_source.add_vectors_(vectors=vectors, keys=keys)

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

    def to_index(
        self,
        metric_fn: SearchMetric,
        keys: Optional[Sequence[str]] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> SearchIndex:
        """Create a SearchIndex from this space.

        Args:
            metric_fn (SearchMetric): The metric to use.
            keys (Optional[Sequence[str]], optional): The keys of the vectors. Defaults to None.
            transform (Optional[Callable[[torch.Tensor], torch.Tensor]], optional): A transform to apply to the vectors. Defaults to None.
        """
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

    def transform(self, transform: Union[Transform, Translator]) -> "LatentSpace":
        return LatentSpace.like(
            space=self,
            vector_source=transform(x=self.vectors),
        )

    def to_hf_dataset(self, name: str) -> Any:
        raise NotImplementedError
