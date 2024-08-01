from __future__ import annotations

import copy
import importlib
import logging
from enum import auto
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Type, Union

from latentis.nn import LatentisModule
from latentis.serialize.disk_index import DiskIndex
from latentis.serialize.io_utils import SerializableMixin, load_json, load_model, save_json, save_model
from latentis.space.search import SearchMetric, SearchResult
from latentis.space.vector_source import SearchSource, TensorSource, VectorSource
from latentis.transform import Transform
from latentis.types import Metadata, StrEnum

if TYPE_CHECKING:
    from latentis.sample import Sampler
    from latentis.types import LatentisSpace
    from latentis.measure import MetricFn
    from latentis.transform.translate import Translator

import torch

pylogger = logging.getLogger(__name__)


class _SpaceMetadata(StrEnum):
    _VERSION = auto()
    _VECTOR_SOURCE = auto()
    _TYPE = auto()


_METADATA_FILE_NAME = "metadata.json"


class Space(SerializableMixin):
    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, **kwargs) -> Space:
        return cls(vector_source=tensor, **kwargs)

    def __init__(
        self,
        vector_source: Optional[Union[torch.Tensor, Tuple[torch.Tensor, Sequence[str]], VectorSource]],
        source_model: Optional[LatentisModule] = None,
        metadata: Optional[Metadata] = None,
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

        metadata = metadata or {}
        # metadata[SpaceMetadata._NAME] = self._name
        metadata[_SpaceMetadata._VERSION] = self.version

        # removing this metadata from the index
        metadata[_SpaceMetadata._TYPE] = self.__class__.__module__ + "." + self.__class__.__name__
        metadata[_SpaceMetadata._VECTOR_SOURCE] = (
            self._vector_source.__class__.__module__ + "." + self._vector_source.__class__.__name__
        )

        self._metadata = metadata.copy()

    def search_knn(
        self,
        k: int,
        *,
        metric_fn: Optional[SearchMetric] = None,
        query_offsets: Optional[Sequence[int]] = None,
        query_vectors: Optional[torch.Tensor] = None,
        query_keys: Optional[Sequence[str]] = None,
        return_keys: bool = False,
        **kwargs,
    ) -> SearchResult:
        """Performs k-nearest neighbors search based on the provided query.

        Args:
            k (int): The number of nearest neighbors to retrieve.
            query_offsets (Optional[Sequence[int]], optional): The offsets of the queries.
                Exactly one of `query_offsets`, `query_vectors`, or `query_keys` must be provided.
                Defaults to None.
            query_vectors (Optional[torch.Tensor], optional): The query vectors.
                Exactly one of `query_offsets`, `query_vectors`, or `query_keys` must be provided.
                Defaults to None.
            query_keys (Optional[Sequence[str]], optional): The keys of the queries.
                Exactly one of `query_offsets`, `query_vectors`, or `query_keys` must be provided.
                Defaults to None.
            return_keys (bool): Whether to return the keys of the nearest neighbors.

        Returns:
            The nearest neighbors based on the provided query.

        Raises:
            AssertionError: If none or more than one of query_offsets, query_vectors, or query_keys are provided.
        """
        return self._vector_source.search_knn(
            metric_fn=metric_fn,
            k=k,
            query_offsets=query_offsets,
            query_vectors=query_vectors,
            query_keys=query_keys,
            return_keys=return_keys,
            **kwargs,
        )

    def search_range(
        self,
        radius: float,
        *,
        metric_fn: Optional[SearchMetric] = None,
        query_offsets: Optional[Sequence[int]] = None,
        query_vectors: Optional[torch.Tensor] = None,
        query_keys: Optional[Sequence[str]] = None,
        return_keys: bool = False,
        **kwargs,
    ) -> SearchResult:
        """Perform a range search in the latent space.

        Args:
            radius (float): The radius of the search range.
            query_offsets (Optional[Sequence[int]], optional): The offsets of the queries.
                Exactly one of `query_offsets`, `query_vectors`, or `query_keys` must be provided.
                Defaults to None.
            query_vectors (Optional[Union[np.ndarray, torch.Tensor]], optional): The query vectors.
                Exactly one of `query_offsets`, `query_vectors`, or `query_keys` must be provided.
                Defaults to None.
            query_keys (Optional[Sequence[str]], optional): The keys of the queries.
                Exactly one of `query_offsets`, `query_vectors`, or `query_keys` must be provided.
                Defaults to None.
            transform (bool, optional): Whether to transform the query vectors. Defaults to False.
            return_keys (bool, optional): Whether to return the keys of the search results. Defaults to False.

        Returns:
            The search results based on the provided query type.
        """
        return self._vector_source.search_range(
            metric_fn=metric_fn,
            radius=radius,
            query_offsets=query_offsets,
            query_vectors=query_vectors,
            query_keys=query_keys,
            return_keys=return_keys,
            **kwargs,
        )

    def to(self, device: Union[str, torch.device]) -> "Space":
        return Space.like(
            space=self,
            vector_source=self._vector_source.to(device),
        )

    def size(self) -> torch.Size:
        return self.shape

    def partition(self, sizes: Sequence[float], seed: int) -> Sequence["Space"]:
        """Partition this space into multiple spaces.

        Args:
            sizes (Sequence[float]): The sizes of the partitions. Either a list of floats that sum to 1, or a list of integers.

        Returns:
            Sequence[Space]: The partitions.
        """
        if sum(sizes) != 1:
            if sum(sizes) != len(self):
                raise ValueError("Sizes must sum to 1 or be the same length as the number of vectors.")

            sizes = [size / sum(sizes) for size in sizes]

        return self._vector_source.partition(sizes, seed=seed)

    @property
    def metadata(self) -> Metadata:
        return copy.deepcopy(self._metadata)

    @property
    def split(self):
        return self.metadata["split"]

    @property
    def name(self) -> str:
        return self.metadata.get("name", "space")

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

    def get_vector(self, offset: Optional[int] = None, key: Optional[str] = None) -> torch.Tensor:
        if (offset is not None) and (key is not None):
            raise ValueError("Only one of offset or key can be provided.")

        if offset is not None:
            return self._vector_source[offset]

        if key is not None:
            return self._vector_source.get_vector_by_key(key=key)

        raise ValueError("One of offset or key must be provided.")

    def save_to_disk(
        self,
        target_path: Path,
        save_vector_source=True,
        save_metadata=True,
        save_source_model=True,
    ):
        target_path.mkdir(parents=True, exist_ok=True)

        # save VectorSource
        if save_vector_source:
            vector_path = target_path / "vectors"
            vector_path.mkdir(parents=True, exist_ok=True)
            self._vector_source.save_to_disk(vector_path)

        # save metadata
        if save_metadata:
            save_json(self.metadata, target_path / _METADATA_FILE_NAME)

        # save model
        if save_source_model:
            if self.source_model is not None:
                save_model(model=self.source_model, target_path=target_path / "model.pt", version=self.version)

        # TODO: remove save to disk from disk index
        if self._decoders is not None:
            self._decoders.save_to_disk()

    @classmethod
    def load_metadata(cls, space_path: Path) -> Dict[str, Any]:
        metadata = load_json(space_path / _METADATA_FILE_NAME)

        return metadata

    @classmethod
    def load_from_disk(cls, path: Path, load_source_model: bool = False) -> Space:
        # load metadata
        metadata = cls.load_metadata(path)

        # load correct VectorSource
        vector_source_cls = metadata[_SpaceMetadata._VECTOR_SOURCE]
        vector_source_pkg, vector_source_cls = vector_source_cls.rsplit(".", 1)
        vector_source_cls = getattr(importlib.import_module(vector_source_pkg), vector_source_cls)

        vector_source = vector_source_cls.load_from_disk(path / "vectors")

        # load model
        model_path = path
        model = (
            load_model(model_path, version=metadata[_SpaceMetadata._VERSION])
            if (model_path.exists() and load_source_model)
            else None
        )

        space = Space.__new__(cls)
        space._metadata = metadata
        space._vector_source = vector_source
        space._source_model = model
        space.root_path = path

        try:
            space._decoders = DiskIndex.load_from_disk(path=path / "decoders")
        except FileNotFoundError:
            space._decoders = DiskIndex(root_path=path / "decoders", item_class=LatentisModule)
        return space

    @property
    def keys(self) -> Sequence[str]:
        return list(self._vector_source.keys)

    def get_keys(self, offsets: Sequence[int]) -> Sequence[str]:
        return [self.keys[offset] for offset in offsets]

    def as_tensor(self, device: torch.device = "cpu") -> torch.Tensor:
        return self._vector_source.as_tensor(device=device)

    def to_memory(self) -> "Space":
        if isinstance(self._vector_source, torch.Tensor):
            pylogger.info("Already in memory, skipping.")
            return self

        return Space.like(
            space=self,
            vector_source=self.as_tensor(),
        )

    def like_(
        self,
        vector_source: Optional[Union[torch.Tensor, Tuple[torch.Tensor, Sequence[str]], VectorSource]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        deepcopy: bool = False,
    ):
        return type(self).like(
            space=self,
            vector_source=vector_source,
            metadata=metadata,
            deepcopy=deepcopy,
        )

    @classmethod
    def like(
        cls,
        #
        space: Space,
        vector_source: Optional[Union[torch.Tensor, Tuple[torch.Tensor, Sequence[str]], VectorSource]] = None,
        # decoders: Optional[Dict[str, LatentisModule]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        #
        deepcopy: bool = False,
    ):
        """Create a new space with the arguments not provided taken from the given space.

        There is no copy of the vectors, so changes to the vectors of the new space will also affect the vectors of the given space.

        Args:
            space (Space): The space to copy.
            name (Optional[str]): The name of the new space.
            vector_source (Optional[torch.Tensor], optional): The vectors of the new space.
            metadata (Optional[Dict[str, Any]], optional): The metadata of the new space.

        Returns:
            Space: The new space, with the arguments not provided taken from the given space.
        """
        if vector_source is None:
            vector_source = space.vector_source if not deepcopy else copy.deepcopy(space.vector_source)

        # if decoders is None:
        #     decoders = space.decoders if not deepcopy else copy.deepcopy(space.decoders)
        # if source_model is None:
        #     source_model = space.source_model if not deepcopy else copy.deepcopy(space.source_model)

        if metadata is None:
            metadata = space.metadata if not deepcopy else copy.deepcopy(space.metadata)

        # TODO: test deepcopy
        return Space(
            vector_source=vector_source,
            # decoders=decoders,
            metadata=metadata,
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
        return f"{self.__class__.__name__}(vectors={self.shape}, metadata={self.metadata})"

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Space):
            return False

        return self.metadata == __value.metadata and self._vector_source == __value._vector_source

    def get_vectors_by_key(self, keys: Sequence[str]) -> torch.Tensor:
        return self._vector_source.get_vectors_by_key(keys=keys)

    def add_vectors(self, vectors: torch.Tensor, keys: Optional[Sequence[str]] = None, **kwargs) -> None:
        """Add vectors to this space.

        Args:
            vectors (torch.Tensor): The vectors to add.
            keys (Optional[Sequence[str]], optional): The keys of the vectors. Defaults to None.

        Returns:
            Space: The new space.
        """
        self._vector_source.add_vectors(vectors=vectors, keys=keys, **kwargs)

    def sample(self, sampler: Sampler, n: int) -> "Space":
        """Sample n vectors from this space using the given sampler.

        Args:
            sampler (Sampler): The sampler to use.
            n (int): The number of vectors to sample.

        Returns:
            Space: The sampled space.

        """
        return sampler(self, n=n)

    def compare(
        self,
        *others: LatentisSpace,
        metrics: Mapping[str, Union[SearchMetric, Callable[[LatentisSpace, LatentisSpace], torch.Tensor]]],
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

    def to_source(
        self,
        source_cls: Type[VectorSource],
        **kwargs,
    ) -> SearchSource:
        """Create a SearchIndex from this space.

        Args:
            source_cls (Type[SearchSource]): The class of the source to create.
        """
        source = source_cls.from_source(source=self._vector_source, **kwargs)

        return self.like(
            space=self,
            vector_source=source,
        )

    def transform(self, transform: Union[Transform, Translator]) -> "Space":
        return Space.like(
            space=self,
            vector_source=(transform(x=self.as_tensor()), self.keys),
        )

    def to_hf_dataset(self, name: str) -> Any:
        raise NotImplementedError
