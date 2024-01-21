from __future__ import annotations

import copy
import hashlib
import json
import logging
from enum import auto
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Mapping, Optional, Sequence, Union

from latentis.io_utils import load_json, load_model, save_json, save_model
from latentis.modules import Decoder, LatentisModule
from latentis.space.search import SearchIndex, SearchMetric
from latentis.space.vector_source import TensorSource, VectorSource
from latentis.transform import Transform
from latentis.types import SerializableMixin, StrEnum

if TYPE_CHECKING:
    from latentis.sample import Sampler
    from latentis.types import Space
    from latentis.measure import MetricFn
    from latentis.transform.translate import Translator

import torch

pylogger = logging.getLogger(__name__)


class SpaceMetadata(StrEnum):
    _VERSION = auto()
    _VECTOR_SOURCE = auto()
    _DECODER_KEYS = auto()
    _TYPE = auto()
    _ENCODING_KEY = auto()


class EncodingKey(dict):
    dataset: str
    feature: str
    model_name: str
    split: str

    def __init__(self, dataset: str, feature: str, model_name: str, split: str, **kwargs):
        super().__init__(
            dataset=dataset,
            feature=feature,
            model_name=model_name,
            split=split,
            **kwargs,
        )

    def hash(self) -> str:
        hashcode = hashlib.sha256()

        hashcode.update(json.dumps(self, sort_keys=True).encode("utf-8"))

        return hashcode.hexdigest()[:8]

    def __getattr__(self, __name: str) -> Any:
        return super().__getitem__(__name)

    def get_path(self, data_root: Path) -> Path:
        return data_root / self.dataset / "encodings" / self.feature / self.split / self.hash()

    def to_space(self, data_root: Path, load_source_model: bool = False, load_decoders: bool = False) -> LatentSpace:
        return LatentSpace.load_from_disk(
            self.get_path(data_root), load_source_model=load_source_model, load_decoders=load_decoders
        )


class LatentSpace(SerializableMixin):
    def __init__(
        self,
        vector_source: Optional[Union[torch.Tensor, VectorSource]],
        keys: Optional[Sequence[str]] = None,
        decoders: Optional[Dict[str, Decoder]] = None,
        source_model: Optional[LatentisModule] = None,
        encoding_key: Optional[EncodingKey] = None,
    ):
        super().__init__()

        if vector_source is None:
            vector_source = torch.empty(0)

        assert isinstance(
            vector_source, (torch.Tensor, VectorSource)
        ), f"Expected {torch.Tensor} or {VectorSource}, got {type(vector_source)}"

        self._vector_source: torch.Tensor = (
            TensorSource(vectors=vector_source, keys=keys) if torch.is_tensor(vector_source) else vector_source
        )
        self._source_model = source_model
        self._decoders: Dict[str, Decoder] = decoders or {}

        metadata = {}
        # metadata[SpaceMetadata._NAME] = self._name
        metadata[SpaceMetadata._VERSION] = self.version
        metadata[SpaceMetadata._TYPE] = LatentSpace.__name__
        metadata[SpaceMetadata._VECTOR_SOURCE] = type(self._vector_source).__name__
        metadata[SpaceMetadata._ENCODING_KEY] = encoding_key

        self._metadata = metadata

    @property
    def encoding_key(self) -> Optional[EncodingKey]:
        return self.metadata[SpaceMetadata._ENCODING_KEY]

    @property
    def name(self) -> str:
        return self.encoding_key.name if self.encoding_key is not None else "space"

    @property
    def decoders(self) -> Dict[str, Decoder]:
        return self._decoders

    @property
    def version(self) -> str:
        return -42

    @property
    def source_model(self) -> Optional[LatentisModule]:
        return self._source_model

    def add_property(self, key: str, value: Any):
        assert key not in self._metadata, f"Property with key {key} already exists."
        self._metadata[key] = value

    def get_vector_by_key(self, key: str) -> torch.Tensor:
        return self._vector_source.get_vector_by_key(key=key)

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata.copy()

    def save_to_disk(
        self,
        target_path: Path,
        save_vector_source=True,
        save_metadata=True,
        save_source_model=True,
        save_decoders=True,
    ):
        target_path.mkdir(parents=True, exist_ok=True)

        # save VectorSource
        if save_vector_source:
            vector_path = target_path / "vectors"
            vector_path.mkdir(parents=True, exist_ok=True)
            self._vector_source.save_to_disk(vector_path)

        # save metadata
        if save_metadata:
            save_json(self.metadata, target_path / "metadata.json")

        # save model
        if save_source_model:
            if self.source_model is not None:
                save_model(model=self.source_model, target_path=target_path / "model.pt", version=self.version)

        # save decoders
        if save_decoders:
            for decoder in self._decoders.values():
                save_model(model=decoder, target_path=target_path / "decoders" / decoder.name, version=self.version)

    @staticmethod
    def load_metadata(space_path: Path) -> Dict[str, Any]:
        metadata = load_json(space_path / "metadata.json")

        metadata[SpaceMetadata._ENCODING_KEY] = (
            EncodingKey(**metadata[SpaceMetadata._ENCODING_KEY]) if metadata[SpaceMetadata._ENCODING_KEY] else None
        )

        return metadata

    @classmethod
    def load_from_disk(cls, path: Path, load_source_model: bool = False, load_decoders: bool = False) -> LatentSpace:
        # load VectorSource
        vector_source = TensorSource.load_from_disk(path / "vectors")

        # load metadata
        metadata = cls.load_metadata(path)

        # load decoders
        decoders = {}
        for decoder_path in (path / "decoders").glob("*"):
            decoder = load_model(decoder_path, version=metadata[SpaceMetadata._VERSION]) if load_decoders else None
            decoders[decoder.name] = decoder

        # load model
        model_path = path / "model.pt"
        model = (
            load_model(model_path, version=metadata[SpaceMetadata._VERSION])
            if (model_path.exists() and load_source_model)
            else None
        )

        space = LatentSpace.__new__(cls)
        space._metadata = metadata
        space._vector_source = vector_source
        space._decoders = decoders
        space._source_model = model

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
        # name: Optional[str] = None,
        vector_source: Optional[VectorSource] = None,
        decoders: Optional[Dict[str, Decoder]] = None,
        space_spec: Optional[EncodingKey] = None,
        keys: Optional[Sequence[str]] = None,
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
        if decoders is None:
            decoders = space.decoders if not deepcopy else copy.deepcopy(space.decoders)
        # if source_model is None:
        #     source_model = space.source_model if not deepcopy else copy.deepcopy(space.source_model)
        if keys is None:
            keys = space.keys if not deepcopy else copy.deepcopy(space.keys)
            assert keys is not None, "Keys must not be None."
        if space_spec is None:
            space_spec = space.encoding_key if not deepcopy else copy.deepcopy(space.encoding_key)

        # TODO: test deepcopy
        return LatentSpace(
            vector_source=vector_source,
            decoders=decoders,
            keys=keys,
            encoding_key=space_spec,
        )

    @property
    def shape(self) -> torch.Size:
        return self._vector_source.shape

    def __getitem__(self, index: Union[int, Sequence[int], slice]) -> Mapping[str, torch.Tensor]:
        return self._vector_source[index]

    def __len__(self) -> int:
        return len(self._vector_source)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(vectors={self.vectors.shape}, metadata={self.metadata}"
            + ("space_spec=")
            + (str(self.encoding_key) if self.encoding_key is not None else "")
            + ")"
        )

    def __eq__(self, __value: object) -> bool:
        return (
            self._name == __value.name
            and self.metadata == __value.metadata
            and self._vector_source == __value._vector_source
        )

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

    def add_decoder(self, decoder: Decoder):
        assert decoder.name not in self._decoders, f"Decoder with name {decoder.name} already exists."
        self._decoders[decoder.name] = decoder
