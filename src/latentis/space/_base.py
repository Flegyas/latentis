from __future__ import annotations

import copy
import json
import logging
import shutil
from enum import auto
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Mapping, Optional, Sequence, Union

from latentis.data.utils import ModelSpec
from latentis.modules import Decoder
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


class SpaceProperty(StrEnum):
    NAME = auto()
    VECTOR_SOURCE = auto()
    DATASET = auto()
    MODEL_SPEC = auto()


class LatentSpace(SerializableMixin):
    def __init__(
        self,
        vector_source: Optional[Union[torch.Tensor, VectorSource]],
        keys: Optional[Sequence[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        model_spec: Optional[ModelSpec] = None,
        name: str = "space",
        decoders: Optional[Dict[str, Decoder]] = None,
    ):
        super().__init__()

        if vector_source is None:
            vector_source = torch.empty(0)

        assert isinstance(
            vector_source, (torch.Tensor, VectorSource)
        ), f"Expected {torch.Tensor} or {VectorSource}, got {type(vector_source)}"

        assert name is not None and len(name.strip()) > 0, "Name must be a non-empty string."
        self.name: str = name

        if metadata is None:
            metadata = {}
        assert isinstance(metadata, Mapping), f"Expected {Mapping}, got {type(metadata)}"

        self._vector_source: torch.Tensor = (
            TensorSource(vectors=vector_source, keys=keys) if torch.is_tensor(vector_source) else vector_source
        )
        if SpaceProperty.VECTOR_SOURCE not in metadata:
            metadata[SpaceProperty.VECTOR_SOURCE] = type(self._vector_source).__name__

        self._metadata = metadata
        self._model_spec = model_spec
        self.decoders: Dict[str, Decoder] = decoders or {}

    @property
    def model_spec(self) -> Optional[ModelSpec]:
        return self._model_spec

    def add_property(self, key: str, value: Any):
        assert key not in self._metadata, f"Property with key {key} already exists."
        self._metadata[key] = value

    def get_vector_by_key(self, key: str) -> torch.Tensor:
        return self._vector_source.get_vector_by_key(key=key)

    @property
    def metadata(self) -> Dict[str, Any]:
        result = self._metadata
        result[SpaceProperty.NAME] = self.name
        return result

    def save_to_disk(self, target_path: Path, overwrite: bool = False):
        if overwrite:
            if target_path.exists():
                pylogger.warning(f"Overwriting existing space at {target_path}")

                shutil.rmtree(target_path)

        target_path.mkdir(parents=True, exist_ok=False)

        # save VectorSource
        vector_path = target_path / "vectors"
        vector_path.mkdir(parents=True, exist_ok=False)
        self._vector_source.save_to_disk(vector_path)

        # save metadata
        with open(target_path / "metadata.json", "w") as fw:
            json.dump(self.metadata, fw, indent=4, default=lambda o: o.__dict__)

        if self.model_spec is not None:
            self.model_spec.save_to_disk(target_path / "model_spec")

        # save decoders
        for decoder in self.decoders.values():
            decoder.save_to_disk(target_path / "decoders" / decoder.name)

    @classmethod
    def load_from_disk(cls, path: Path) -> LatentSpace:
        # load VectorSource
        vector_source = TensorSource.load_from_disk(path / "vectors")

        # load metadata
        with open(path / "metadata.json", "r") as fr:
            metadata = json.load(fr)

        # load decoders
        decoders = {}
        for decoder_path in (path / "decoders").glob("*"):
            decoder = Decoder.load_from_disk(decoder_path)
            decoders[decoder.name] = decoder

        space = LatentSpace(
            name=metadata[SpaceProperty.NAME], vector_source=vector_source, metadata=metadata, decoders=decoders
        )

        return space

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
        decoders: Optional[Dict[str, Decoder]] = None,
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
        if decoders is None:
            decoders = space.decoders if not deepcopy else copy.deepcopy(space.decoders)

        # TODO: test deepcopy
        return LatentSpace(name=name, vector_source=vector_source, metadata=metadata, decoders=decoders)

    @property
    def shape(self) -> torch.Size:
        return self._vector_source.shape

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

    def add_vectors(self, vectors: torch.Tensor, keys: Optional[Sequence[str]] = None) -> "LatentSpace":
        """Add vectors to this space.

        Args:
            vectors (torch.Tensor): The vectors to add.
            keys (Optional[Sequence[str]], optional): The keys of the vectors. Defaults to None.

        Returns:
            LatentSpace: The new space.
        """
        return self._vector_source.add_vectors(vectors=vectors, keys=keys)

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
        assert decoder.name not in self.decoders, f"Decoder with name {decoder.name} already exists."
        self.decoders[decoder.name] = decoder
