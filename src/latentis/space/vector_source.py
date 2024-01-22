from __future__ import annotations

import logging
from abc import abstractmethod
from pathlib import Path
from typing import Optional, Sequence, Union

import torch

from latentis.data.utils import BiMap
from latentis.serialize.io_utils import SerializableMixin

pylogger = logging.getLogger(__name__)


_VECTOR_SOURCES = {}


class VectorSourceMeta(type):
    def __new__(mcs, name, bases, attrs):
        cls = super().__new__(mcs, name, bases, attrs)
        key = cls.__module__ + "." + cls.__name__

        assert key not in _VECTOR_SOURCES, f"Duplicated vector source with key {key}"

        _VECTOR_SOURCES[key] = cls
        return cls


class VectorSource(metaclass=VectorSourceMeta):
    @abstractmethod
    def shape(self) -> torch.Size:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: Union[int, Sequence[int], slice]) -> torch.Tensor:
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

    @abstractmethod
    def add_vectors_(self, vectors: torch.Tensor, keys: Optional[Sequence[str]] = None) -> VectorSource:
        raise NotImplementedError

    @abstractmethod
    def get_vector_by_key(self, key: str) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def keys(self) -> Sequence[str]:
        raise NotImplementedError


class TensorSource(VectorSource, SerializableMixin):
    def __init__(self, vectors: torch.Tensor, keys: Optional[Sequence[str]] = None):
        assert (
            keys is None or len(keys) == 0 or len(keys) == vectors.size(0)
        ), "Keys must be None, empty, or have the same length as vectors"
        self._vectors = vectors
        keys = keys or []
        self._keys2offset = BiMap(x=keys, y=range(len(keys)))

    def shape(self) -> torch.Size:
        return self._vectors.shape

    def __getitem__(self, index: Union[int, Sequence[int], slice]) -> torch.Tensor:
        return self._vectors[index]

    def __len__(self) -> int:
        return self._vectors.size(0)

    def __eq__(self, __value: TensorSource) -> bool:
        assert isinstance(__value, TensorSource), f"Expected {TensorSource}, got {type(__value)}"
        return torch.allclose(self._vectors, __value._vectors)

    def as_tensor(self) -> torch.Tensor:
        return self._vectors

    def save_to_disk(self, parent_dir: Path):
        torch.save(self._vectors, parent_dir / "vectors.pt")
        self._keys2offset.save_to_disk(parent_dir / "mapping.tsv")

    @classmethod
    def load_from_disk(cls, path: Path) -> TensorSource:
        vectors = torch.load(path / "vectors.pt")
        keys = BiMap.load_from_disk(path / "mapping.tsv")
        result = cls(vectors=vectors)
        result._keys2offset = keys  # TODO: ugly
        return result

    def add_vectors_(self, vectors: torch.Tensor, keys: Optional[Sequence[str]] = None) -> TensorSource:
        assert (keys is None) == (
            len(self._keys2offset) == 0
        ), "Keys must be provided only if the source already has keys"
        if keys is not None:
            assert len(keys) == vectors.size(0), "Keys must have the same length as vectors"
            self._keys2offset.add_all(x=keys, y=range(len(self._keys2offset), len(self._keys2offset) + len(keys)))

        self._vectors = torch.cat([self._vectors, vectors], dim=0)
        return self

    def get_vector_by_key(self, key: str) -> torch.Tensor:
        assert len(self._keys2offset) > 0, "This source does not have keys enabled"
        try:
            return self[self._keys2offset.get_y(key)]
        except KeyError:
            raise KeyError(f"Key {key} not found in {self._keys2offset}")

    @property
    def keys(self) -> Sequence[str]:
        return self._keys2offset.x
