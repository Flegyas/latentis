from __future__ import annotations

import logging
from abc import abstractmethod
from pathlib import Path

import torch

from latentis.types import SerializableMixin

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


class InMemorySource(VectorSource, SerializableMixin):
    def __init__(self, vectors: torch.Tensor):
        self._vectors = vectors

    def shape(self) -> torch.Size:
        return self._vectors.shape

    def __getitem__(self, index: int) -> torch.Tensor:
        return self._vectors[index]

    def __len__(self) -> int:
        return self._vectors.size(0)

    def __eq__(self, __value: InMemorySource) -> bool:
        assert isinstance(__value, InMemorySource), f"Expected {InMemorySource}, got {type(__value)}"
        return torch.allclose(self._vectors, __value._vectors)

    def as_tensor(self) -> torch.Tensor:
        return self._vectors

    def save_to_disk(self, parent_dir: Path):
        torch.save(self._vectors, parent_dir / "vectors.pt")

    @classmethod
    def load_from_disk(cls, path: Path) -> InMemorySource:
        return cls(torch.load(path / "vectors.pt"))
