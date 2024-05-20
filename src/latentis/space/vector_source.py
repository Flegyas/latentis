from __future__ import annotations

import logging
from abc import abstractmethod
from pathlib import Path
from typing import Optional, Sequence, Union

import h5py
import numpy as np
import torch

from latentis.data.utils import BiMap
from latentis.serialize.io_utils import SerializableMixin

pylogger = logging.getLogger(__name__)


def _torch_dtype_to_numpy(dtype: torch.dtype) -> np.dtype:
    try:
        return np.dtype(str(dtype).replace("torch.", "np."))
    except TypeError:
        return None


_VECTOR_SOURCES = {}


class VectorSourceMeta(type):
    def __new__(mcs, name, bases, attrs):
        cls = super().__new__(mcs, name, bases, attrs)
        key = cls.__module__ + "." + cls.__name__

        assert key not in _VECTOR_SOURCES, f"Duplicated vector source with key {key}"

        _VECTOR_SOURCES[key] = cls
        return cls


class VectorSource(metaclass=VectorSourceMeta):
    @property
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
    def get_vector_by_key(self, key: str) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def keys(self) -> Sequence[str]:
        raise NotImplementedError

    @abstractmethod
    def add_vectors(self, vectors: torch.Tensor, keys: Optional[Sequence[str]] = None) -> VectorSource:
        raise NotImplementedError

    def get_vectors_by_key(self, keys: Sequence[str]) -> torch.Tensor:
        return torch.stack([self.get_vector_by_key(key) for key in keys], dim=0)

    def partition(self, sizes: Sequence[float], seed: int) -> Sequence[VectorSource]:
        assert sum(sizes) == 1, "Sizes must sum to 1"
        sizes = [int(size * len(self)) for size in sizes]
        sizes[-1] += len(self) - sum(sizes)
        return [self.__class__(self[i : i + size]) for i, size in enumerate(sizes)]


class TensorSource(VectorSource, SerializableMixin):
    def __init__(self, vectors: torch.Tensor, keys: Optional[Sequence[str]] = None):
        assert (
            keys is None or len(keys) == 0 or len(keys) == vectors.size(0)
        ), "Keys must be None, empty, or have the same length as vectors"
        self._vectors = vectors
        keys = keys or []
        self._keys2offset = BiMap(x=keys, y=range(len(keys)))

    @property
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
        torch.save(self._vectors, parent_dir / "data.pt")
        self._keys2offset.save_to_disk(parent_dir / "mapping.tsv")

    @classmethod
    def load_from_disk(cls, path: Path) -> TensorSource:
        vectors = torch.load(path / "data.pt")
        keys = BiMap.load_from_disk(path / "mapping.tsv")
        result = cls(vectors=vectors)
        result._keys2offset = keys  # TODO: ugly
        return result

    def get_vector_by_key(self, key: str) -> torch.Tensor:
        assert len(self._keys2offset) > 0, "This source does not have keys enabled"
        try:
            return self[self._keys2offset.get_y(key)]
        except KeyError:
            raise KeyError(f"Key {key} not found in {self._keys2offset}")

    @property
    def keys(self) -> Sequence[str]:
        return self._keys2offset.x

    def add_vectors(self, vectors: torch.Tensor, keys: Sequence[str] | None = None) -> VectorSource:
        if vectors.ndim == 1:
            vectors = vectors.unsqueeze(0)

        assert (keys is None) == (
            len(self._keys2offset) == 0
        ), "Keys must be provided only if the source already has keys"
        if keys is not None:
            assert len(keys) == vectors.size(0), "Keys must have the same length as vectors"
            self._keys2offset.add_all(x=keys, y=range(len(self._keys2offset), len(self._keys2offset) + len(keys)))

        self._vectors = torch.cat([self._vectors, vectors], dim=0)
        return self


class HDF5Source(VectorSource):
    def __init__(self, num_elements: int, dimension: int, root_dir: Path, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        root_dir.mkdir(parents=True, exist_ok=True)
        self.root_dir = root_dir
        data_path = root_dir / "vectors" / "data.h5"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        self.h5_file = h5py.File(data_path, mode="w")
        self.data: h5py.Dataset = self.h5_file.create_dataset(
            "data",
            shape=(num_elements, dimension),
            dtype=_torch_dtype_to_numpy(dtype) or np.float32,
            fillvalue=0,
            maxshape=(num_elements, dimension),
        )
        self.h5_file.attrs["last_index"] = 0  # Tracking the last index used
        self._keys2offset = BiMap(x=[], y=[])

    @property
    def keys(self) -> Sequence[str]:
        return self._keys2offset.x

    @property
    def shape(self) -> torch.Size:
        return self.data.shape

    @property
    def dtype(self) -> torch.dtype:
        return torch.dtype(self.data.dtype)

    def __getitem__(self, index: Union[int, Sequence[int], slice]) -> torch.Tensor:
        if isinstance(index, int):
            return torch.as_tensor(self.data[index])
        sort_idx = np.argsort(index)
        return torch.as_tensor(self.data[index[sort_idx]][sort_idx.argsort()])

    def __len__(self) -> int:
        return self.data.shape[0]

    def as_tensor(self) -> torch.Tensor:
        return torch.as_tensor(np.asarray(self.data))

    def __eq__(self, __value: HDF5Source) -> bool:
        assert isinstance(__value, HDF5Source), f"Expected {HDF5Source}, got {type(__value)}"
        return torch.allclose(self.data, __value.data)

    def add_vectors(
        self, vectors: torch.Tensor, keys: Optional[Sequence[str]] = None, write: bool = False
    ) -> VectorSource:
        if vectors.ndim == 1:
            vectors = vectors.unsqueeze(0)

        # assert (keys is None) == (
        #     len(self._keys2offset) == 0
        # ), "Keys must be provided only if the source already has keys"
        if keys is not None:
            assert len(keys) == vectors.size(0), "Keys must have the same length as vectors"
            self._keys2offset.add_all(x=keys, y=range(len(self._keys2offset), len(self._keys2offset) + len(keys)))

        self.data[
            self.h5_file.attrs["last_index"] : self.h5_file.attrs["last_index"] + vectors.size(0)
        ] = vectors.cpu().numpy()
        self.h5_file.attrs["last_index"] += vectors.size(0)

        if write:
            self.h5_file.flush()
            self._save_mapping()

        return self

    def _save_mapping(self):
        self._keys2offset.save_to_disk(self.root_dir / "vectors" / "mapping.tsv")

    def to_tensor_source(self) -> TensorSource:
        return TensorSource(vectors=self.as_tensor(), keys=self.keys)

    def save_to_disk(self, parent_dir: Path):
        self.h5_file.flush()
        self._save_mapping()

    @classmethod
    def load_from_disk(cls, path: Path) -> HDF5Source:
        data = path / "data.h5"
        data = h5py.File(data, mode="r")

        result = HDF5Source.__new__(HDF5Source)
        result.h5_file = data
        result.data = data["data"]
        result._keys2offset = BiMap.load_from_disk(path / "mapping.tsv")

        return result

    def get_vector_by_key(self, key: str) -> torch.Tensor:
        assert len(self._keys2offset) > 0, "This source does not have keys enabled"
        try:
            return self[self._keys2offset.get_y(key)]
        except KeyError:
            raise KeyError(f"Key {key} not found in {self._keys2offset}")
