from __future__ import annotations

import json
import logging
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import faiss as _faiss
import h5py
import numpy as np
import pandas as pd
import torch
import torch.types

from latentis.data.utils import BiMap
from latentis.serialize.io_utils import SerializableMixin
from latentis.space.search import BackendIndex, SearchMetric, SearchResult
from latentis.transform._abstract import Transform

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
    @classmethod
    def from_source(cls, source: VectorSource, *args, **kwargs) -> VectorSource:
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

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
    def as_tensor(self, device: torch.device = "cpu") -> torch.Tensor:
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
    def add_vectors(
        self, vectors: torch.Tensor, keys: Optional[Sequence[str]] = None
    ) -> VectorSource:
        raise NotImplementedError

    def get_vectors_by_key(self, keys: Sequence[str]) -> torch.Tensor:
        return torch.stack([self.get_vector_by_key(key) for key in keys], dim=0)

    def partition(self, sizes: Sequence[float], seed: int) -> Sequence[VectorSource]:
        assert sum(sizes) == 1, "Sizes must sum to 1"
        sizes = [int(size * len(self)) for size in sizes]
        sizes[-1] += len(self) - sum(sizes)
        return [self.__class__(self[i : i + size]) for i, size in enumerate(sizes)]

    def to(self, device: Union[str, torch.device]) -> VectorSource:
        return TensorSource(vectors=self.as_tensor(device=device), keys=self.keys)

    @abstractmethod
    def save_to_disk(self, target_path: Path):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load_from_disk(cls, path: Path) -> VectorSource:
        raise NotImplementedError

    def select(self, indices: Sequence[int]) -> VectorSource:
        source = TensorSource(
            vectors=self[indices], keys=[self.keys[i] for i in indices]
        )
        return source


class TensorSource(VectorSource, SerializableMixin):
    @classmethod
    def from_source(cls, source: VectorSource) -> VectorSource:
        return cls(vectors=source.as_tensor(), keys=source.keys)

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
        assert isinstance(
            __value, TensorSource
        ), f"Expected {TensorSource}, got {type(__value)}"
        return torch.allclose(self._vectors, __value._vectors)

    def as_tensor(self, device: torch.device = "cpu") -> torch.Tensor:
        return self._vectors.to(device=device)

    def save_to_disk(self, target_path: Path):
        assert target_path.is_dir(), f"Target path {target_path} must be a directory"

        torch.save(self._vectors, target_path / "data.pt")
        self._keys2offset.save_to_disk(target_path / "mapping.tsv")

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
            raise KeyError(f"Key {key} not found in {self._keys2offset}") from None

    @property
    def keys(self) -> Sequence[str]:
        return list(self._keys2offset.x)

    def add_vectors(
        self, vectors: torch.Tensor, keys: Sequence[str] | None = None
    ) -> VectorSource:
        if vectors.ndim == 1:
            vectors = vectors.unsqueeze(0)

        assert (keys is None) == (
            len(self._keys2offset) == 0
        ), "Keys must be provided only if the source already has keys"
        if keys is not None:
            assert len(keys) == vectors.size(
                0
            ), "Keys must have the same length as vectors"
            self._keys2offset.add_all(
                x=keys,
                y=range(len(self._keys2offset), len(self._keys2offset) + len(keys)),
            )

        self._vectors = torch.cat([self._vectors, vectors], dim=0)
        return self


_default_h5py_params = dict(
    dtype=np.float32,
    fillvalue=0,
    # "maxshape"= shape,
)


class HDF5Source(VectorSource):
    @classmethod
    def from_source(
        cls, source: VectorSource, root_dir: Path, write: bool = True
    ) -> VectorSource:
        return HDF5Source(
            shape=source.shape, root_dir=root_dir, h5py_params=dict(dtype=source.dtype)
        ).add_vectors(source.as_tensor(), keys=source.keys, write=write)

    def __init__(
        self,
        shape: Union[torch.Size, Tuple[int, ...]],
        root_dir: Path,
        h5py_params: Mapping[str, Any] = {},
    ) -> None:
        super().__init__()
        root_dir.mkdir(parents=True, exist_ok=True)
        self.root_dir = root_dir
        data_path = root_dir / "vectors" / "data.h5"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        self.h5_file = h5py.File(data_path, mode="w")

        h5py_params = {**_default_h5py_params, **h5py_params}

        self.data: h5py.Dataset = self.h5_file.create_dataset(
            "data",
            shape=shape,
            **h5py_params,
        )
        self.h5_file.attrs["last_index"] = 0  # Tracking the last index used
        self._keys2offset = BiMap(x=[], y=[])

    @property
    def keys(self) -> Sequence[str]:
        return list(self._keys2offset.x)

    @property
    def shape(self) -> torch.Size:
        return self.data.shape

    @property
    def dtype(self) -> torch.dtype:
        return torch.dtype(self.data.dtype)

    def __getitem__(
        self, index: Union[int, Sequence[int], Tuple[Union[int, slice]]]
    ) -> torch.Tensor:
        if isinstance(index, int):
            return torch.as_tensor(self.data[index])

        if isinstance(index, torch.Tensor):
            index = index.detach().cpu().numpy()

        def to_np(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            elif isinstance(x, np.ndarray):
                return x
            elif isinstance(x, int):
                return x
            elif isinstance(x, slice):
                return x
            elif isinstance(x, Sequence) and all(
                isinstance(i, (int, torch.Tensor)) for i in x
            ):
                return np.array(x)
            else:
                return x

        if isinstance(index, Iterable):
            if isinstance(index, np.ndarray) or (
                isinstance(index, Iterable)
                and all(isinstance(i, (int, torch.Tensor, np.integer)) for i in index)
            ):
                # the index is an array or a sequence of integers, we add another dimension to
                # make it compatible with the sub-indexing below
                index = [index]

            # convert all indices to numpy arrays
            index = [to_np(sub_index) for sub_index in index]

            fancy_indexing_dim = [
                i
                for i, sub_index in enumerate(index)
                if isinstance(sub_index, np.ndarray)
            ]
            if len(fancy_indexing_dim) > 1:
                raise ValueError(
                    "Only one dimension can be indexed with an array at a time (HDF5 allows only one fancy index)"
                )
            fancy_indexing_dim = fancy_indexing_dim[0] if fancy_indexing_dim else None

            if fancy_indexing_dim is not None:
                # if we are in a fancy indexing situation it is important to sort the index
                fancy_index = index[fancy_indexing_dim]

                # hdf5 does not support duplicates...`
                if len(fancy_index) != len(np.unique(fancy_index)):
                    # TODO: fix this inefficient access
                    temp_data = []
                    temp_index = list(index)
                    for i in fancy_index:
                        temp_index[fancy_indexing_dim] = int(i)
                        temp_data.append(torch.as_tensor(self.data[tuple(temp_index)]))

                    return torch.stack(temp_data, dim=fancy_indexing_dim)

                sort_idx = np.argsort(fancy_index)
                index[fancy_indexing_dim] = fancy_index[sort_idx]
                was_sorted = not np.array_equal(fancy_index, index[fancy_indexing_dim])

            # index is now sorted (if needed) for hdf5 compatibility
            data = self.data[tuple(index)]

            if fancy_indexing_dim and was_sorted:
                data = data.take(sort_idx.argsort(), axis=fancy_indexing_dim)

            return torch.as_tensor(data)

        return torch.as_tensor(self.data[index])

    def __len__(self) -> int:
        return self.data.shape[0]

    def as_tensor(self, device: torch.device = "cpu") -> torch.Tensor:
        return torch.as_tensor(np.asarray(self.data), device=device)

    def __eq__(self, __value: HDF5Source) -> bool:
        assert isinstance(
            __value, HDF5Source
        ), f"Expected {HDF5Source}, got {type(__value)}"
        return torch.allclose(self.data, __value.data)

    def add_vectors(
        self,
        vectors: torch.Tensor,
        keys: Optional[Sequence[str]] = None,
        write: bool = False,
    ) -> VectorSource:
        if vectors.ndim == 1:
            vectors = vectors.unsqueeze(0)

        # assert (keys is None) == (
        #     len(self._keys2offset) == 0
        # ), "Keys must be provided only if the source already has keys"
        if keys is not None:
            assert len(keys) == vectors.size(
                0
            ), "Keys must have the same length as vectors"
            self._keys2offset.add_all(
                x=keys,
                y=range(len(self._keys2offset), len(self._keys2offset) + len(keys)),
            )

        if self.h5_file.attrs["last_index"] + vectors.size(0) > self.data.shape[0]:
            self.data.resize(
                (
                    self.h5_file.attrs["last_index"] + vectors.size(0),
                    *self.data.shape[1:],
                )
            )

        self.data[
            self.h5_file.attrs["last_index"] : self.h5_file.attrs["last_index"]
            + vectors.size(0)
        ] = vectors.detach().cpu().numpy()
        self.h5_file.attrs["last_index"] += vectors.size(0)

        if write:
            self.h5_file.flush()
            self._save_mapping()

        return self

    def _save_mapping(self):
        self._keys2offset.save_to_disk(self.root_dir / "vectors" / "mapping.tsv")

    def to_tensor_source(self, device: torch.device = "cpu") -> TensorSource:
        return TensorSource(vectors=self.as_tensor(device=device), keys=self.keys)

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
            raise KeyError(f"Key {key} not found in {self._keys2offset}") from None

    # def select(self, indices: Sequence[int]) -> VectorSource:
    #     return HDF5Source(shape=(len(indices), self.data.shape[1]), root_dir=self.root_dir).add_vectors(
    #         self[indices], keys=[self.keys[i] for i in indices]
    #     )


class SearchSource(VectorSource):
    @classmethod
    def from_source(
        cls,
        source: VectorSource,
        metric_fn: SearchMetric,
        factory_string: str = "Flat",
        transform: Optional[Transform] = None,
        name: Optional[str] = None,
    ) -> SearchSource:
        new_source = cls.create(
            num_dimensions=source.shape[1],
            metric_fn=metric_fn,
            factory_string=factory_string,
            transform=transform,
            name=name,
        )
        new_source.add_vectors(vectors=source.as_tensor(), keys=source.keys)

        return new_source

    def __init__(
        self,
        backend_index: BackendIndex,
        metric_fn: SearchMetric,
        transform: Transform,
        name: Optional[str] = None,
        key2offset: Optional[Mapping[str, int]] = None,
    ) -> None:
        self.backend_index: BackendIndex = backend_index
        self._metric_fn: SearchMetric = metric_fn
        self._transform: Transform = transform
        self._name: Optional[str] = name or "SearchSource"
        self._key2offset: Mapping[str, int] = (
            key2offset if key2offset is not None else {}
        )
        self._offset2key: Mapping[int, str] = {
            offset: key for key, offset in self._key2offset.items()
        }

    @property
    def shape(self) -> torch.Size:
        return torch.Size([self.num_elements, self.num_dimensions])

    def __getitem__(
        self, index: Union[int, Sequence[int], slice, np.integer, torch.long]
    ) -> torch.Tensor:
        if isinstance(index, int):
            return self.get_vector(query_offset=index, return_tensors=True)
        elif isinstance(index, slice):
            return self.get_vectors(
                query_offsets=range(*index.indices(self.num_elements)),
                return_tensors=True,
            )
        elif isinstance(index, Sequence):
            return self.get_vectors(query_offsets=index, return_tensors=True)
        elif isinstance(index, (np.integer, torch.long)):
            return self.get_vector(query_offset=int(index), return_tensors=True)
        else:
            raise NotImplementedError(f"Index type {type(index)} not supported")

    def as_tensor(self, device: torch.device = "cpu") -> torch.Tensor:
        return self.get_vectors(
            query_offsets=range(self.num_elements), return_tensors=True
        ).to(device)

    def _add_mapping(self, key: str, offset: int):
        assert key not in self._key2offset, f"Vector ID {key} already exists"
        assert offset not in self._offset2key, f"Vector offset {offset} already exists"
        self._key2offset[key] = offset
        self._offset2key[offset] = key

    @property
    def name(self) -> Optional[str]:
        return self._name

    @classmethod
    def create(
        cls,
        num_dimensions: int,
        metric_fn: SearchMetric,
        factory_string: str = "Flat",
        transform: Optional[Transform] = None,
        name: Optional[str] = None,
    ) -> None:
        assert num_dimensions > 0, "Number of dimensions must be greater than 0"
        assert isinstance(
            metric_fn, SearchMetric
        ), f"Metric must be one of {SearchMetric}"
        if transform is not None and metric_fn.transformation is not None:
            # TODO: support Transform.compose or similar
            raise NotImplementedError(
                "transform and metric_fn.transformation cannot be both not None"
            )

        transform = transform or metric_fn.transformation

        index: _faiss.Index = _faiss.index_factory(
            num_dimensions, factory_string, metric_fn.backend_metric
        )
        return cls(
            backend_index=index, metric_fn=metric_fn, transform=transform, name=name
        )

    def __contains__(self, key: str) -> bool:
        return key in self._key2offset

    def __len__(self) -> int:
        return self.num_elements

    def get_vector_by_key(self, key: str) -> torch.Tensor:
        return self.get_vector(query_key=key, return_tensors=True)

    @property
    def keys(self) -> Sequence[str]:
        return list(self._key2offset.keys())

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self._name}|{self.backend_index}|{self.metric_fn.key}|{self.transform}|{self.metadata})"

    @property
    def num_dimensions(self) -> int:
        return self.backend_index.d

    @property
    def num_elements(self) -> int:
        return self.backend_index.ntotal

    @property
    def metric_fn(self) -> SearchMetric:
        return self._metric_fn

    @property
    def transform(self) -> Transform:
        return self._transform

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "metric": self.metric_fn.name,
            "transform": self.transform,
            "name": self.name,
        }

    def add_vector(
        self,
        vector: torch.Tensor,
        key: Optional[str] = None,
    ) -> int:
        # TODO: without a key/offset check here, we can end up adding vectors and then failing to map it properly
        assert vector.ndim == 1, "Vector must be 1-dimensional"
        assert (
            vector.shape[0] == self.num_dimensions
        ), f"Vector must have {self.num_dimensions} dimensions"

        vector = vector.unsqueeze(dim=0)
        vector = vector.detach().cpu()

        if self.transform is not None:
            vector = self.transform(vector)

        if key is not None:
            self._add_mapping(key=key, offset=self.num_elements)

        self.backend_index.add(vector.numpy())

        return self.num_elements - 1

    def add_vectors(
        self,
        vectors: torch.Tensor,
        keys: Optional[Sequence[str]] = None,
    ) -> Sequence[int]:
        if vectors.ndim == 1:
            vectors = vectors.unsqueeze(dim=0)

        assert vectors.ndim == 2, "vectors must be 2-dimensional"
        assert (
            vectors.shape[1] == self.num_dimensions
        ), f"Vectors must have {self.num_dimensions} dimensions"
        assert (
            keys is None or len(keys) == 0 or len(keys) == vectors.shape[0]
        ), "Must provide a key for each vector"

        start_id = self.num_elements

        if self.transform is not None:
            vectors = self.transform(vectors)

        vectors = vectors.cpu().detach().numpy()

        self.backend_index.add(vectors)

        if keys is not None:
            for key, offset in zip(keys, range(start_id, start_id + vectors.shape[0])):
                self._add_mapping(key=key, offset=offset)

        return list(range(start_id, start_id + vectors.shape[0]))

    # def get_distance(
    #     self,
    #     x: torch.Tensor,
    #     y: torch.Tensor,
    # ) -> float:
    #     assert x.ndim == 1, "Vector must be 1-dimensional"
    #     assert y.ndim == 1, "Vector must be 1-dimensional"
    #     assert x.shape[0] == self.num_dimensions, f"Vector must have {self.num_dimensions} dimensions"
    #     assert y.shape[0] == self.num_dimensions, f"Vector must have {self.num_dimensions} dimensions"

    #     x = x.cpu()
    #     y = y.cpu()

    #     if self.transform is not None:
    #         x = self.transform(x)
    #         y = self.transform(y)

    #     if isinstance(x, torch.Tensor):
    #         x = x.cpu().detach().numpy()
    #     if isinstance(y, torch.Tensor):
    #         y = y.cpu().detach().numpy()
    #     return _faiss.pairwise_distances()

    def search_knn(
        self,
        k: int,
        *,
        query_offsets: Optional[Sequence[int]] = None,
        query_vectors: Optional[Union[np.ndarray, torch.Tensor]] = None,
        query_keys: Optional[Sequence[str]] = None,
        transform: bool = False,
        return_keys: bool = False,
        metric_fn: Optional[SearchMetric] = None,
        **kwargs,
    ):
        """Performs k-nearest neighbors search based on the provided query.

        Args:
            k (int): The number of nearest neighbors to retrieve.
            query_offsets (Optional[Sequence[int]], optional): The offsets of the queries.
                Exactly one of `query_offsets`, `query_vectors`, or `query_keys` must be provided.
                Defaults to None.
            query_vectors (Optional[Union[np.ndarray, torch.Tensor]], optional): The query vectors.
                Exactly one of `query_offsets`, `query_vectors`, or `query_keys` must be provided.
                Defaults to None.
            query_keys (Optional[Sequence[str]], optional): The keys of the queries.
                Exactly one of `query_offsets`, `query_vectors`, or `query_keys` must be provided.
                Defaults to None.
            transform (bool): Whether to apply transformation to the query vectors.
            return_keys (bool): Whether to return the keys of the nearest neighbors.

        Returns:
            The nearest neighbors based on the provided query.

        Raises:
            AssertionError: If none or more than one of query_offsets, query_vectors, or query_keys are provided.
        """
        assert (
            sum(x is not None for x in [query_offsets, query_vectors, query_keys]) == 1
        ), "Must provide exactly one of query_offsets, query_vectors, or query_keys"

        assert (
            metric_fn is None or self._metric_fn == metric_fn
        ), "Metric function must match the source metric function. Source metric function is {self._metric_fn}, but provided metric function is {metric_fn}"

        if query_offsets is not None:
            return self._search_by_offsets(
                query_offsets=query_offsets, k=k, return_keys=return_keys
            )
        elif query_vectors is not None:
            return self._search_by_vectors(
                query_vectors=query_vectors,
                k=k,
                transform=transform,
                return_keys=return_keys,
            )
        elif query_keys is not None:
            return self._search_by_keys(
                query_keys=query_keys, k=k, return_keys=return_keys
            )

    def search_range(
        self,
        radius: float,
        *,
        query_offsets: Optional[Sequence[int]] = None,
        query_vectors: Optional[Union[np.ndarray, torch.Tensor]] = None,
        query_keys: Optional[Sequence[str]] = None,
        transform: bool = False,
        return_keys: bool = False,
        sort: bool = True,
        metric_fn: Optional[SearchMetric] = None,
    ):
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
            sort (bool, optional): Whether to sort the search results. Defaults to True.

        Returns:
            The search results based on the provided query type.
        """
        if not (
            sum(x is not None for x in [query_offsets, query_vectors, query_keys]) == 1
        ):
            raise ValueError(
                "Must provide exactly one of query_offsets, query_vectors, or query_keys"
            )

        if metric_fn is not None and metric_fn != self._metric_fn:
            raise ValueError(
                f"Metric function must match the source metric function. Source metric function is {self._metric_fn}, but provided metric function is {metric_fn}"
            )

        if query_offsets is not None:
            return self._search_by_offsets_range(
                query_offsets=query_offsets, radius=radius, sort=sort
            )
        elif query_vectors is not None:
            return self._search_by_vectors_range(
                query_vectors=query_vectors,
                radius=radius,
                transform=transform,
                return_keys=return_keys,
                sort=sort,
            )
        elif query_keys is not None:
            return self._search_by_keys_range(
                query_keys=query_keys, radius=radius, return_keys=return_keys, sort=sort
            )

    def get_vector(
        self,
        query_offset: Optional[int] = None,
        query_key: Optional[str] = None,
        return_tensors: bool = False,
    ) -> Union[np.ndarray, torch.Tensor]:
        assert (
            sum(x is not None for x in [query_offset, query_key]) == 1
        ), "Must provide exactly one of query_offset, or query_key"

        if query_offset is not None:
            return self._get_vector_by_offset(
                offset=query_offset, return_tensors=return_tensors
            )

        elif query_key is not None:
            return self._get_vector_by_key(key=query_key, return_tensors=return_tensors)

    def _get_vector_by_offset(
        self, offset: int, return_tensors: bool = False
    ) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(offset, np.int64):
            offset = int(offset)
        assert offset < self.num_elements, f"offset {offset} does not exist"
        result = self.backend_index.reconstruct(offset)
        if return_tensors:
            result = torch.as_tensor(result)
        return result

    def _get_vector_by_key(
        self, key: str, return_tensors: bool = False
    ) -> Union[np.ndarray, torch.Tensor]:
        assert self._key2offset, "No keys have been added to this index"
        offset = self._key2offset[key]
        return self._get_vector_by_offset(offset=offset, return_tensors=return_tensors)

    def get_vectors(
        self,
        query_offsets: Optional[Sequence[int]] = None,
        query_keys: Optional[Sequence[str]] = None,
        return_tensors: bool = False,
    ) -> Union[np.ndarray, torch.Tensor]:
        assert (
            sum(x is not None for x in [query_offsets, query_keys]) == 1
        ), "Must provide exactly one of query_offsets, or query_keys"

        if query_offsets is not None:
            return self._get_vectors_by_offsets(
                offsets=query_offsets, return_tensors=return_tensors
            )

        elif query_keys is not None:
            return self._get_vectors_by_key(
                keys=query_keys, return_tensors=return_tensors
            )

    def _get_vectors_by_offsets(
        self, offsets: Sequence[int], return_tensors: bool = False
    ) -> Union[np.ndarray, torch.Tensor]:
        assert all(
            offset < self.num_elements for offset in offsets
        ), f"Some of these offsets do not exist: {offsets}"
        result = self.backend_index.reconstruct_batch(offsets)

        if return_tensors:
            result = torch.as_tensor(result)

        return result

    def get_vectors_by_key(self, keys: Sequence[str]) -> torch.Tensor:
        return self._get_vectors_by_key(keys=keys, return_tensors=True)

    def _get_vectors_by_key(
        self, keys: Sequence[str], return_tensors: bool = False
    ) -> Union[np.ndarray, torch.Tensor]:
        offsets = [self._key2offset[key] for key in keys]
        return self._get_vectors_by_offsets(
            offsets=offsets, return_tensors=return_tensors
        )

    def _search_by_keys(
        self, query_keys: List[str], k, return_keys: bool = False
    ) -> SearchResult:
        query_vectors = self._get_vectors_by_key(keys=query_keys, return_tensors=False)

        return self._search_by_vectors(query_vectors, k=k, return_keys=return_keys)

    def _search_by_keys_range(
        self,
        query_keys: List[str],
        radius: float,
        return_keys: bool = False,
        sort: bool = True,
    ) -> SearchResult:
        query_vectors = self._get_vectors_by_key(query_keys, return_tensors=False)

        return self._search_by_vectors_range(
            query_vectors, radius=radius, return_keys=return_keys, sort=sort
        )

    def _search_by_vectors(
        self,
        query_vectors: Union[np.ndarray, torch.Tensor],
        k: int,
        transform: bool = False,
        return_keys: bool = False,
    ) -> SearchResult:
        if query_vectors.ndim == 1:
            query_vectors = query_vectors[None, :]
        assert (
            query_vectors.shape[1] == self.num_dimensions
        ), f"query_vectors must have {self.num_dimensions} dimensions"

        assert (
            not transform or self.transform is not None
        ), "Cannot transform vectors without a transform!"
        if transform:
            query_vectors = self.transform(query_vectors)

        if isinstance(query_vectors, torch.Tensor):
            query_vectors = query_vectors.cpu().detach().numpy()

        distances, offsets = self.backend_index.search(query_vectors, k)

        keys = (
            [
                [self._offset2key[offset] for offset in item_offsets]
                for item_offsets in offsets
            ]
            if return_keys
            else None
        )

        if distances.shape[0] == 1:
            assert offsets.shape[0] == 1
            distances = np.squeeze(distances, axis=0)
            offsets = np.squeeze(offsets, axis=0)
            keys = keys[0] if keys is not None else None

        return SearchResult(
            distances=distances,
            offsets=offsets,
            keys=keys,
        )

    def _search_by_vectors_range(
        self,
        query_vectors: Union[np.ndarray, torch.Tensor],
        radius: float,
        transform: bool = False,
        return_keys: bool = False,
        sort: bool = True,
    ) -> SearchResult:
        assert (
            query_vectors.shape[-1] == self.num_dimensions
        ), f"query_vectors must have {self.num_dimensions} dimensions"

        if query_vectors.ndim == 1:
            query_vectors = query_vectors[None, :]

        assert (
            not transform or self.transform is not None
        ), "Cannot transform vectors without a transform!"
        if transform:
            query_vectors = self.transform(query_vectors)

        if isinstance(query_vectors, torch.Tensor):
            query_vectors = query_vectors.cpu().detach().numpy()

        lims, distances, offsets = self.backend_index.range_search(
            query_vectors, radius
        )
        split_distances = []
        split_offsets = []
        for i in range(len(lims) - 1):
            i_distances = distances[lims[i] : lims[i + 1]]
            i_offsets = offsets[lims[i] : lims[i + 1]]
            # sort by distance (Faiss returns unsorted results)
            if sort:
                sorted_idx = np.argsort(i_distances)
                if _faiss.is_similarity_metric(self._metric_fn.backend_metric):
                    sorted_idx = sorted_idx[::-1]

                i_distances = i_distances[sorted_idx]
                i_offsets = i_offsets[sorted_idx]

            split_distances.append(i_distances)
            split_offsets.append(i_offsets)

        keys = (
            [
                [self._offset2key[offset] for offset in item_offsets]
                for item_offsets in split_offsets
            ]
            if return_keys
            else None
        )
        if len(split_distances) == 1:
            assert len(split_offsets) == 1
            split_distances = split_distances[0]
            split_offsets = split_offsets[0]
            keys = keys[0] if keys is not None else None

        return SearchResult(
            distances=split_distances,
            offsets=split_offsets,
            keys=keys,
        )

    def _search_by_offsets(
        self, query_offsets: Sequence[int], k: int, return_keys: bool = False
    ) -> SearchResult:
        query_vectors = self.get_vectors(
            query_offsets=query_offsets, return_tensors=False
        )
        return self._search_by_vectors(
            query_vectors=query_vectors, k=k, transform=False, return_keys=return_keys
        )

    def _search_by_offsets_range(
        self, query_offsets: Sequence[int], radius: float, sort: bool = True
    ) -> SearchResult:
        query_vectors = self.get_vectors(
            query_offsets=query_offsets, return_tensors=False
        )
        return self._search_by_vectors_range(
            query_vectors=query_vectors, radius=radius, transform=False, sort=sort
        )

    # TODO: query for farthest neighbors https://gist.github.com/mdouze/c7653aaa8c3549b28bad75bd67543d34#file-demo_farthest_l2-ipynb

    # def resize(self, new_size: int) -> None:
    #     self.backend_index.resize(new_size=new_size)

    def save_to_disk(self, target_path: Union[str, Path]) -> None:
        target_path = Path(target_path)
        assert target_path.is_dir(), f"Target path {target_path} must be a directory"
        target_path.mkdir(parents=True, exist_ok=True)

        data = _faiss.serialize_index(self.backend_index)
        # _faiss.write_index(self.backend_index, str(filename))

        metadata = self.metadata
        # metadata = json.dumps(metadata, indent=4, default=lambda x: x.key if isinstance(x, SearchMetric) else x)

        key2offset = pd.DataFrame(self._key2offset.items(), columns=["key", "offset"])

        # key2offset
        key2offset = key2offset.to_csv(
            target_path / "key2offset.tsv", sep="\t", index=False, encoding="utf-8"
        )

        # metadata
        metadata = json.dumps(metadata, indent=4, default=lambda x: x.__dict__)
        (target_path / "metadata.json").write_text(metadata, encoding="utf-8")

        # index
        data = data.tobytes()
        (target_path / "index.bin").write_bytes(data)

    @classmethod
    def load_from_disk(cls, path: Path) -> SearchSource:
        if not path.is_dir():
            raise ValueError(f"Path {path} is not a directory")

        # key2offset
        key2offset = pd.read_csv(
            path / "key2offset.tsv", sep="\t", index_col="key"
        ).to_dict()["offset"]

        # metadata
        metadata = json.load((path / "metadata.json").open("r", encoding="utf-8"))

        # index
        backend_index = _faiss.read_index(str(path / "index.bin"))

        metric_fn: SearchMetric = SearchMetric[metadata["metric"]]

        return cls(
            backend_index=backend_index,
            metric_fn=metric_fn,
            transform=metadata["transform"] or metric_fn.transformation,
            name=metadata["name"],
            key2offset=key2offset,
        )
