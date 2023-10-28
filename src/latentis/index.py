from __future__ import annotations

from enum import auto
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import voyager as _voyager

try:
    # be ready for 3.10 when it drops
    from enum import StrEnum
except ImportError:
    from backports.strenum import StrEnum


class Similarity(StrEnum):
    EUCLIDEAN = auto()
    COSINE = auto()
    INNERPRODUCT = auto()


class DataType(StrEnum):
    FLOAT8 = auto()
    FLOAT32 = auto()
    E4M3 = auto()


BackendIndex = _voyager.Index

_LATENTIS_SIMILARITY2BACKEND: Dict[Similarity, _voyager.Space] = {
    Similarity.EUCLIDEAN: _voyager.Space.Euclidean,
    Similarity.COSINE: _voyager.Space.Cosine,
    Similarity.INNERPRODUCT: _voyager.Space.InnerProduct,
}
_BACKEND2LATENTIS_SIMILARITY: Dict[_voyager.Space, Similarity] = {v: k for k, v in _LATENTIS_SIMILARITY2BACKEND.items()}

_LATENTIS_DATA_TYPE2BACKEND: Dict[DataType, _voyager.StorageDataType] = {
    DataType.FLOAT8: _voyager.StorageDataType.Float8,
    DataType.FLOAT32: _voyager.StorageDataType.Float32,
    DataType.E4M3: _voyager.StorageDataType.E4M3,
}
_BACKEND2LATENTIS_DATA_TYPE: Dict[_voyager.StorageDataType, DataType] = {
    v: k for k, v in _LATENTIS_DATA_TYPE2BACKEND.items()
}


class Index:
    def __init__(self, backend_index: BackendIndex) -> None:
        self.index: BackendIndex = backend_index

    @classmethod
    def create(
        cls,
        similarity_fn: Similarity,
        num_dimensions: int,
        serialization_data_type: DataType = DataType.FLOAT32,
    ) -> None:
        index: _voyager.Index = _voyager.Index(
            _LATENTIS_SIMILARITY2BACKEND[similarity_fn],
            num_dimensions=num_dimensions,
            storage_data_type=_LATENTIS_DATA_TYPE2BACKEND[serialization_data_type],
        )
        return cls(backend_index=index)

    def __contains__(self, id: int) -> bool:
        return id in self.index

    def __len__(self) -> int:
        return len(self.index)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.index})"

    @property
    def ids(self) -> FrozenSet:
        return self.index.ids

    @property
    def max_elements(self) -> int:
        return self.index.max_elements

    @property
    def num_dimensions(self) -> int:
        return self.index.num_dimensions

    @property
    def num_elements(self) -> int:
        return self.index.num_elements

    @property
    def similarity_fn(self) -> Similarity:
        return _BACKEND2LATENTIS_SIMILARITY[self.index.space]

    @property
    def serialization_data_type(self) -> DataType:
        return _BACKEND2LATENTIS_DATA_TYPE[self.index.storage_data_type]

    def add_item(
        self,
        vector: Union[torch.Tensor, np.ndarray],
        id: Optional[int] = None,
    ) -> int:
        if isinstance(vector, torch.Tensor):
            vector = vector.cpu().detach().numpy()
        return self.index.add_item(vector=vector, id=id)

    def add_items(
        self,
        vectors: Union[torch.Tensor, np.ndarray],
        ids: Optional[List[int]] = None,
        num_threads: int = -1,
    ) -> Sequence[int]:
        if isinstance(vectors, torch.Tensor):
            vectors = vectors.cpu().detach().numpy()
        return self.index.add_items(vectors=vectors, ids=ids, num_threads=num_threads)

    def as_bytes(self) -> bytes:
        return self.index.as_bytes()

    def get_distance(
        self,
        a: Union[torch.Tensor, np.ndarray],
        b: Union[torch.Tensor, np.ndarray],
    ) -> float:
        if isinstance(a, torch.Tensor):
            a = a.cpu().detach().numpy()
        if isinstance(b, torch.Tensor):
            b = b.cpu().detach().numpy()
        return self.index.get_distance(a=a, b=b)

    def get_vector(self, id: int) -> np.ndarray:
        return self.index.get_vector(id=id)

    def get_vectors(self, ids: np.ndarray) -> np.ndarray:
        return self.index.get_vectors(ids=ids)

    @classmethod
    def load(cls, filename: Union[str, Path]) -> Index:
        backend_index = _voyager.Index.load(filename=str(filename))
        return cls(backend_index=backend_index)

    def mark_deleted(self, id: int) -> None:
        self.index.mark_deleted(id=id)

    def query(
        self, vectors: Union[torch.Tensor, np.ndarray], k: int = 1, num_threads: int = -1, query_ef: int = -1
    ) -> Tuple[np.ndarray[Any, np.dtype[np.uint64]], np.ndarray[Any, np.dtype[np.float32]]]:
        return self.index.query(vectors=vectors, k=k, num_threads=num_threads, query_ef=query_ef)

    def resize(self, new_size: int) -> None:
        self.index.resize(new_size=new_size)

    def save(self, filename: Union[str, Path]) -> None:
        self.index.save(output_path=str(filename))

    def unmark_deleted(self, id: int) -> None:
        self.index.unmark_deleted(id=id)
