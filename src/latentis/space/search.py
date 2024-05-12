from __future__ import annotations

import json
import tarfile
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import faiss as _faiss
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from latentis.transform._abstract import Transform


# https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances
class SearchMetric(Enum):
    EUCLIDEAN = ("euclidean", _faiss.METRIC_L2)
    COSINE = ("cosine", _faiss.METRIC_INNER_PRODUCT, lambda x: F.normalize(x, p=2, dim=-1))
    INNER_PRODUCT = ("inner_product", _faiss.METRIC_INNER_PRODUCT)
    L2 = ("l2", _faiss.METRIC_L2)

    # L1 = ("l1", _faiss.METRIC_L1)
    # LINF = ("linf", _faiss.METRIC_Linf)
    # LP = ("lp", _faiss.METRIC_Lp)
    # BRAY_CURTIS = ("bray_curtis", _faiss.METRIC_BrayCurtis)
    # CANBERRA = ("canberra", _faiss.METRIC_Canberra)
    # JENSEN_SHANNON = ("jensen_shannon", _faiss.METRIC_JensenShannon)
    # JACCARD = ("jaccard", _faiss.METRIC_Jaccard)
    # MAHALANOBIS https://gist.github.com/mdouze/6cc12fa967e5d9911580ef633e559476

    def __init__(self, name: str, backend_metric, transformation: Optional[Transform] = None) -> None:
        self.key: str = name
        self.backend_metric = backend_metric
        self.transformation = transformation


@dataclass
class SearchResult:
    distances: Union[np.ndarray, torch.Tensor]
    offsets: Union[np.ndarray, torch.Tensor]
    keys: Optional[Sequence[Sequence[str]]] = None

    def __iter__(self):
        items = [self.offsets, self.distances]
        if self.keys is not None:
            items.append(self.keys)

        return iter(zip(*items))

    def asdict(self) -> Dict[str, Any]:
        return {
            "offsets": self.offsets,
            "distances": self.distances,
            "keys": self.keys,
        }


# _LATENTIS_DATA_TYPE2BACKEND: Dict[DataType, _voyager.StorageDataType] = {
#     DataType.FLOAT8: _voyager.StorageDataType.Float8,
#     DataType.FLOAT32: _voyager.StorageDataType.Float32,
#     DataType.E4M3: _voyager.StorageDataType.E4M3,
# }

# class DataType(StrEnum):
#     FLOAT8 = auto()
#     FLOAT32 = auto()
#     E4M3 = auto()

BackendIndex = _faiss.Index


class SearchIndex:
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
        self._name: Optional[str] = name
        self._key2offset: Mapping[str, int] = key2offset if key2offset is not None else {}
        self._offset2key: Mapping[int, str] = {offset: key for key, offset in self._key2offset.items()}

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
        assert isinstance(metric_fn, SearchMetric), f"Metric must be one of {SearchMetric}"
        if transform is not None and metric_fn.transformation is not None:
            # TODO: support Transform.compose or similar
            raise NotImplementedError("transform and metric_fn.transformation cannot be both not None")

        transform = transform or metric_fn.transformation

        index: _faiss.Index = _faiss.index_factory(num_dimensions, factory_string, metric_fn.backend_metric)
        return cls(backend_index=index, metric_fn=metric_fn, transform=transform, name=name)

    def __contains__(self, key: str) -> bool:
        return key in self._key2offset

    def __len__(self) -> int:
        return self.num_elements

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self._name}|{self.backend_index}|{self.metric_fn.key}|{self.transform}|{self.metadata})"

    # @property
    # def ids(self) -> FrozenSet:
    #     return self.backend_index.ids

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

    # @property
    # def storage_data_type(self) -> DataType:
    #     return _BACKEND2LATENTIS_DATA_TYPE[self.index.storage_data_type]

    def add_vector(
        self,
        vector: torch.Tensor,
        key: Optional[str] = None,
    ) -> int:
        # TODO: without a key/offset check here, we can end up adding vectors and then failing to map it properly
        assert vector.ndim == 1, "Vector must be 1-dimensional"
        assert vector.shape[0] == self.num_dimensions, f"Vector must have {self.num_dimensions} dimensions"

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
        assert vectors.ndim == 2, "vectors must be 2-dimensional"
        assert vectors.shape[1] == self.num_dimensions, f"Vectors must have {self.num_dimensions} dimensions"
        assert keys is None or len(keys) == 0 or len(keys) == vectors.shape[0], "Must provide a key for each vector"

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

        if query_offsets is not None:
            return self._search_by_offsets(query_offsets=query_offsets, k=k, return_keys=return_keys)
        elif query_vectors is not None:
            return self._search_by_vectors(
                query_vectors=query_vectors, k=k, transform=transform, return_keys=return_keys
            )
        elif query_keys is not None:
            return self._search_by_keys(query_keys=query_keys, k=k, return_keys=return_keys)

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
        assert (
            sum(x is not None for x in [query_offsets, query_vectors, query_keys]) == 1
        ), "Must provide exactly one of query_offsets, query_vectors, or query_keys"

        if query_offsets is not None:
            return self._search_by_offsets_range(query_offsets=query_offsets, radius=radius, sort=sort)
        elif query_vectors is not None:
            return self._search_by_vectors_range(
                query_vectors=query_vectors, radius=radius, transform=transform, return_keys=return_keys, sort=sort
            )
        elif query_keys is not None:
            return self._search_by_keys_range(query_keys=query_keys, radius=radius, return_keys=return_keys, sort=sort)

    # def get_vector(self, offset: int, return_tensors: bool = False) -> Union[np.ndarray, torch.Tensor]:
    #     if isinstance(offset, np.int64):
    #         offset = int(offset)
    #     assert offset < self.num_elements, f"offset {offset} does not exist"
    #     result = self.backend_index.reconstruct(offset)
    #     if return_tensors:
    #         result = torch.as_tensor(result)
    #     return result

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
            return self._get_vector_by_offset(offset=query_offset, return_tensors=return_tensors)

        elif query_key is not None:
            return self._get_vector_by_key(key=query_key, return_tensors=return_tensors)

    def _get_vector_by_offset(self, offset: int, return_tensors: bool = False) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(offset, np.int64):
            offset = int(offset)
        assert offset < self.num_elements, f"offset {offset} does not exist"
        result = self.backend_index.reconstruct(offset)
        if return_tensors:
            result = torch.as_tensor(result)
        return result

    def _get_vector_by_key(self, key: str, return_tensors: bool = False) -> Union[np.ndarray, torch.Tensor]:
        assert self._key2offset, "No keys have been added to this index"
        offset = self._key2offset[key]
        return self._get_vector_by_offset(offset=offset, return_tensors=return_tensors)

    # def get_vectors(self, offsets: Sequence[int], return_tensors: bool = False) -> Union[np.ndarray, torch.Tensor]:
    #     assert all(offset < self.num_elements for offset in offsets), f"Some of these offsets do not exist: {offsets}"
    #     result = self.backend_index.reconstruct_batch(offsets)

    #     if return_tensors:
    #         result = torch.as_tensor(result)

    #     return result

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
            return self._get_vectors_by_offsets(offsets=query_offsets, return_tensors=return_tensors)

        elif query_keys is not None:
            return self._get_vectors_by_keys(keys=query_keys, return_tensors=return_tensors)

    def _get_vectors_by_offsets(
        self, offsets: Sequence[int], return_tensors: bool = False
    ) -> Union[np.ndarray, torch.Tensor]:
        assert all(offset < self.num_elements for offset in offsets), f"Some of these offsets do not exist: {offsets}"
        result = self.backend_index.reconstruct_batch(offsets)

        if return_tensors:
            result = torch.as_tensor(result)

        return result

    def _get_vectors_by_keys(
        self, keys: Sequence[str], return_tensors: bool = False
    ) -> Union[np.ndarray, torch.Tensor]:
        offsets = [self._key2offset[key] for key in keys]
        return self._get_vectors_by_offsets(offsets=offsets, return_tensors=return_tensors)

    def _search_by_keys(self, query_keys: List[str], k, return_keys: bool = False) -> SearchResult:
        query_vectors = self._get_vectors_by_keys(keys=query_keys, return_tensors=False)

        return self._search_by_vectors(query_vectors, k=k, return_keys=return_keys)

    def _search_by_keys_range(
        self, query_keys: List[str], radius: float, return_keys: bool = False, sort: bool = True
    ) -> SearchResult:
        query_vectors = self._get_vectors_by_keys(query_keys, return_tensors=False)

        return self._search_by_vectors_range(query_vectors, radius=radius, return_keys=return_keys, sort=sort)

    def _search_by_vectors(
        self, query_vectors: Union[np.ndarray, torch.Tensor], k: int, transform: bool = False, return_keys: bool = False
    ) -> SearchResult:
        if query_vectors.ndim == 1:
            query_vectors = query_vectors[None, :]
        assert (
            query_vectors.shape[1] == self.num_dimensions
        ), f"query_vectors must have {self.num_dimensions} dimensions"

        assert not transform or self.transform is not None, "Cannot transform vectors without a transform!"
        if transform:
            query_vectors = self.transform(query_vectors)

        if isinstance(query_vectors, torch.Tensor):
            query_vectors = query_vectors.cpu().detach().numpy()

        distances, offsets = self.backend_index.search(query_vectors, k)

        if distances.shape[0] == 1:
            assert offsets.shape[0] == 1
            distances = np.squeeze(distances, axis=0)
            offsets = np.squeeze(offsets, axis=0)

        return SearchResult(
            distances=distances,
            offsets=offsets,
            keys=[[self._offset2key[offset] for offset in item_offsets] for item_offsets in offsets]
            if return_keys
            else None,
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

        assert not transform or self.transform is not None, "Cannot transform vectors without a transform!"
        if transform:
            query_vectors = self.transform(query_vectors)

        if isinstance(query_vectors, torch.Tensor):
            query_vectors = query_vectors.cpu().detach().numpy()

        lims, distances, offsets = self.backend_index.range_search(query_vectors, radius)
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

        if len(split_distances) == 1:
            assert len(split_offsets) == 1
            split_distances = split_distances[0]
            split_offsets = split_offsets[0]

        return SearchResult(
            distances=split_distances,
            offsets=split_offsets,
            keys=[[self._offset2key[offset] for offset in item_offsets] for item_offsets in offsets]
            if return_keys
            else None,
        )

    def _search_by_offsets(self, query_offsets: Sequence[int], k: int, return_keys: bool = False) -> SearchResult:
        query_vectors = self.get_vectors(query_offsets=query_offsets, return_tensors=False)
        return self._search_by_vectors(query_vectors=query_vectors, k=k, transform=False, return_keys=return_keys)

    def _search_by_offsets_range(self, query_offsets: Sequence[int], radius: float, sort: bool = True) -> SearchResult:
        query_vectors = self.get_vectors(query_offsets=query_offsets, return_tensors=False)
        return self._search_by_vectors_range(query_vectors=query_vectors, radius=radius, transform=False, sort=sort)

    # TODO: query for farthest neighbors https://gist.github.com/mdouze/c7653aaa8c3549b28bad75bd67543d34#file-demo_farthest_l2-ipynb

    # def resize(self, new_size: int) -> None:
    #     self.backend_index.resize(new_size=new_size)

    def save(self, filename: Union[str, Path]) -> None:
        filename = Path(filename).with_suffix(".tar")
        filename.parent.mkdir(parents=True, exist_ok=True)

        data = _faiss.serialize_index(self.backend_index)
        # _faiss.write_index(self.backend_index, str(filename))

        metadata = self.metadata
        # metadata = json.dumps(metadata, indent=4, default=lambda x: x.key if isinstance(x, SearchMetric) else x)

        key2offset = pd.DataFrame(self._key2offset.items(), columns=["key", "offset"])

        with tarfile.open(filename, "w") as tar:
            # key2offset
            key2offset = key2offset.to_csv(sep="\t", index=False)
            tarinfo = tarfile.TarInfo("key2offset.tsv")
            tarinfo.size = len(key2offset)
            tar.addfile(tarinfo, BytesIO(key2offset.encode("utf-8")))

            # metadata
            metadata = json.dumps(metadata, indent=4, default=lambda x: x.__dict__)
            tarinfo = tarfile.TarInfo("metadata.json")
            tarinfo.size = len(metadata)
            tar.addfile(tarinfo, BytesIO(metadata.encode("utf-8")))

            # index
            data = data.tobytes()
            tarinfo = tarfile.TarInfo("index.bin")
            tarinfo.size = len(data)
            tar.addfile(tarinfo, BytesIO(data))

    @classmethod
    def load(cls, filename: Union[str, Path]) -> SearchIndex:
        filename = Path(filename)
        with tarfile.open(filename, "r") as tar:
            # key2offset
            tarinfo = tar.getmember("key2offset.tsv")
            f = tar.extractfile(tarinfo)
            key2offset = pd.read_csv(f, sep="\t", index_col="key").to_dict()["offset"]

            # metadata
            tarinfo = tar.getmember("metadata.json")
            f = tar.extractfile(tarinfo)
            metadata = json.load(f)

            # index
            tarinfo = tar.getmember("index.bin")
            f = tar.extractfile(tarinfo)
            data = f.read()
            data = np.frombuffer(data, dtype=np.uint8)
            backend_index = _faiss.deserialize_index(data)

            metric_fn: SearchMetric = SearchMetric[metadata["metric"]]

            return cls(
                backend_index=backend_index,
                metric_fn=metric_fn,
                transform=metadata["transform"] or metric_fn.transformation,
                name=metadata["name"],
                key2offset=key2offset,
            )
