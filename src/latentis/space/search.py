from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Sequence, Union

import faiss as _faiss
import numpy as np
import torch
import torch.nn.functional as F

from latentis.transform._abstract import Transform


# https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances
class SearchMetric(Enum):
    EUCLIDEAN = ("euclidean", _faiss.METRIC_L2)
    COSINE_SIM = ("cosine", _faiss.METRIC_INNER_PRODUCT, lambda x: F.normalize(x, p=2, dim=-1))
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
