from abc import abstractmethod
from pathlib import Path
from typing import Union

import torch
from torch._tensor import Tensor

from latentis.data.dataset import LatentisDataset
from latentis.serialize.disk_index import DiskIndex
from latentis.serialize.io_utils import IndexableMixin, SerializableMixin
from latentis.space import LatentSpace

# WikimatrixCorrespondence(source_dataset="en", target_dataset="fr", source_id=0)


# IdentityCorrespondence(dataset)(source_id=0)


class CorrespondenceIndex:
    def __init__(self, dataset: LatentisDataset):
        self.dataset = dataset
        self.correspondences = DiskIndex(dataset)


class Correspondence(IndexableMixin):
    def __init__(self, **properties):
        self._properties = properties

    @property
    def properties(self):
        return self._properties

    def load_properties(self, path: Path):
        raise NotImplementedError

    def save_to_disk(self, parent_dir: Path, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def load_from_disk(cls, path: Path, *args, **kwargs) -> SerializableMixin:
        raise NotImplementedError

    @abstractmethod
    def get_x_ids(self) -> torch.Tensor:
        return self.x2y[:, 0]

    def get_y_ids(self) -> torch.Tensor:
        return self.x2y[:, 1]

    @abstractmethod
    def split(self):
        pass

    def add_noise(self, x: LatentSpace, y: LatentSpace):
        pass

    def random_subset(self, factor: Union[int, float], seed: int):
        n = int(len(self.x2y) * factor) if 0 < factor <= 1 else factor
        x_ids = self.get_x_ids()
        y_ids = self.get_y_ids()
        subset = torch.randperm(x_ids.size(0), generator=torch.Generator().manual_seed(seed))[:n]

        return TensorCorrespondence(x2y=torch.stack([x_ids[subset, :], y_ids[subset, :]], dim=-1))


class TensorCorrespondence(Correspondence):
    def __init__(self, x2y: torch.Tensor):
        super().__init__()
        if x2y.ndim != 2:
            raise ValueError(f"Expected x2y to have ndim=2, got {x2y.ndim}")
        if x2y.shape[1] != 2:
            raise ValueError(f"Expected x2y to be (n, 2), got {x2y.shape}")

        self.x2y: torch.Tensor = x2y

    def get_x_ids(self) -> torch.Tensor:
        return self.x2y[:, 0]

    def get_y_ids(self) -> torch.Tensor:
        return self.x2y[:, 1]


class IdentityCorrespondence(Correspondence):
    def __init__(self, n_samples: int):
        super().__init__()
        self.n_samples: int = n_samples

    def get_x_ids(self) -> Tensor:
        return torch.arange(self.n_samples)

    def get_y_ids(self) -> Tensor:
        return torch.arange(self.n_samples)
