from __future__ import annotations

from abc import abstractmethod
from enum import auto
from pathlib import Path

import torch

from latentis.data import DATA_DIR
from latentis.serialize.disk_index import DiskIndex
from latentis.serialize.io_utils import IndexableMixin, load_json, save_json
from latentis.space import LatentSpace
from latentis.types import Properties, StrEnum

_CORRESPONDENCE_DIR = DATA_DIR / "correspondences"
_PROPERTIES_FILE_NAME = "properties.json"


class _CorrespondenceMetadata(StrEnum):
    _VERSION = auto()
    _TYPE = auto()


class Correspondence(IndexableMixin):
    def __init__(self, **properties):
        properties = properties or {}
        properties[_CorrespondenceMetadata._VERSION] = self.version
        properties[_CorrespondenceMetadata._TYPE] = Correspondence.__name__
        self._properties = properties.copy()

    @property
    def version(self) -> 0:
        return 0

    @property
    def properties(self) -> Properties:
        return self._properties

    @classmethod
    def load_properties(cls, correspondence_path: Path) -> Properties:
        metadata = load_json(correspondence_path / _PROPERTIES_FILE_NAME)
        return metadata

    def save_to_disk(self, target_path: Path) -> None:
        target_path.mkdir(parents=True, exist_ok=True)
        save_json(self.properties, target_path / _PROPERTIES_FILE_NAME)
        # TODO: serialize save correspondence

    @classmethod
    def load_from_disk(cls, path: Path) -> Correspondence:
        # TODO: deserialize the correspondence
        properties = cls.load_properties(path)
        correspondence = cls.__new__(cls)
        correspondence._properties = properties
        return correspondence

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

    # def random_subset(self, factor: Union[int, float], seed: int):
    #     n = int(len(self.x2y) * factor) if 0 < factor <= 1 else factor
    #     x_ids = self.get_x_ids()
    #     y_ids = self.get_y_ids()
    #     subset = torch.randperm(x_ids.size(0), generator=torch.Generator().manual_seed(seed))[:n]

    #     return TensorCorrespondence(x2y=torch.stack([x_ids[subset, :], y_ids[subset, :]], dim=-1))


try:
    correspondences = DiskIndex.load_from_disk(path=_CORRESPONDENCE_DIR)
except FileNotFoundError:
    correspondences = DiskIndex(root_path=_CORRESPONDENCE_DIR, item_class=Correspondence)
    correspondences.save_to_disk()
