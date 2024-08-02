from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from enum import auto
from pathlib import Path
from typing import Sequence

import torch

from latentis.data import DATA_DIR
from latentis.serialize.io_utils import SerializableMixin, load_json, save_json
from latentis.space import Space
from latentis.types import Metadata, StrEnum

_CORRESPONDENCE_DIR = DATA_DIR / "correspondences"
_METADATA_FILE_NAME = "metadata.json"


class _CorrespondenceMetadata(StrEnum):
    _VERSION = auto()
    _TYPE = auto()


@dataclass(frozen=True)
class PI:
    x_indices: torch.Tensor
    y_indices: torch.Tensor


class Correspondence(SerializableMixin):
    def __init__(self, **metadata):
        metadata = metadata or {}
        metadata[_CorrespondenceMetadata._VERSION] = self.version
        metadata[_CorrespondenceMetadata._TYPE] = self.__class__.__name__

        self._metadata = metadata.copy()

    @property
    def version(self) -> 0:
        return 0

    @property
    def metadata(self) -> Metadata:
        return self._metadata

    @classmethod
    def load_metadata(cls, correspondence_path: Path) -> Metadata:
        metadata = load_json(correspondence_path / _METADATA_FILE_NAME)
        return metadata

    def save_to_disk(self, target_path: Path) -> None:
        target_path.mkdir(parents=True, exist_ok=True)
        save_json(self.metadata, target_path / _METADATA_FILE_NAME)
        # TODO: serialize save correspondence

    @classmethod
    def load_from_disk(cls, path: Path) -> Correspondence:
        # TODO: deserialize the correspondence
        metadata = cls.load_metadata(path)
        correspondence = cls.__new__(cls)
        correspondence._metadata = metadata
        return correspondence

    @abstractmethod
    def match(self, x_keys: Sequence[str], y_keys: Sequence[str]) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def split(self):
        pass

    def add_noise(self, x: Space, y: Space):
        pass

    def subset(self, x_keys: Sequence[str], y_keys: Sequence[str], size: int, seed: int = 42) -> PI:
        raise NotImplementedError
