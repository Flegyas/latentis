from __future__ import annotations

from abc import abstractmethod

import torch

try:
    # be ready for 3.10 when it drops
    from enum import StrEnum as PythonStrEnum
except ImportError:
    from backports.strenum import StrEnum as PythonStrEnum

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Union

if TYPE_CHECKING:
    from latentis.space import LatentSpace

    Space = Union[LatentSpace, torch.Tensor]

StrEnum = PythonStrEnum


class SerializableMixin:
    @abstractmethod
    def save_to_disk(self, parent_dir: Path, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load_from_disk(cls, path: Path, *args, **kwargs) -> SerializableMixin:
        raise NotImplementedError

    @property
    @abstractmethod
    def version(self) -> int:
        raise NotImplementedError


class IndexSerializableMixin(SerializableMixin):
    @abstractmethod
    def properties(self) -> Dict[str, Any]:
        raise NotImplementedError


class MetadataMixin:
    _METADATA_FILE_NAME: str = "metadata.json"

    @property
    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        raise NotImplementedError
