from __future__ import annotations

from abc import abstractmethod

import pandas as pd
import torch

try:
    # be ready for 3.10 when it drops
    from enum import StrEnum as PythonStrEnum
except ImportError:
    from backports.strenum import StrEnum as PythonStrEnum

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Sequence, Union

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
    @classmethod
    @abstractmethod
    def load_properties(cls, path: Path) -> Properties:
        raise NotImplementedError

    @abstractmethod
    def properties(self) -> Dict[str, Any]:
        raise NotImplementedError


Properties = Mapping[str, Any]


class IndexMixin:
    @abstractmethod
    def add_item(self, item: IndexSerializableMixin, save_args: Mapping[str, Any] = None) -> str:
        raise NotImplementedError

    @abstractmethod
    def add_items(self, items: Sequence[IndexSerializableMixin], save_args: Mapping[str, Any] = None) -> Sequence[str]:
        raise NotImplementedError

    @abstractmethod
    def remove_item(self, item_key: Optional[str] = None, **properties: Any) -> str:
        raise NotImplementedError

    @abstractmethod
    def remove_items(self, item_key: Optional[str] = None, **properties: Any) -> Sequence[str]:
        raise NotImplementedError

    @abstractmethod
    def load_item(self, item_key: Optional[str] = None, **properties: Any) -> IndexSerializableMixin:
        raise NotImplementedError

    @abstractmethod
    def load_items(self, item_key: Optional[str] = None, **properties: Any) -> Mapping[str, IndexSerializableMixin]:
        raise NotImplementedError

    @abstractmethod
    def get_item_path(self, item_key: Optional[str] = None, **properties: Any) -> Path:
        raise NotImplementedError

    @abstractmethod
    def get_items_path(self, item_key: Optional[str] = None, **properties: Any) -> Mapping[str, Path]:
        raise NotImplementedError

    @abstractmethod
    def get_item(self, item_key: Optional[str] = None, **properties: Any) -> Mapping[str, Properties]:
        raise NotImplementedError

    @abstractmethod
    def get_items(self, item_key: Optional[str] = None, **properties: Any) -> Mapping[str, Properties]:
        raise NotImplementedError

    @abstractmethod
    def get_items_df(self, item_key: Optional[str] = None, **properties: Any) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def get_item_key(self, **properties: Any) -> str:
        raise NotImplementedError

    @abstractmethod
    def clear(self):
        raise NotImplementedError


class MetadataMixin:
    _METADATA_FILE_NAME: str = "metadata.json"

    @property
    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        raise NotImplementedError
