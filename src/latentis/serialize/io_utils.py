from __future__ import annotations

import hashlib
import json
from abc import abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch
from torch import nn

from latentis.types import Properties


# TODO: Handle versioning
def save_model(model: nn.Module, target_path: Path, version: int):
    torch.save(model, target_path)


def load_model(model_path: Path, version: int) -> nn.Module:
    return torch.load(model_path)


def _default_json(o):
    return o.__dict__


def save_json(
    obj: object,
    path: Path,
    indent: Optional[int] = 4,
    sort_keys: bool = True,
    default: Optional[Callable] = _default_json,
):
    with open(path, "w", encoding="utf-8") as fw:
        json.dump(obj, fw, indent=indent, sort_keys=sort_keys, default=default)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as fr:
        return json.load(fr)


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


class IndexableMixin(SerializableMixin):
    @classmethod
    def id_from_properties(cls, properties: Properties) -> str:
        hash_object = hashlib.sha256(json.dumps(properties, sort_keys=True).encode(encoding="utf-8"))
        return hash_object.hexdigest()

    @property
    def item_id(self) -> str:
        return IndexableMixin.id_from_properties(self.properties)

    @classmethod
    @abstractmethod
    def load_properties(cls, path: Path) -> Properties:
        raise NotImplementedError

    @property
    @abstractmethod
    def properties(self) -> Dict[str, Any]:
        raise NotImplementedError


class MetadataMixin:
    _METADATA_FILE_NAME: str = "metadata.json"

    @property
    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        raise NotImplementedError
