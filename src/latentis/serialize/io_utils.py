from __future__ import annotations

import hashlib
import json
from abc import abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

import torch
from torch import nn


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
    _METADATA_FILE_NAME: str = "metadata.json"

    @classmethod
    def hash_metadata(cls, metadata: Mapping[str, Any]) -> str:
        hash_obj = hashlib.sha256(
            json.dumps(metadata, default=lambda o: o.__dict__, sort_keys=True).encode(encoding="utf-8")
        )
        return hash_obj.hexdigest()[:10]

    @property
    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def save_to_disk(self, parent_dir: Path, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load_from_disk(cls, path: Path, *args, **kwargs) -> SerializableMixin:
        raise NotImplementedError

    @property
    def version(self) -> int:
        raise NotImplementedError

    def __repr__(self) -> str:
        # public_properties = {k: v for k, v in self.metadata.items() if not k.startswith("_")}
        return f"{self.__class__.__name__}(id={self.item_id[:5]}, metadata={self.metadata})"

    @property
    def hash(self):
        return SerializableMixin.hash_metadata(self.metadata)
