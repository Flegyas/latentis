import hashlib
import importlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Type

from latentis.io_utils import load_json, save_json
from latentis.types import IndexSerializableMixin, SerializableMixin

_INDEX_FILE: str = "index.json"

Properties = Mapping[str, Any]


class DiskIndex(SerializableMixin):
    def __init__(self, root_path: Path, item_class: Type[IndexSerializableMixin]):
        self.root_path = root_path
        self._item_class = item_class
        self._index: Mapping[str, Mapping[str, Any]] = {}

    @property
    def version(self) -> str:
        return -42

    def save_to_disk(self):
        self.root_path.mkdir(exist_ok=True)
        index_path = self.root_path / _INDEX_FILE
        save_json(self._index, index_path)

        info = {
            "item_class": self._item_class.__name__,
            "item_class_module": self._item_class.__module__,
            "version": self.version,
        }
        save_json(info, self.root_path / "info.json")

    @classmethod
    def load_from_disk(cls, path: Path, *args, **kwargs) -> IndexSerializableMixin:
        info = load_json(path / "info.json")
        module = importlib.import_module(info["item_class_module"])
        item_class = getattr(module, info["item_class"])

        index_path = path / _INDEX_FILE
        index = load_json(index_path)

        instance = cls.__new__(cls)
        instance._index = index
        instance.root_path = path
        instance._item_class = item_class

        return instance

    @staticmethod
    def _compute_item_key(item: IndexSerializableMixin) -> str:
        item_properties = item.properties()
        if len(item_properties) == 0:
            raise ValueError("Item does not have any properties")
        hash_object = hashlib.sha256(json.dumps(item_properties, sort_keys=True).encode(encoding="utf-8"))
        return hash_object.hexdigest()

    def _resolve_items(self, item_key: Optional[str] = None, **properties: Any) -> Sequence[str]:
        if item_key is not None:
            if item_key not in self._index:
                raise KeyError(f"Key {item_key} does not exist in index")
            return [item_key]
        else:
            result = []
            for key, item in self._index.items():
                if all(item.get(p, None) == v for p, v in properties.items()):
                    result.append(key)
            return result

    def _resolve_item(self, item_key: Optional[str] = None, **properties: Any) -> str:
        items = self._resolve_items(item_key=item_key, **properties)
        if len(items) == 0:
            raise KeyError(f"No items matching {properties} found")
        elif len(items) > 1:
            raise ValueError(f"Multiple items matching {properties} found, colliding on {items}")
        return items[0]

    def _remove_item_by_key(self, item_key: str) -> None:
        if item_key not in self._index:
            raise KeyError(f"Key {item_key} does not exist in index")

        del self._index[item_key]
        shutil.rmtree(self.root_path / item_key)
        self.save_to_disk()

    def add_item(
        self,
        item: IndexSerializableMixin,
        save_args: Mapping[str, Any] = None,
    ) -> str:
        item_key = self._compute_item_key(item)

        if item_key in self._index:
            raise KeyError(f"Key {item_key} already exists in index")

        primary_keys = item.properties()
        if len(primary_keys) == 0:
            raise ValueError("Item does not have any properties")

        self._index[item_key] = item.properties()

        item.save_to_disk(self.root_path / item_key, **(save_args or {}))
        self.save_to_disk()
        return item_key

    def add_items(self, items: Sequence[IndexSerializableMixin], save_args: Mapping[str, Any] = None) -> Sequence[str]:
        item_keys = [self._compute_item_key(item) for item in items]

        # Avoid adding any of the items if any of the keys already exist
        if any(item_key in self._index for item_key in item_keys):
            raise KeyError("One of the keys already exists in index")

        if any(len(item.properties()) == 0 for item in items):
            raise ValueError("One of the items does not have any properties")

        for item, item_key in zip(items, item_keys):
            self._index[item_key] = item.properties()
            item.save_to_disk(self.root_path / item_key, **(save_args or {}))

        self.save_to_disk()
        return item_keys

    def remove_item(self, item_key: Optional[str] = None, **properties: Any) -> str:
        item_to_remove = self._resolve_item(item_key=item_key, **properties)
        self._remove_item_by_key(item_to_remove)
        return item_to_remove

    def remove_items(self, item_key: Optional[str] = None, **properties: Any) -> Sequence[str]:
        items_to_remove = self._resolve_items(item_key=item_key, **properties)
        for item in items_to_remove:
            self._remove_item_by_key(item)
        return items_to_remove

    def load_item(self, item_key: Optional[str] = None, **properties: Any) -> IndexSerializableMixin:
        item_to_load = self._resolve_item(item_key=item_key, **properties)
        return self._item_class.load_from_disk(self.root_path / item_to_load)

    def load_items(self, item_key: Optional[str] = None, **properties: Any) -> Mapping[str, IndexSerializableMixin]:
        items_to_load = self._resolve_items(item_key=item_key, **properties)
        return {item: self._item_class.load_from_disk(self.root_path / item) for item in items_to_load}

    def get_item_path(self, item_key: Optional[str] = None, **properties: Any) -> Path:
        item_to_load = self._resolve_item(item_key=item_key, **properties)
        return self.root_path / item_to_load

    def get_items_path(self, item_key: Optional[str] = None, **properties: Any) -> Mapping[str, Path]:
        items_to_load = self._resolve_items(item_key=item_key, **properties)
        return {item: self.root_path / item for item in items_to_load}

    def get_item(self, item_key: Optional[str] = None, **properties: Any) -> Mapping[str, Properties]:
        item_to_get = self._resolve_item(item_key=item_key, **properties)
        return {item_to_get: self._index[item_to_get]}

    def get_items(self, item_key: Optional[str] = None, **properties: Any) -> Mapping[str, Properties]:
        items_to_get = self._resolve_items(item_key=item_key, **properties)
        return {item: self._index[item] for item in items_to_get}

    def get_item_key(self, **properties: Any) -> str:
        item_to_load = self._resolve_item(item_key=None, **properties)
        return item_to_load

    def clear(self):
        self._index = {}
        shutil.rmtree(self.root_path)
        self.root_path.mkdir()
        self.save_to_disk()

    def __len__(self):
        return len(self._index)
