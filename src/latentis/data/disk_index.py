import importlib
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping, Optional, Type

from latentis.io_utils import load_json, save_json
from latentis.types import SerializableMixin

_INDEX_FILE: str = "index.json"


class DiskIndex(SerializableMixin):
    def save_to_disk(self, *args, **kwargs):
        index_path = self.root_path / _INDEX_FILE
        save_json(self._index, index_path)

        info = {
            "item_class": self._item_class.__name__,
            "item_class_module": self._item_class.__module__,
        }
        save_json(info, self.root_path / "info.json")

    @classmethod
    def load_from_disk(cls, path: Path, *args, **kwargs) -> SerializableMixin:
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

    def __init__(self, root_path: Path, item_class: Type[SerializableMixin]):
        self.root_path = root_path
        self._item_class = item_class
        self._index: Mapping[str, Mapping[str, Any]] = {}

    def add_item(self, item: SerializableMixin, item_key: Optional[str] = None, properties: Mapping[str, Any] = None):
        if item_key is None:
            item_key = uuid.uuid4().hex

        if item_key in self._index:
            raise KeyError(f"Key {item_key} already exists in index")

        self._index[item_key] = {
            "properties": properties or {},
            "version": item.version,
        }

        item.save_to_disk(self.root_path / item_key)
        self.save_to_disk()

    def remove_item_by_key(self, item_key: str):
        if item_key not in self._index:
            raise KeyError(f"Key {item_key} does not exist in index")

        del self._index[item_key]
        shutil.rmtree(self.root_path / item_key)
        self.save_to_disk()

    def remove_items_by_properties(self, **properties: Mapping[str, Any]):
        removed_items = []

        for key, item in self.index.items():
            if all(item["properties"].get(p, None) == v for p, v in properties.items()):
                del self._index[key]
                self.save_to_disk()
                shutil.rmtree(self.root_path / key)
                removed_items.append(key)

        return removed_items

    def get_item_by_key(self, item_key: str) -> SerializableMixin:
        return self._item_class.load_from_disk(self.root_path / item_key)

    def get_items_by_properties(self, **properties: Mapping[str, Any]) -> SerializableMixin:
        result = []

        for key, item in self._index.items():
            if all(item["properties"].get(p, None) == v for p, v in properties.items()):
                result.append(self._item_class.load_from_disk(self.root_path / key))

        return result

    def get_item_by_properties(self, **properties: Mapping[str, Any]) -> SerializableMixin:
        result = None

        for key, item in self._index.items():
            if all(item["properties"].get(p, None) == v for p, v in properties.items()):
                if result is not None:
                    raise KeyError(f"Multiple items matching {properties} found")

                result = self._item_class.load_from_disk(self.root_path / key)

        if result is None:
            raise KeyError(f"No item matching {properties} found")

        return result

    def clear(self):
        self._index = {}
        shutil.rmtree(self.root_path)
        self.root_path.mkdir()
        self.save_to_disk()

    def __len__(self):
        return len(self._index)
