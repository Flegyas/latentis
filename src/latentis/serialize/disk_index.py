import importlib
import shutil
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Type

import pandas as pd

from latentis.serialize.io_utils import IndexableMixin, SerializableMixin, load_json, save_json
from latentis.types import Properties


class DiskIndex(SerializableMixin):
    def __init__(self, root_path: Path, item_class: Type[IndexableMixin]):
        self.root_path = root_path
        self._item_class = item_class
        self._index: Mapping[str, Mapping[str, Any]] = {}

    @property
    def version(self) -> str:
        return -42

    def save_to_disk(self):
        self.root_path.mkdir(exist_ok=True)

        info = {
            "item_class": self._item_class.__name__,
            "item_class_module": self._item_class.__module__,
            "version": self.version,
        }

        # TODO see if the disk index should implement the indexable mixin
        save_json(info, self.root_path / "info.json")

    @classmethod
    def _read_index(cls, path: Path, item_class: Type[IndexableMixin]) -> Mapping[str, Mapping[str, Any]]:
        item_dirs = [item_dir for item_dir in path.iterdir() if item_dir.is_dir()]
        index = {}
        for item_dir in item_dirs:
            item_key = item_dir.name
            index[item_key] = item_class.load_properties(item_dir)
            assert item_key == item_class.id_from_properties(index[item_key])

        return index

    @classmethod
    def load_from_disk(cls, path: Path, *args, **kwargs) -> IndexableMixin:
        info = load_json(path / "info.json")
        module = importlib.import_module(info["item_class_module"])
        item_class = getattr(module, info["item_class"])

        instance = cls.__new__(cls)
        instance._index = cls._read_index(path, item_class=item_class)
        instance.root_path = path
        instance._item_class = item_class

        return instance

    def _resolve_items(self, item_key: Optional[str] = None, **properties: Any) -> Sequence[str]:
        candidates_items = list(self._index.keys())

        if item_key is not None:
            candidates_items = [x for x in self._index.keys() if x.startswith(item_key)]
            if len(candidates_items) == 0:
                raise KeyError(f"No items with key prefix '{item_key}' found")

        return [
            key for key in candidates_items if all(self._index[key].get(p, None) == v for p, v in properties.items())
        ]

    def _resolve_item(self, item_key: Optional[str] = None, **properties: Any) -> str:
        items = self._resolve_items(item_key=item_key, **properties)
        if len(items) == 0:
            raise KeyError(f"No items with key prefix '{item_key}' matching {properties} found")
        elif len(items) > 1:
            raise ValueError(f"Multiple items with key prefix '{item_key}' matching {properties} found: {items}")
        return items[0]

    def _remove_item_by_key(self, item_key: str) -> None:
        if item_key not in self._index:
            raise KeyError(f"Key {item_key} does not exist in index")

        del self._index[item_key]
        shutil.rmtree(self.root_path / item_key)
        self.save_to_disk()

    def add_item(
        self,
        item: IndexableMixin,
        save_args: Mapping[str, Any] = None,
    ) -> str:
        item_key = item.item_id

        if item_key in self._index:
            raise KeyError(f"Key {item_key} already exists in index")

        primary_keys = item.properties
        if len(primary_keys) == 0:
            raise ValueError("Item does not have any properties")

        self._index[item_key] = item.properties

        item.save_to_disk(self.root_path / item_key, **(save_args or {}))
        self.save_to_disk()
        return item_key

    def add_items(self, items: Sequence[IndexableMixin], save_args: Mapping[str, Any] = None) -> Sequence[str]:
        item_keys = [item.item_id for item in items]

        # Avoid adding any of the items if any of the keys already exist
        if any(item_key in self._index for item_key in item_keys):
            raise KeyError("One of the keys already exists in index")

        if any(len(item.properties) == 0 for item in items):
            raise ValueError("One of the items does not have any properties")

        for item, item_key in zip(items, item_keys):
            self._index[item_key] = item.properties
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

    def load_item(self, item_key: Optional[str] = None, **properties: Any) -> IndexableMixin:
        item_to_load = self._resolve_item(item_key=item_key, **properties)
        return self._item_class.load_from_disk(self.root_path / item_to_load)

    def load_items(self, item_key: Optional[str] = None, **properties: Any) -> Mapping[str, IndexableMixin]:
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

    def get_items_df(self, item_key: Optional[str] = None, **properties: Any) -> pd.DataFrame:
        items = self.get_items(item_key=item_key, **properties)
        return pd.DataFrame.from_dict(items, orient="index").reset_index(names="item_key")

    def get_item_key(self, item_key: Optional[str] = None, **properties: Any) -> str:
        item_to_load = self._resolve_item(item_key=item_key, **properties)
        return item_to_load

    def get_items_key(self, item_key: Optional[str] = None, **properties: Any) -> str:
        item_to_load = self._resolve_items(item_key=item_key, **properties)
        return item_to_load

    def clear(self):
        self._index = {}
        shutil.rmtree(self.root_path)
        self.root_path.mkdir()
        self.save_to_disk()

    def __len__(self):
        return len(self._index)
