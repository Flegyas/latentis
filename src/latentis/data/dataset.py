from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import auto
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from datasets import DatasetDict
from torch.utils.data import DataLoader

from latentis.data import DATA_DIR
from latentis.serialize.io_utils import MetadataMixin, SerializableMixin, load_json, save_json
from latentis.types import StrEnum

pylogger = logging.getLogger(__name__)


class DataType(StrEnum):
    TEXT = auto()
    IMAGE = auto()
    LABEL = auto()


class FeatureProperty(StrEnum):
    LANGUAGE = auto()
    FINE_GRAINED = auto()


class DatasetProperty(StrEnum):
    pass


@dataclass(frozen=True)
class Feature:
    name: str
    data_type: DataType
    properties: Mapping[FeatureProperty, str] = field(default_factory=lambda: {})

    def __hash__(self):
        return hash((self.name, self.data_type, frozenset(self.properties.items())))


@dataclass(frozen=True)
class FeatureMapping:
    source_col: str
    target_col: str


# class DatasetView(torch.utils.data.Dataset):
#     def __init__(
#         self,
#         latentis_dataset: DatasetView,
#         encodings_key: Optional[Union[str, Sequence[str]]],
#         hf_x_keys: Optional[Union[str, Sequence[str]]],
#         hf_y_keys: Optional[Union[str, Sequence[str]]] = ("label",),
#     ):
#         super().__init__()
#         if isinstance(encodings_key, str):
#             encodings_key = [encodings_key]
#         if isinstance(hf_x_keys, str):
#             hf_x_keys = [hf_x_keys]
#         if isinstance(hf_y_keys, str):
#             hf_y_keys = [hf_y_keys]

#         self.spaces = [space_index.load_item(item_key=key) for key in encodings_key]
#         spaces_split = set(space.split for space in self.spaces)
#         if len(spaces_split) > 1:
#             raise ValueError(f"Spaces {encodings_key} are not all from the same split!")
#         self.split = spaces_split.pop()

#         self.data = latentis_dataset.hf_dataset[self.split]
#         self.encodings_key = encodings_key or []
#         self.hf_x_keys = hf_x_keys or []
#         self.hf_y_keys = hf_y_keys or []

#     def __getitem__(self, idx: int) -> Mapping[str, Any]:
#         sample = {
#             "encodings_key": [space[idx] for space in self.spaces],
#             "hf_x_keys": [self.data[idx][key] for key in self.hf_x_keys],
#             "hf_y_keys": [self.data[idx][key] for key in self.hf_y_keys],
#         }
#         return {key: value for key, value in sample.items() if value is not None and len(value) > 0}


#     def __len__(self) -> int:
#         return len(self.data)
class DatasetView(SerializableMixin, MetadataMixin):
    pass


class HFDatasetView(DatasetView):
    def __init__(
        self,
        name: str,
        hf_dataset: DatasetDict,
        id_column: str,
        features: Sequence[Feature],
        perc: float = 1,
        properties: Optional[Mapping[str, Any]] = None,
        parent_dir: Path = DATA_DIR,
    ):
        super().__init__()
        assert isinstance(hf_dataset, DatasetDict), f"Expected {DatasetDict}, got {type(hf_dataset)}"
        assert len(set(features)) == len(features), f"Features {features} contain duplicates!"
        assert len(features) > 0, f"Features {features} must not be empty!"
        assert all(
            id_column in hf_dataset[split].column_names for split in hf_dataset.keys()
        ), f"ID column {id_column} not in all splits of dataset {hf_dataset}"
        assert all(
            feature.name in hf_dataset[split].column_names for feature in features for split in hf_dataset.keys()
        ), f"Specified features not in all splits of dataset {hf_dataset}"
        assert 0 < perc <= 1, f"Percentage {perc} not in (0, 1]"

        self._name: str = name
        self._hf_dataset: DatasetDict = hf_dataset
        self._id_column: str = id_column
        self._features: Sequence[Feature] = features
        self._perc: float = perc
        self._properties: Mapping[DatasetProperty, Any] = properties or {}
        self._root_dir: Path = parent_dir / name

    def save_to_disk(self):
        target_path = self._root_dir
        if target_path.exists():
            raise FileExistsError(f"Destination {self._root_dir} is not empty!")

        self._hf_dataset.save_to_disk(target_path / "hf_dataset")

        save_json(self.metadata, target_path / self._METADATA_FILE_NAME, indent=4)

    @classmethod
    def load_from_disk(
        cls,
        path: Path,
        load_hf_dataset: bool = True,
    ) -> "DatasetView":
        assert (
            path / cls._METADATA_FILE_NAME
        ).exists(), f"Metadata file {path / cls._METADATA_FILE_NAME} does not exist! Are you sure about the parameters?"

        metadata = load_json(path / cls._METADATA_FILE_NAME)

        hf_dataset = None
        if load_hf_dataset:
            hf_dataset = DatasetDict.load_from_disk(path / "hf_dataset")

        properties = metadata["properties"]
        features = [Feature(**feature) for feature in metadata["features"]]

        dataset = DatasetView.__new__(cls)

        dataset._name = metadata["name"]
        dataset._hf_dataset = hf_dataset
        dataset._id_column = metadata["id_column"]
        dataset._features[Feature] = features
        dataset._perc = metadata["perc"]
        dataset._root_dir = path
        dataset._properties = properties

        return dataset

    @property
    def name(self) -> str:
        return self._name

    @property
    def hf_dataset(self) -> DatasetDict:
        return self._hf_dataset

    @property
    def perc(self) -> float:
        return self._perc

    @property
    def id_column(self) -> str:
        return self._id_column

    @property
    def features(self) -> Sequence[Feature]:
        return self._features

    @property
    def root_dir(self) -> Path:
        return self._root_dir

    def get_feature(self, col_name: str) -> Optional[Feature]:
        for feature in self._features:
            if feature.target_name == col_name:
                return feature

        return None

    @property
    def metadata(self) -> Mapping[str, Any]:
        return {
            "name": self._name,
            "id_column": self._id_column,
            "features": tuple(self._features),
            "perc": self._perc,
            "timestamp": str(datetime.now()),
            "timestamp_ms": int(datetime.now().timestamp()),
            "properties": self._properties,
        }

    def __repr__(self):
        return f"{self.__class__.__name__}(features={self._features}, perc={self._perc}, metadata={self.metadata})"

    def select(
        self,
        feature_keys: Sequence[str],
    ):
        return DatasetView(
            name=self._name,
            hf_dataset=self._hf_dataset,
            id_column=self._id_column,
            features=[feature for feature in self._features if feature.target_name in feature_keys],
            perc=self._perc,
            properties=self._properties,
            parent_dir=self._root_dir,
        )

    def get_dataloader(
        self,
        space_id: Sequence[str],
        hf_x_keys: Sequence[str],
        hf_y_keys: Sequence[str],
        batch_size: int,
        shuffle: bool,
        num_workers: int = 0,
        **kwargs,
    ):
        return DataLoader(
            DatasetView(
                latentis_dataset=self,
                encodings_key=space_id,
                hf_x_keys=hf_x_keys,
                hf_y_keys=hf_y_keys,
            ),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            **kwargs,
        )
