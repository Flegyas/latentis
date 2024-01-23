from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from enum import auto
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch
from datasets import DatasetDict
from torch import nn
from torch.utils.data import Dataset

from latentis.data import DATA_DIR
from latentis.serialize.disk_index import DiskIndex
from latentis.serialize.io_utils import MetadataMixin, SerializableMixin, load_json, save_json
from latentis.space import LatentSpace
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


# make it json serializable
@dataclass(frozen=True)
class Feature:
    col_name: str
    data_type: DataType
    properties: Mapping[FeatureProperty, str] = field(default_factory=lambda: {})

    def __hash__(self):
        return hash(self.col_name)


@dataclass(frozen=True)
class FeatureMapping:
    source_col: str
    target_col: str


class DatasetView(torch.utils.data.Dataset):
    def __init__(
        self,
        split: str,
        latentis_dataset: LatentisDataset,
        encodings_key: Optional[Sequence[str]],
        hf_x_keys: Optional[Sequence[str]],
        hf_y_keys: Optional[Sequence[str]] = ("label",),
    ):
        super().__init__()
        if isinstance(encodings_key, str):
            encodings_key = [encodings_key]
        if isinstance(hf_x_keys, str):
            hf_x_keys = [hf_x_keys]
        if isinstance(hf_y_keys, str):
            hf_y_keys = [hf_y_keys]

        self.data = latentis_dataset.hf_dataset[split]
        self.encodings_key = encodings_key or []
        self.hf_x_keys = hf_x_keys or []
        self.hf_y_keys = hf_y_keys or []

        self.spaces = [latentis_dataset.encodings.load_item(item_key=key) for key in encodings_key]

    def __getitem__(self, idx: int) -> Mapping[str, Any]:
        sample = {
            "encodings_key": [space[idx] for space in self.spaces],
            "hf_x_keys": [self.data[idx][key] for key in self.hf_x_keys],
            "hf_y_keys": [self.data[idx][key] for key in self.hf_y_keys],
        }
        return {key: value for key, value in sample.items() if value is not None and len(value) > 0}

    def __len__(self) -> int:
        return len(self.data)


class LatentisDataset(SerializableMixin, MetadataMixin):
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
            feature.col_name in hf_dataset[split].column_names for feature in features for split in hf_dataset.keys()
        ), f"Specified features not in all splits of dataset {hf_dataset}"
        assert 0 < perc <= 1, f"Percentage {perc} not in (0, 1]"

        self._name: str = name
        self._hf_dataset: DatasetDict = hf_dataset
        self._id_column: str = id_column
        self._features: Sequence[Feature] = features
        self._perc: float = perc
        self._properties: Mapping[DatasetProperty, Any] = properties or {}
        self._root_dir: Path = parent_dir / name
        self.encodings = DiskIndex(self._root_dir / "encodings", item_class=LatentSpace)

    def save_to_disk(self):
        target_path = self._root_dir
        if target_path.exists():
            raise FileExistsError(f"Destination {self._root_dir} is not empty!")

        self._hf_dataset.save_to_disk(target_path / "hf_dataset")

        # Copy the encodings directory, if needed
        if (target_path / "encodings").exists():
            shutil.copy(self._root_dir / "encodings", target_path / "encodings")

        save_json(self.metadata, target_path / self._METADATA_FILE_NAME, indent=4)

        self.encodings.save_to_disk()

    @classmethod
    def load_from_disk(
        cls,
        path: Path,
    ) -> "LatentisDataset":
        assert (
            path / cls._METADATA_FILE_NAME
        ).exists(), f"Metadata file {path / cls._METADATA_FILE_NAME} does not exist! Are you sure about the parameters?"

        metadata = load_json(path / cls._METADATA_FILE_NAME)

        hf_dataset = DatasetDict.load_from_disk(path / "hf_dataset")

        properties = metadata["properties"]
        features = [Feature(**feature) for feature in metadata["features"]]

        dataset = LatentisDataset.__new__(cls)

        dataset._name: str = metadata["name"]
        dataset._hf_dataset: DatasetDict = hf_dataset
        dataset._id_column: str = metadata["id_column"]
        dataset._features: Sequence[Feature] = features
        dataset._perc: float = metadata["perc"]
        dataset._root_dir: Path = path
        dataset._properties = properties
        dataset.encodings = DiskIndex.load_from_disk(path=path / "encodings")

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
            if feature.col_name == col_name:
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

    def add_decoder(self, encoding_key: str, decoder: nn.Module):
        raise NotImplementedError

    def load_decoders(self, encodings: Sequence[str]) -> Mapping[str, nn.Module]:
        raise NotImplementedError

    def add_encoding(
        self,
        item: LatentSpace,
        save_source_model: bool,
    ) -> LatentSpace:
        # TODO: add consistency check to make sure that the item is compatible with the dataset
        try:
            existing_space = self.encodings.load_item(**item.properties)
            existing_space.add_vectors(vectors=item.vectors, keys=item.keys)

            # TODO: This is a hack, we are bypassing the index
            target_path = self.encodings.get_item_path(**item.properties)
            existing_space.save_to_disk(
                target_path,
                save_vector_source=True,
                save_info=False,
                save_source_model=save_source_model,
                save_decoders=False,
            )
        except KeyError:
            return self.encodings.add_item(
                item=item,
                save_args={
                    "save_vector_source": True,
                    "save_info": True,
                    "save_source_model": save_source_model,
                    "save_decoders": True,
                },
            )

    def add_encodings(
        self,
        items: Sequence[LatentSpace],
        save_source_model: bool,
    ) -> Sequence[str]:
        for space in items:
            self.add_encoding(item=space, save_source_model=save_source_model)

    def get_dataset_view(
        self,
        split: str,
        encodings_key: Sequence[str],
        hf_x_keys: Sequence[str],
        hf_y_keys: Sequence[str],
    ) -> Dataset:
        return DatasetView(
            latentis_dataset=self,
            encodings_key=encodings_key,
            hf_x_keys=hf_x_keys,
            hf_y_keys=hf_y_keys,
            split=split,
        )


if __name__ == "__main__":
    dataset = LatentisDataset.load_from_disk(DATA_DIR / "imdb")
    print(dataset.encodings.get_items_df())
