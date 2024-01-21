import json
import logging
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import auto
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import pyarrow.parquet as pq
import torch
from datasets import DatasetDict
from torch import nn

from latentis.data import DATA_DIR
from latentis.space import EncodingKey, LatentSpace
from latentis.types import MetadataMixin, SerializableMixin, StrEnum

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

    #
    # def load_encodings(self, encoding2splits: Mapping[str, Sequence[str]]) -> Mapping[str, Mapping[str, torch.Tensor]]:
    #     result = {}

    #     for encoding, splits in encoding2splits.items():
    #         split2dataset = {}
    #         for split in splits:
    #             encodings_path = (
    #                 self.parent_dir
    #                 / self.name
    #                 / LatentisDataset.get_key(dataset_name=self.name, perc=self.perc, properties=self.properties)
    #                 / "encodings"
    #                 / encoding
    #                 / split
    #             )

    #             if not encodings_path.exists():
    #                 raise FileNotFoundError

    #             split2dataset[split] = Dataset.load_from_disk(encodings_path)

    #         result[encoding] = DatasetDict(split2dataset)

    #     return result

    # DATASETS VERSION
    # def with_encodings(self, encodings: Sequence[str]) -> "LatentisDataset":
    #     result = self.dataset
    #     for encoding in encodings:
    #         encoding2split2data = self.load_encodings({encoding: self.dataset.keys()})

    #         for encoding, split2data in encoding2split2data.items():
    #             for split, split_data in split2data.items():
    #                 result[split] = datasets.concatenate_datasets([result[split], split_data], axis=1)

    #     return LatentisDataset(
    #         name=self.name,
    #         dataset=result,
    #         id_column=self.id_column,
    #         features=self.features,
    #         perc=self.perc,
    #         properties=self.properties,
    #         parent_dir=self.parent_dir,
    #     )

    def load_encodings(self, encoding2splits: Mapping[str, Sequence[str]]) -> Mapping[str, Mapping[str, torch.Tensor]]:
        result = {}

        for encoding, splits in encoding2splits.items():
            split2dataset = {}
            for split in splits:
                encodings_path = self._root_dir / "encodings" / encoding / split / "encodings.parquet"

                if not encodings_path.exists():
                    raise FileNotFoundError

                split2dataset[split] = pq.read_table(encodings_path)

            result[encoding] = split2dataset

        return result

    # def with_encodings(self, encodings: Sequence[str]) -> "LatentisDataset":
    #     # result = self.dataset

    #     encoding2split2data = self.load_encodings({encoding: self.dataset.keys() for encoding in encodings})

    #     split2all_data = {}

    #     for encoding, split2data in encoding2split2data.items():
    #         for split, data in split2data.items():
    #             if split not in split2all_data:
    #                 split2all_data[split] = data
    #             else:
    #                 split2all_data[split] = pa.concat_tables([split2all_data[split], data])

    #     return LatentisDataset(
    #         name=self.name,
    #         dataset=result,
    #         id_column=self.id_column,
    #         features=self.features,
    #         perc=self.perc,
    #         properties=self.properties,
    #         parent_dir=self.parent_dir,
    #     )

    def add_decoder(self, encoding_key: str, decoder: nn.Module):
        raise NotImplementedError

    def load_decoders(self, encodings: Sequence[str]) -> Mapping[str, nn.Module]:
        raise NotImplementedError

    def get_available_encodings(self, *features: Feature) -> Mapping[Feature, Mapping[str, Sequence[str]]]:
        """Scans the encodings directory and returns a mapping from feature to available encodings.

        Returns:
            Mapping[Feature, Mapping[str, Sequence[str]]]: Mapping from feature to split to available encodings.

        """
        features = features or self._features

        feature2split2model2encodings = {}

        for feature in features:
            assert feature in self._features, f"Feature {feature} not available in dataset {self._name}!"

            encodings_dir = self.root_dir / "encodings" / feature.col_name

            if not encodings_dir.exists():
                pylogger.warning(f"Feature {feature} has no encodings!")

            if not encodings_dir.exists():
                return {}

            split2model2encodings = defaultdict(dict)

            for split_dir in encodings_dir.iterdir():
                assert split_dir.is_dir(), f"Expected {split_dir} to be a directory!"
                split = split_dir.name
                for model_dir in split_dir.iterdir():
                    assert model_dir.is_dir(), f"Expected {model_dir} to be a directory!"
                    model = model_dir.name
                    encodings = list(model_dir.iterdir())
                    split2model2encodings[split][model] = [encoding.name for encoding in encodings]

            feature2split2model2encodings[feature] = split2model2encodings

        return feature2split2model2encodings

    # DATASETS VERSION
    # def add_encoding(self, encoding_key: str, split: str, id2encoding: Mapping[str, torch.Tensor]):
    #     encoding_dataset = Dataset.from_dict(
    #         {
    #             self.id_column: id2encoding.keys(),
    #             encoding_key: [encoding.numpy() for encoding in id2encoding.values()],
    #         }
    #     )

    #     target_path = (
    #         self.parent_dir
    #         / self.name
    #         / LatentisDataset.get_key(dataset_name=self.name, perc=self.perc, properties=self.properties)
    #         / "encodings"
    #         / encoding_key
    #         / split
    #     )

    #     if target_path.exists():
    #         current_dataset = Dataset.load_from_disk(target_path)
    #         encoding_dataset = datasets.concatenate_datasets([current_dataset, encoding_dataset], axis=0)

    #     if target_path.exists():
    #         # TODO: Highly inefficient
    #         temp_path = target_path.parent / (target_path.name + "_temp")
    #         encoding_dataset.save_to_disk(temp_path)
    #         shutil.rmtree(target_path)
    #         shutil.move(temp_path, target_path)
    #     else:
    #         encoding_dataset.save_to_disk(target_path)

    # SQLITE VERSION
    # def add_encoding(self, encoding_key: str, split: str, id2encoding: Mapping[str, torch.Tensor]):
    #     ids, encodings = zip(*id2encoding.items())
    #     data_list = [{self.id_column: sample_id, "encoding": encoding} for sample_id, encoding in zip(ids, encodings)]
    #     encoding_size: int = encodings[0].shape[0]

    #     target_path = (
    #         self.parent_dir
    #         / self.name
    #         / LatentisDataset.get_key(dataset_name=self.name, perc=self.perc, properties=self.properties)
    #         / "encodings"
    #         / encoding_key
    #         / split
    #     )
    #     target_path.mkdir(parents=True, exist_ok=True)

    #     db = SQLiteDatabase(
    #         path=str(target_path / "encodings.db"),
    #         name="encodings",
    #         schema={
    #             self.id_column: int,
    #             "encoding": dict(dtype=torch.float, size=(encoding_size,)),
    #         },
    #     )

    #     db.multi_insert(indices=ids, data_list=data_list, log=True)

    def get_encoding(
        self, feature: Feature, encoder_key: str, encoding_key: str, split: str, load_source_model: bool = False
    ) -> LatentSpace:
        target_path = self._root_dir / "encodings" / feature.col_name / split / encoder_key / encoding_key

        if not target_path.exists():
            raise FileNotFoundError(
                f"Encoding {encoding_key} for feature {feature} and split {split} not found! Path: {target_path}"
            )

        space = LatentSpace.load_from_disk(path=target_path, load_source_model=load_source_model)

        return space

    def add_encoding(
        self,
        vectors: torch.Tensor,
        keys: Sequence[str],
        encoding_key: EncodingKey,
        save_source_model: bool,
    ) -> LatentSpace:
        target_path = encoding_key.get_path(self.root_dir.parent)

        if target_path.exists():
            space = LatentSpace.load_from_disk(path=target_path, load_source_model=False)

            assert (
                encoding_key == space.encoding_key
            ), f"Encoding key of pre-existing space {space.encoding_key} does not match {encoding_key}"

            space.add_vectors(vectors=vectors, keys=keys)
            space.save_to_disk(
                target_path,
                save_vector_source=True,
                save_metadata=False,
                save_source_model=False,
                save_decoders=False,
            )
        else:
            space = LatentSpace(vector_source=vectors, keys=keys, encoding_key=encoding_key)
            space.save_to_disk(
                target_path,
                save_vector_source=True,
                save_metadata=True,
                save_source_model=True,
                save_decoders=True,
            )

        return space

    def save_to_disk(self):
        target_path = self._root_dir
        if target_path.exists():
            raise FileExistsError(f"Destination {self._root_dir} is not empty!")

        self._hf_dataset.save_to_disk(target_path / "hf_dataset")

        # Copy the encodings directory, if needed
        if (target_path / "encodings").exists():
            shutil.copy(self._root_dir / "encodings", target_path / "encodings")

        with open(target_path / self._METADATA_FILE_NAME, "w") as f:
            json.dump(self.metadata, f, indent=4, sort_keys=True, default=lambda x: x.__dict__)

    @classmethod
    def load_from_disk(
        cls,
        path: Path,
    ) -> "LatentisDataset":
        assert (
            path / cls._METADATA_FILE_NAME
        ).exists(), f"Metadata file {path / cls._METADATA_FILE_NAME} does not exist! Are you sure about the parameters?"

        with open(path / cls._METADATA_FILE_NAME, "r") as f:
            metadata = json.load(f)

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

        return dataset
