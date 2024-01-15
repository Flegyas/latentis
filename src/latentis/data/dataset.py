import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from enum import auto
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence, Union

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from datasets import DatasetDict
from torch import nn

from latentis.data import BENCHMARK_DIR
from latentis.types import MetadataMixin, SerializableMixin, StrEnum


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
@dataclass(
    frozen=True,
)
class Feature:
    col_name: str
    data_type: DataType
    properties: Mapping[FeatureProperty, str] = field(default_factory=lambda: {})

    def __hash__(self):
        return hash((self.col_name, self.data_type, tuple(self.properties.items())))


@dataclass(frozen=True)
class FeatureMapping:
    source_col: str
    target_col: str


class LatentisDataset(SerializableMixin, MetadataMixin):
    @classmethod
    def get_key(cls, dataset_name: str, perc: float, metadata: Mapping[DatasetProperty, Any]) -> str:
        hashcode = sha256()

        hashcode.update(dataset_name.encode())
        hashcode.update(str(perc).encode())
        hashcode.update(json.dumps(metadata, sort_keys=True).encode())

        return hashcode.hexdigest()[:8]

    def __init__(
        self,
        name: str,
        dataset: DatasetDict,
        id_column: str,
        features: Sequence[Feature],
        perc: float = 1,
        metadata: Mapping[str, Any] = {},
        parent_dir: Path = BENCHMARK_DIR,
    ):
        super().__init__()
        assert isinstance(dataset, DatasetDict), f"Expected {DatasetDict}, got {type(dataset)}"
        assert len(set(features)) == len(features), f"Features {features} contain duplicates!"
        assert len(features) > 0, f"Features {features} must not be empty!"
        assert all(
            id_column in dataset[split].column_names for split in dataset.keys()
        ), f"ID column {id_column} not in all splits of dataset {dataset}"
        assert all(
            feature.col_name in dataset[split].column_names for feature in features for split in dataset.keys()
        ), f"Specified features not in all splits of dataset {dataset}"
        assert 0 < perc <= 1, f"Percentage {perc} not in (0, 1]"

        self.name: str = name
        self.dataset = dataset
        self.id_column: str = id_column
        self.features = set(features)
        self.perc = perc
        self.metadata = metadata
        self.parent_dir = parent_dir

    def __repr__(self):
        return f"{self.__class__.__name__}(features={self.features}, perc={self.perc}, metadata={self.metadata})"

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
                encodings_path = (
                    self.parent_dir
                    / self.name
                    / LatentisDataset.get_key(dataset_name=self.name, perc=self.perc, metadata=self.metadata)
                    / "encodings"
                    / encoding
                    / split
                    / "encodings.parquet"
                )

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

    def get_available_encodings(
        self, *features: Feature
    ) -> Union[Mapping[str, Sequence[str]], Mapping[Feature, Mapping[str, Sequence[str]]]]:
        """Scans the encodings directory and returns a mapping from feature to available encodings.

        Returns:
            Mapping[str, Sequence[str]]: Mapping from feature to available encodings.

        """
        raise NotImplementedError
        # assert len(features) == 0 or all(
        #     feature in self.features for feature in features
        # ), f"Features {features} not available in dataset {self.name}!"
        # features = features or self.features

        # encodings_dir = (
        #     self.parent_dir
        #     / self.name
        #     / LatentisDataset.get_key(dataset_name=self.name, perc=self.perc, metadata=self.metadata)
        #     / "encodings"
        # )
        # if not encodings_dir.exists():
        #     return {}

        # available_features = [
        #     feature
        #     for feature in self.features
        #     if (encodings_dir / feature.col_name).exists() and (encodings_dir / feature.col_name).is_dir()
        # ]

        # return {
        #     encoding_path.name: list(encoding_path.iterdir())
        #     for encoding_path in encoded_dir.iterdir()
        #     if encoding_path.is_dir()
        # }

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

    def add_encoding(self, feature: Feature, encoding_key: str, split: str, id2encoding: Mapping[str, torch.Tensor]):
        ids, encodings = zip(*id2encoding.items())
        data_list = [
            {self.id_column: sample_id, "encoding": encoding.numpy()} for sample_id, encoding in zip(ids, encodings)
        ]

        table = pa.Table.from_pylist(data_list)

        target_path = (
            self.parent_dir
            / self.name
            / LatentisDataset.get_key(dataset_name=self.name, perc=self.perc, metadata=self.metadata)
            / "encodings"
            / feature.col_name
            / split
            / encoding_key
            / "encodings.parquet"
        )

        target_path.parent.mkdir(parents=True, exist_ok=True)

        if target_path.exists():
            current_table = pa.parquet.read_table(target_path)
            table = pa.concat_tables([current_table, table])

        pa.parquet.write_table(table, target_path)

    def save_to_disk(self, overwrite: bool = False):
        dataset_path = self.parent_dir / self.name
        instance_path = dataset_path / LatentisDataset.get_key(dataset_name=self.name, perc=self.perc, metadata={})
        processed_path = instance_path / "processed"

        # path = parent_dir / (
        #     dir_name or ProcessedDataset.get_key(hf_name=hf_name, perc=self.perc, properties=self.properties)
        # )

        if not overwrite and (processed_path / self._METADATA_FILE_NAME).exists():
            raise FileExistsError

        if processed_path.exists():
            shutil.rmtree(processed_path)

        self.dataset.save_to_disk(processed_path)

        data = {
            "name": self.name,
            "id_column": self.id_column,
            "features": tuple(self.features),
            "perc": self.perc,
            "metadata": self.metadata,
            "timestamp": str(datetime.now()),
            "timestamp_ms": int(datetime.now().timestamp()),
        }

        with open(processed_path / self._METADATA_FILE_NAME, "w") as f:
            json.dump(data, f, indent=4, sort_keys=True, default=lambda x: x.__dict__)

    @classmethod
    def load_from_disk(
        cls,
        dataset_name: str,
        perc: float,
        metadata: Mapping[DatasetProperty, Any] = {},
        parent_dir: Path = BENCHMARK_DIR,
    ) -> "LatentisDataset":
        path: Path = (
            parent_dir
            / dataset_name
            / LatentisDataset.get_key(dataset_name=dataset_name, perc=perc, metadata=metadata)
            / "processed"
        )

        assert (
            path / cls._METADATA_FILE_NAME
        ).exists(), f"Metadata file {path / cls._METADATA_FILE_NAME} does not exist! Are you sure about the parameters?"

        dataset = DatasetDict.load_from_disk(path)
        with open(path / cls._METADATA_FILE_NAME, "r") as f:
            data = json.load(f)

        features = [Feature(**feature) for feature in data["features"]]
        perc = data["perc"]
        metadata = data["metadata"]
        id_column = data["id_column"]

        processed_dataset = LatentisDataset(
            name=dataset_name,
            dataset=dataset,
            id_column=id_column,
            features=features,
            perc=perc,
            metadata=metadata,
        )

        return processed_dataset
