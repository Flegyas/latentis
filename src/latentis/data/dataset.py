import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from enum import auto
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from datasets import DatasetDict

from latentis.data import BENCHMARK_DIR
from latentis.types import StrEnum


class FeatureDataType(StrEnum):
    TEXT = auto()
    IMAGE = auto()


class FeatureProperty(StrEnum):
    LANGUAGE = auto()


class TaskProperty(StrEnum):
    pass


class DatasetProperty(StrEnum):
    FINE_GRAINED = auto()


# make it json serializable
@dataclass(frozen=True)
class Feature:
    col_name: str
    data_type: FeatureDataType
    properties: Mapping[FeatureProperty, str] = field(default_factory=lambda: {})


class TaskType(StrEnum):
    CLASSIFICATION = auto()
    AUTOENCODING = auto()


@dataclass(frozen=True)
class Task:
    col_name: str
    task_type: TaskType
    properties: Mapping[TaskProperty, str] = field(default_factory=lambda: {})


@dataclass(frozen=True)
class FeatureMapping:
    source_col: str
    target_col: str


class LatentisDataset:
    _METADATA_FILE_NAME: str = "metadata.json"

    @classmethod
    def get_key(cls, dataset_name: str, perc: float, properties: Mapping[DatasetProperty, Any]) -> str:
        hashcode = sha256()

        hashcode.update(dataset_name.encode())
        hashcode.update(str(perc).encode())
        hashcode.update(json.dumps(properties, sort_keys=True).encode())

        return hashcode.hexdigest()[:8]

    def __init__(
        self,
        name: str,
        dataset: DatasetDict,
        id_column: str,
        features: Sequence[Feature],
        tasks: Sequence[Task],
        perc: float = 1,
        properties: Mapping[str, Any] = {},
        parent_dir: Path = BENCHMARK_DIR,
    ):
        self.name: str = name
        self.dataset = dataset
        self.id_column: str = id_column
        self.features = features
        self.tasks = tasks
        self.perc = perc
        self.properties = properties
        self.parent_dir = parent_dir

    def __repr__(self):
        return f"{self.__class__.__name__}(features={self.features}, tasks={self.tasks}, perc={self.perc}, properties={self.properties})"

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
    #         tasks=self.tasks,
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
                    / LatentisDataset.get_key(dataset_name=self.name, perc=self.perc, properties=self.properties)
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
    #         tasks=self.tasks,
    #         perc=self.perc,
    #         properties=self.properties,
    #         parent_dir=self.parent_dir,
    #     )

    def get_available_encodings(self) -> Mapping[str, Sequence[str]]:
        """Scans the encodings directory of this dataset and returns the available ones.

        Returns:
            Mapping[str, Sequence[str]]: Mapping from encoding key to splits

        """
        encoded_dir = (
            self.parent_dir
            / self.name
            / LatentisDataset.get_key(dataset_name=self.name, perc=self.perc, properties=self.properties)
            / "encodings"
        )

        if not encoded_dir.exists():
            return []

        return {
            encoding_path.name: list(encoding_path.iterdir())
            for encoding_path in encoded_dir.iterdir()
            if encoding_path.is_dir()
        }

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

    def add_encoding(self, encoding_key: str, split: str, id2encoding: Mapping[str, torch.Tensor]):
        ids, encodings = zip(*id2encoding.items())
        data_list = [
            {self.id_column: sample_id, "encoding": encoding.numpy()} for sample_id, encoding in zip(ids, encodings)
        ]

        table = pa.Table.from_pylist(data_list)

        target_path = (
            self.parent_dir
            / self.name
            / LatentisDataset.get_key(dataset_name=self.name, perc=self.perc, properties=self.properties)
            / "encodings"
            / encoding_key
            / split
            / "encodings.parquet"
        )

        target_path.parent.mkdir(parents=True, exist_ok=True)

        if target_path.exists():
            current_table = pa.parquet.read_table(target_path)
            table = pa.concat_tables([current_table, table])

        pa.parquet.write_table(table, target_path)

    def save_to_disk(self, overwrite: bool = False):
        dataset_path = self.parent_dir / self.name
        instance_path = dataset_path / LatentisDataset.get_key(dataset_name=self.name, perc=self.perc, properties={})
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
            "features": self.features,
            "tasks": self.tasks,
            "perc": self.perc,
            "properties": self.properties,
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
        properties: Mapping[DatasetProperty, Any] = {},
        parent_dir: Path = BENCHMARK_DIR,
    ) -> "LatentisDataset":
        path: Path = (
            parent_dir
            / dataset_name
            / LatentisDataset.get_key(dataset_name=dataset_name, perc=perc, properties=properties)
            / "processed"
        )

        assert (
            path / cls._METADATA_FILE_NAME
        ).exists(), f"Metadata file {path / cls._METADATA_FILE_NAME} does not exist! Are you sure about the parameters?"

        dataset = DatasetDict.load_from_disk(path)
        with open(path / cls._METADATA_FILE_NAME, "r") as f:
            data = json.load(f)

        features = [Feature(**feature) for feature in data["features"]]
        tasks = [Task(**task) for task in data["tasks"]]
        perc = data["perc"]
        properties = data["properties"]
        id_column = data["id_column"]

        processed_dataset = LatentisDataset(
            name=dataset_name,
            dataset=dataset,
            id_column=id_column,
            features=features,
            tasks=tasks,
            perc=perc,
            properties=properties,
        )

        return processed_dataset