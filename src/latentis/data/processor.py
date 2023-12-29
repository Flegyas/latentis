import json
import logging
import shutil
from abc import abstractmethod
from dataclasses import dataclass, field
from enum import auto
from hashlib import sha256
from pathlib import Path
from typing import Mapping, Optional, Sequence

from altair import Any
from datasets import ClassLabel, Dataset, DatasetDict, load_dataset

from latentis import PROJECT_ROOT
from latentis.types import StrEnum

pylogger = logging.getLogger(__name__)


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


def map_features(dataset: Dataset, *feature_mappings: FeatureMapping):
    dataset = dataset.map(
        lambda *source_col_vals: {
            target_col: source_col_val
            for source_col_val, target_col in zip(
                source_col_vals, [feature_mapping.target_col for feature_mapping in feature_mappings]
            )
        },
        batched=True,
        input_columns=[feature_mapping.source_col for feature_mapping in feature_mappings],
    )

    # for feature_mapping in feature_mappings:
    #     dataset = dataset.cast_column(
    #         feature_mapping.target_col,
    #         feature=dataset.features[feature_mapping.source_col].dtype,
    #     )

    return dataset


_NAME_SEP: str = "_"
_RANDOM_SEED: int = 42

PROCESSED_DATA_DIR: Path = PROJECT_ROOT / "benchmark" / "processed"


class ProcessedDataset:
    _METADATA_FILE_NAME: str = "metadata.json"

    @classmethod
    def get_key(cls, dataset_name: str, perc: float, properties: Mapping[DatasetProperty, Any]) -> str:
        hashcode = sha256()

        hashcode.update(dataset_name.encode())
        hashcode.update(str(perc).encode())
        hashcode.update(json.dumps(properties, sort_keys=True).encode())

        return hashcode.hexdigest()[:8]

    def __init__(
        self, dataset: Dataset, features: Sequence[Feature], tasks: Sequence[Task], perc: float = 1, properties={}
    ):
        self.dataset = dataset
        self.features = features
        self.tasks = tasks
        self.perc = perc
        self.properties = properties

    def __repr__(self):
        return f"{self.__class__.__name__}(features={self.features}, tasks={self.tasks}, perc={self.perc}, properties={self.properties})"

    def save_to_disk(
        self, parent_dir: Path = PROCESSED_DATA_DIR, overwrite: bool = False, dir_name: Optional[str] = None
    ):
        dataset_info = next(iter(self.dataset.values())).info
        dataset_name = dataset_info.builder_name

        dataset_path = parent_dir / dataset_name
        instance_path = dataset_path / ProcessedDataset.get_key(
            dataset_name=dataset_name, perc=self.perc, properties=self.properties
        )

        # path = parent_dir / (
        #     dir_name or ProcessedDataset.get_key(hf_name=hf_name, perc=self.perc, properties=self.properties)
        # )

        if not overwrite and (instance_path / self._METADATA_FILE_NAME).exists():
            raise FileExistsError

        if instance_path.exists():
            shutil.rmtree(instance_path)

        self.dataset.save_to_disk(instance_path)

        data = {
            "features": self.features,
            "tasks": self.tasks,
            "perc": self.perc,
            "properties": self.properties,
        }

        with open(instance_path / self._METADATA_FILE_NAME, "w") as f:
            json.dump(data, f, indent=4, sort_keys=True, default=lambda x: x.__dict__)

    @classmethod
    def load_from_disk(
        cls,
        dataset_name: str,
        perc: float,
        properties: Mapping[DatasetProperty, Any] = {},
        data_dir: Path = PROCESSED_DATA_DIR,
    ) -> "ProcessedDataset":
        path = data_dir / dataset_name / ProcessedDataset.get_key(dataset_name=dataset_name, perc=perc, properties={})

        assert (
            path / cls._METADATA_FILE_NAME
        ).exists(), f"Metadata file {path / cls._METADATA_FILE_NAME} does not exist! Are you sure about the"

        dataset = DatasetDict.load_from_disk(path)
        with open(path / cls._METADATA_FILE_NAME, "r") as f:
            data = json.load(f)

        features = [Feature(**feature) for feature in data["features"]]
        tasks = [Task(**task) for task in data["tasks"]]
        perc = data["perc"]
        properties = data["properties"]

        processed_dataset = ProcessedDataset(
            dataset=dataset,
            features=features,
            tasks=tasks,
            perc=perc,
            properties=properties,
        )

        return processed_dataset


class DataProcessor:
    def __init__(
        self, load_dataset_params: Mapping[str, Any], features: Sequence[Feature], tasks: Sequence[Task], properties={}
    ):
        self.load_dataset_params = load_dataset_params
        self.features = features
        self.tasks = tasks
        self.properties = properties

    @abstractmethod
    def _process(self, dataset: DatasetDict) -> DatasetDict:
        raise NotImplementedError

    def process(self, perc: float = 1) -> ProcessedDataset:
        dataset: DatasetDict = load_dataset(**self.load_dataset_params)

        # Select a random subset, if needed
        if perc != 1:
            dataset = DatasetDict(
                {
                    split: dataset[split]
                    .shuffle(seed=_RANDOM_SEED)
                    .select(list(range(int(len(dataset[split]) * perc))))
                    for split in dataset.keys()
                }
            )
        start_columns = {col for cols in dataset.column_names.values() for col in cols}
        core_columns = set([feature.col_name for feature in self.features] + [task.col_name for task in self.tasks])

        dataset: DatasetDict = self._process(dataset=dataset)

        dataset = dataset.remove_columns([col for col in start_columns if col not in core_columns])

        processed_dataset = ProcessedDataset(
            dataset=dataset,
            features=self.features,
            tasks=self.tasks,
            perc=perc,
            properties=self.properties,
        )

        return processed_dataset


class DBPedia14Processor(DataProcessor):
    def __init__(self):
        super().__init__(
            load_dataset_params=dict(path="dbpedia_14"),
            features=[
                Feature(col_name="x", data_type=FeatureDataType.TEXT, properties={FeatureProperty.LANGUAGE: "en"})
            ],
            tasks=[
                Task(col_name="y", task_type=TaskType.CLASSIFICATION),
            ],
        )

    def _process(self, dataset: DatasetDict) -> ProcessedDataset:
        dataset = dataset.map(
            lambda title, content: {
                "x": [title + ". " + content.strip('"').strip() for title, content in zip(title, content)]
            },
            input_columns=["title", "content"],
            batched=True,
        )
        dataset = map_features(dataset, FeatureMapping(source_col="label", target_col="y"))

        return dataset


class TREC(DataProcessor):
    def __init__(self, fine_grained: bool = False):
        super().__init__(
            load_dataset_params=dict(path="trec"),
            features=[
                Feature(col_name="text", data_type=FeatureDataType.TEXT, properties={FeatureProperty.LANGUAGE: "en"})
            ],
            tasks=[
                Task(col_name="coarse_label" if not fine_grained else "fine_label", task_type=TaskType.CLASSIFICATION),
            ],
            properties={"fine_grained": fine_grained},
        )

        self.fine_grained = fine_grained

    def _process(self, dataset: DatasetDict) -> DatasetDict:
        return dataset


class AGNews(DataProcessor):
    def __init__(self):
        super().__init__(
            load_dataset_params=dict(path="ag_news"),
            features=[
                Feature(col_name="text", data_type=FeatureDataType.TEXT, properties={FeatureProperty.LANGUAGE: "en"})
            ],
            tasks=[
                Task(col_name="label", task_type=TaskType.CLASSIFICATION),
            ],
        )

    def _process(self, dataset: DatasetDict) -> DatasetDict:
        dataset = dataset.cast_column(
            "label",
            ClassLabel(
                num_classes=len(set(dataset["train"]["label"])),
                names=list(set(dataset["train"]["label"])),
            ),
        )

        return dataset


class IMDB(DataProcessor):
    def __init__(self):
        super().__init__(
            load_dataset_params=dict(path="imdb"),
            features=[
                Feature(col_name="text", data_type=FeatureDataType.TEXT, properties={FeatureProperty.LANGUAGE: "en"})
            ],
            tasks=[
                Task(col_name="label", task_type=TaskType.CLASSIFICATION),
            ],
        )

    def _process(self, dataset: DatasetDict) -> DatasetDict:
        dataset = dataset.cast_column(
            "label",
            ClassLabel(
                num_classes=len(set(dataset["train"]["label"])),
                names=list(set(dataset["train"]["label"])),
            ),
        )

        return dataset
