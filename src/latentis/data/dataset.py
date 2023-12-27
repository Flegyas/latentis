from dataclasses import dataclass, field
from enum import auto
from typing import Mapping, Sequence, Tuple

from datasets import ClassLabel, Dataset, load_dataset
from datasets.arrow_dataset import FeatureType

from latentis.types import StrEnum


class FeatureDataType(StrEnum):
    TEXT = auto()
    IMAGE = auto()


class FeatureProperty(StrEnum):
    LANGUAGE = auto()


class TaskProperty(StrEnum):
    pass


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


@dataclass(
    frozen=True,
)
class FeatureMapping:
    source_col: str
    target_col: str


def map_features(dataset: Dataset, *feature_mappings: FeatureMapping):
    dataset = dataset.map(
        lambda x: {feature_mapping.target_col: x[feature_mapping.source_col] for feature_mapping in feature_mappings}
    )

    for feature_mapping in feature_mappings:
        dataset = dataset.cast_column(
            feature_mapping.target_col,
            FeatureType(dtype=dataset.features[feature_mapping.source_col].dtype),
        )

    return dataset


@dataclass
class DatasetFactory:
    hf_key: Tuple[str, ...]
    features: Sequence[Feature]
    tasks: Sequence[Task]
    split: str
    perc: float = 1
    seed: int = 42

    def preprocess(self, dataset: Dataset) -> Dataset:
        return dataset

    def build(self, remove_columns: bool = True):
        dataset = load_dataset(
            *self.hf_key,
            split=self.split,
            token=True,
        )

        # Select a random subset, if needed
        if self.perc != 1:
            dataset = dataset.shuffle(seed=self.seed).select(list(range(int(len(dataset) * self.perc))))

        start_columns = dataset.column_names
        core_columns = set([feature.col_name for feature in self.features] + [task.col_name for task in self.tasks])

        # Preprocess the dataset
        dataset = self.preprocess(dataset)

        if remove_columns:
            dataset = dataset.remove_columns([col for col in start_columns if col not in core_columns])

        dataset = dataset

        return dataset


class DBPedia14(DatasetFactory):
    def __init__(self, split: str, perc: float, seed: int):
        super().__init__(
            hf_key=("dbpedia_14",),
            features=[
                Feature(col_name="content", data_type=FeatureDataType.TEXT, properties={FeatureProperty.LANGUAGE: "en"})
            ],
            tasks=[
                Task(col_name="label", task_type=TaskType.CLASSIFICATION),
            ],
            split=split,
            perc=perc,
            seed=seed,
        )

    def preprocess(self, dataset: Dataset):
        def transform(batch):
            return [sample["content"].strip('"').strip() for sample in batch]

        dataset.set_transform(transform, columns=["content"], output_all_columns=True)

        return dataset


class TREC(DatasetFactory):
    def __init__(self, split: str, perc: float, seed: int, fine_grained: bool = False):
        super().__init__(
            hf_key=("trec",),
            features=[
                Feature(col_name="text", data_type=FeatureDataType.TEXT, properties={FeatureProperty.LANGUAGE: "en"})
            ],
            tasks=[
                Task(col_name="coarse_label" if not fine_grained else "fine_label", task_type=TaskType.CLASSIFICATION),
            ],
            split=split,
            perc=perc,
            seed=seed,
        )

        self.fine_grained = fine_grained

    def __repr__(self):
        return super().__repr__()[:-1] + f" , fine_grained={self.fine_grained})"


# class AmazonReviewsMulti(DatasetInfo):
#     def __init__(self, split: str, perc: float, seed: int, fine_grained: bool = False):
#         super().__init__(
#             hf_key=("amazon_reviews_multi",),
#             split=split,
#             perc=perc,
#             seed=seed,
#         )

#         self.fine_grained = fine_grained

#     def preprocess(self, dataset: Dataset):
#         target_key: str = "stars"
#         data_key: str = "content"

#         def clean_sample(sample):
#             title: str = sample["review_title"].strip('"').strip(".").strip()
#             body: str = sample["review_body"].strip('"').strip(".").strip()

#             if body.lower().startswith(title.lower()):
#                 title = ""

#             if len(title) > 0 and title[-1].isalpha():
#                 title = f"{title}."

#             sample[data_key] = f"{title} {body}".lstrip(".").strip()
#             if self.fine_grained:
#                 sample[target_key] = str(sample[target_key] - 1)
#             else:
#                 sample[target_key] = sample[target_key] > 3
#             return sample

#         dataset = dataset.map(clean_sample)
#         dataset = dataset.cast_column(
#             target_key,
#             ClassLabel(
#                 num_classes=5 if self.fine_grained else 2,
#                 names=list(map(str, range(1, 6) if self.fine_grained else (0, 1))),
#             ),
#         )

#         dataset = update_input_output(dataset, input_col="content", output_col="stars")

#         return dataset

#     def __repr__(self):
#         return super().__repr__()[:-1] + f" , fine_grained={self.fine_grained})"


# class NewsGroups(DatasetInfo):
#     def __init__(self, split: str, perc: float, seed: int):
#         super().__init__(
#             hf_key=("rungalileo/20_Newsgroups_Fixed",),
#             split=split,
#             perc=perc,
#             seed=seed,
#         )

#     def preprocess(self, dataset: Dataset):
#         dataset = dataset.filter(lambda x: x["label"] != "None")
#         dataset = dataset.map(lambda x: {"coarse_label": x["label"].split(".")[0]})

#         target_key: str = "coarse_label"
#         data_key: str = "text"

#         dataset = dataset.cast_column(
#             "coarse_label",
#             ClassLabel(
#                 num_classes=len(set(dataset["coarse_label"])),
#                 names=list(set(dataset["coarse_label"])),
#             ),
#         )

#         dataset = update_input_output(dataset, input_col="text", output_col="coarse_label")

#         return dataset


class AGNews(DatasetFactory):
    def __init__(self, split: str, perc: float, seed: int):
        super().__init__(
            hf_key=("ag_news",),
            features=[
                Feature(col_name="text", data_type=FeatureDataType.TEXT, properties={FeatureProperty.LANGUAGE: "en"})
            ],
            tasks=[
                Task(col_name="label", task_type=TaskType.CLASSIFICATION),
            ],
            split=split,
            perc=perc,
            seed=seed,
        )

    def preprocess(self, dataset: Dataset):
        dataset = dataset.cast_column(
            "label",
            ClassLabel(
                num_classes=len(set(dataset["label"])),
                names=list(set(dataset["label"])),
            ),
        )

        return dataset


class IMDB(DatasetFactory):
    def __init__(self, split: str, perc: float, seed: int):
        super().__init__(
            hf_key=("imdb",),
            features=[
                Feature(col_name="text", data_type=FeatureDataType.TEXT, properties={FeatureProperty.LANGUAGE: "en"})
            ],
            tasks=[
                Task(col_name="label", task_type=TaskType.CLASSIFICATION),
            ],
            split=split,
            perc=perc,
            seed=seed,
        )

    def preprocess(self, dataset: Dataset):
        dataset = dataset.cast_column(
            "label",
            ClassLabel(
                num_classes=len(set(dataset["label"])),
                names=list(set(dataset["label"])),
            ),
        )

        return dataset
