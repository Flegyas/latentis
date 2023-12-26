from abc import abstractmethod
from dataclasses import dataclass
from typing import Tuple

from datasets import ClassLabel, Dataset, load_dataset

INPUT_COL: str = "input"
OUTPUT_COL: str = "output"


def update_input_output(dataset, input_col: str, output_col: str):
    dataset = dataset.map(lambda x: {INPUT_COL: x[input_col], OUTPUT_COL: x[output_col]})
    dataset = dataset.cast_column(
        OUTPUT_COL,
        dataset.features[output_col],
    )

    return dataset


@dataclass
class DatasetInfo:
    hf_key: Tuple[str, ...]
    split: str
    perc: float = 1
    seed: int = 42
    input_col: str = INPUT_COL
    output_col: str = OUTPUT_COL

    @abstractmethod
    def preprocess(self, dataset: Dataset) -> Dataset:
        raise NotImplementedError

    def build(self, remove_columns: bool = True):
        dataset = load_dataset(
            *self.hf_key,
            split=self.split,
            use_auth_token=True,
        )

        # Select a random subset, if needed
        if self.perc != 1:
            dataset = dataset.shuffle(seed=self.seed).select(list(range(int(len(dataset) * self.perc))))

        start_columns = dataset.column_names

        # Preprocess the dataset
        dataset = self.preprocess(dataset)

        if remove_columns:
            dataset = dataset.remove_columns(start_columns)

        dataset = dataset

        return dataset


class DBPedia14(DatasetInfo):
    def __init__(self, split: str, perc: float, seed: int):
        super().__init__(
            hf_key=("dbpedia_14",),
            split=split,
            perc=perc,
            seed=seed,
        )

    def preprocess(self, dataset: Dataset):
        def clean_sample(example):
            example["content"] = example["content"].strip('"').strip()
            return example

        dataset = dataset.map(clean_sample)
        dataset = update_input_output(dataset, input_col="content", output_col="label")

        return dataset


class TREC(DatasetInfo):
    def __init__(self, split: str, perc: float, seed: int, fine_grained: bool = False):
        super().__init__(
            hf_key=("trec",),
            split=split,
            perc=perc,
            seed=seed,
        )

        self.fine_grained = fine_grained

    def preprocess(self, dataset: Dataset):
        print(dataset)
        dataset = update_input_output(
            dataset,
            input_col="text",
            output_col="coarse_label" if not self.fine_grained else "fine_label",
        )

        return dataset

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


class AgNews(DatasetInfo):
    def __init__(self, split: str, perc: float, seed: int):
        super().__init__(
            hf_key=("ag_news",),
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

        dataset = update_input_output(dataset, input_col="text", output_col="label")

        return dataset


class IMDB(DatasetInfo):
    def __init__(self, split: str, perc: float, seed: int):
        super().__init__(
            hf_key=("imdb",),
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

        dataset = update_input_output(dataset, input_col="text", output_col="label")

        return dataset
