import logging

from datasets import ClassLabel, Dataset

from latentis import PROJECT_ROOT
from latentis.benchmark.task import Task
from latentis.data import actions
from latentis.data.dataset import DatasetView, DataType, Feature, FeatureMapping, FeatureProperty, HFDatasetView
from latentis.pipeline.flow import Flow, Pipeline

pylogger = logging.getLogger(__name__)


_RANDOM_SEED: int = 42
_ID_COLUMN: str = "sample_id"


class BuildDatasetTask(Task):
    def __init__(self, pipeline: Pipeline) -> None:
        super().__init__()
        self.pipeline = pipeline

    def _run(self) -> DatasetView:
        return self.pipeline.run()["dataset_view"]


class FeatureProcessor:
    def process(self, dataset: Dataset) -> Dataset:
        raise NotImplementedError


class FeatureCast(FeatureProcessor):
    def __init__(self, feature_name: str, feature: ClassLabel):
        self.feature_name = feature_name
        self.feature = feature

    def process(self, dataset: Dataset) -> Dataset:
        return dataset.cast_column(self.feature_name, self.feature)
        # dataset.cast_column(
        #     "label",
        #     ClassLabel(
        #         num_classes=len(set(dataset["train"]["label"])),
        #         names=list(set(dataset["train"]["label"])),
        #     ),
        # )


# class X:
#     def __init__(
#         self,
#         load_dataset_params: Mapping[str, Any],
#         features: Sequence[Feature],
#         metadata={},
#         id_column: str = _ID_COLUMN,
#     ):
#         self.load_dataset_params = load_dataset_params
#         self.features = features
#         self.metadata = metadata
#         self.id_column = id_column

#     @abstractmethod
#     def _process(self, dataset: DatasetDict) -> DatasetDict:
#         raise NotImplementedError

#     def process(self, dataset_name: str = None, perc: float = 1, parent_dir: Path = DATA_DIR) -> DatasetView:
#         hf_dataset: DatasetDict = load_dataset(**self.load_dataset_params)
#         dataset_name: str = dataset_name or list(hf_dataset.values())[0].info.dataset_name

#         # Select a random subset, if needed

#         # start_columns = {col for cols in hf_dataset.column_names.values() for col in cols}
#         # core_columns = {feature.target_name for feature in self.features}

#         # hf_dataset: DatasetDict = self._process(dataset=hf_dataset)

#         # hf_dataset = hf_dataset.remove_columns([col for col in start_columns if col not in core_columns])

#         # hf_dataset = hf_dataset.map(
#         # lambda _, index: {self.id_column: index},
#         # with_indices=True,
#         # )

#         processed_dataset = DatasetView(
#             name=dataset_name,
#             perc=perc,
#             hf_dataset=hf_dataset,
#             id_column=self.id_column,
#             features=self.features,
#             parent_dir=parent_dir,
#         )

#         return processed_dataset


DBPedia14 = Pipeline(
    name="process_dbpedia14",
    flows=(
        Flow(outputs=["dataset_view"])
        .add(block="load_dataset", outputs="data")
        .add(block="subset", inputs="data", outputs="data")
        .add(block="map_text", inputs="data", outputs="data")
        .add(block="map_feature_names", inputs="data", outputs="data")
        .add(block="prune_columns", inputs="data", outputs="data")
        .add(block="to_view", inputs="data", outputs="dataset_view")
    ),
    blocks={
        "load_dataset": actions.LoadHFDataset(path="fancyzhx/dbpedia_14"),
        "subset": actions.Subset(perc=0.01, seed=42),
        "map_text": actions.HFMap(
            input_columns=["title", "content"],
            fn=lambda title, content: {
                "x": [title + ". " + content.strip('"').strip() for title, content in zip(title, content)]
            },
        ),
        "map_feature_names": actions.MapFeatures(
            FeatureMapping(source_col="label", target_col="y"),
        ),
        "prune_columns": actions.PruneColumns(keep=["x", "y"]),
        "to_view": actions.ToView(
            name="dbpedia14",
            id_column="sample_id",
            features=[
                Feature(name="x", data_type=DataType.TEXT, properties={FeatureProperty.LANGUAGE: "en"}),
                Feature(name="y", data_type=DataType.LABEL),
            ],
        ),
    },
)

TREC = Pipeline(
    name="process_trec",
    flows=(
        Flow(outputs=["dataset_view"])
        .add(block="load_dataset", outputs="data")
        .add(block="subset", inputs="data", outputs="data")
        .add(block="to_view", inputs="data", outputs="dataset_view")
    ),
    blocks={
        "load_dataset": actions.LoadHFDataset(path="trec"),
        "subset": actions.Subset(perc=1, seed=42),
        "to_view": actions.ToView(
            name="trec",
            id_column="sample_id",
            features=[
                Feature(name="text", data_type=DataType.TEXT, properties={FeatureProperty.LANGUAGE: "en"}),
                Feature(
                    name="coarse_label", data_type=DataType.LABEL, properties={FeatureProperty.FINE_GRAINED: False}
                ),
                Feature(name="fine_label", data_type=DataType.LABEL, properties={FeatureProperty.FINE_GRAINED: True}),
            ],
        ),
    },
)

AGNews = Pipeline(
    name="process_ag_news",
    flows=(
        Flow(outputs=["dataset_view"])
        .add(block="load_dataset", outputs="data")
        .add(block="cast_label", inputs="data", outputs="data")
        .add(block="to_view", inputs="data", outputs="dataset_view")
    ),
    blocks={
        "load_dataset": actions.LoadHFDataset(path="ag_news"),
        "cast_label": actions.ClassLabelCast(column_name="label"),
        "to_view": actions.ToView(
            name="ag_news",
            id_column="sample_id",
            features=[
                Feature(name="text", data_type=DataType.TEXT, properties={FeatureProperty.LANGUAGE: "en"}),
                Feature(name="label", data_type=DataType.LABEL),
            ],
        ),
    },
)


IMDB = Pipeline(
    name="process_imdb",
    flows=(
        Flow(outputs=["dataset_view"])
        .add(block="load_dataset", outputs="data")
        .add(block="imdb_process", inputs="data", outputs="data")
        .add(block="subset", inputs="data", outputs="data")
        .add(block="cast_label", inputs="data", outputs="data")
        .add(block="to_view", inputs="data", outputs="dataset_view")
    ),
    blocks={
        "load_dataset": actions.LoadHFDataset(path="imdb"),
        "imdb_process": actions.imdb_process,
        "subset": actions.Subset(perc=1, seed=42),
        "cast_label": actions.ClassLabelCast(column_name="label"),
        "to_view": actions.ToView(
            name="imdb",
            id_column="sample_id",
            features=[
                Feature(name="text", data_type=DataType.TEXT, properties={FeatureProperty.LANGUAGE: "en"}),
                Feature(name="label", data_type=DataType.LABEL),
            ],
        ),
    },
)

# dataset = dataset.cast_column(
#             "label",
#             ClassLabel(
#                 num_classes=len(set(dataset["train"]["label"])),
#                 names=list(set(dataset["train"]["label"])),
#             ),
#         )
MNIST = Pipeline(
    name="process_mnist",
    flows=(
        Flow(outputs=["dataset_view"])
        .add(block="load_dataset", outputs="data")
        .add(block="map_feature_names", inputs="data", outputs="data")
        .add(block="cast_label", inputs="data", outputs="data")
        .add(block="to_view", inputs="data", outputs="dataset_view")
    ),
    blocks={
        "load_dataset": actions.LoadHFDataset(path="mnist"),
        "map_feature_names": actions.MapFeatures(
            FeatureMapping(source_col="label", target_col="y"),
            FeatureMapping(source_col="image", target_col="x"),
        ),
        "cast_label": actions.ClassLabelCast(column_name="label"),
        "to_view": actions.ToView(
            name="mnist",
            id_column="sample_id",
            features=[
                Feature(name="x", data_type=DataType.IMAGE),
                Feature(name="y", data_type=DataType.LABEL),
            ],
        ),
    },
)

# dataset = dataset.rename_column("img", "image")

#         dataset = dataset.cast_column(
#             "label",
#             ClassLabel(
#                 num_classes=len(set(dataset["train"]["label"])),
#                 names=list(set(dataset["train"]["label"])),
#             ),
#         )
CIFAR10 = Pipeline(
    name="process_cifar10",
    flows=(
        Flow(outputs=["dataset_view"])
        .add(block="load_dataset", outputs="data")
        .add(block="map_feature_names", inputs="data", outputs="data")
        .add(block="cast_label", inputs="data", outputs="data")
        .add(block="to_view", inputs="data", outputs="dataset_view")
    ),
    blocks={
        "load_dataset": actions.LoadHFDataset(path="cifar10"),
        "map_feature_names": actions.MapFeatures(
            FeatureMapping(source_col="label", target_col="y"),
            FeatureMapping(source_col="image", target_col="x"),
        ),
        "cast_label": actions.ClassLabelCast(column_name="label"),
        "to_view": actions.ToView(
            name="cifar10",
            id_column="sample_id",
            features=[
                Feature(name="x", data_type=DataType.IMAGE),
                Feature(name="y", data_type=DataType.LABEL),
            ],
        ),
    },
)

# dataset = dataset.rename_column("img", "image")

#         for label in ("coarse_label", "fine_label"):
#             dataset = dataset.cast_column(
#                 label,
#                 ClassLabel(
#                     num_classes=len(set(dataset["train"][label])),
#                     names=list(set(dataset["train"][label])),
#                 ),
#             )
CIFAR100 = Pipeline(
    name="process_cifar100",
    flows=(
        Flow(outputs=["dataset_view"])
        .add(block="load_dataset", outputs="data")
        .add(block="map_feature_names", inputs="data", outputs="data")
        .add(block="cast_label", inputs="data", outputs="data")
        .add(block="to_view", inputs="data", outputs="dataset_view")
    ),
    blocks={
        "load_dataset": actions.LoadHFDataset(path="cifar100"),
        "cast_label": actions.ClassLabelCast(column_name="fine_label"),
        "to_view": actions.ToView(
            name="cifar100",
            id_column="sample_id",
            features=[
                Feature(name="x", data_type=DataType.IMAGE),
                Feature(name="coarse_label", data_type=DataType.LABEL),
                Feature(name="fine_label", data_type=DataType.LABEL),
            ],
        ),
    },
)


FashionMNIST = Pipeline(
    name="process_fashion_mnist",
    flows=(
        Flow(inputs="perc", outputs=["data"])
        .add(block="load_dataset", outputs="data")
        .add(block="subset", inputs=["data", "perc"], outputs="data")
        .add(block="map_feature_names", inputs="data", outputs="data")
        .add(block="cast_label", inputs="data", outputs="data")
    ),
    blocks={
        "load_dataset": actions.LoadHFDataset(path="fashion_mnist"),
        "subset": actions.subset,
        "map_feature_names": actions.MapFeatures(
            FeatureMapping(source_col="label", target_col="y"),
            FeatureMapping(source_col="image", target_col="x"),
        ),
        "cast_label": actions.ClassLabelCast(column_name="label"),
    },
)


if __name__ == "__main__":
    data: DatasetView = TREC.build().run()["dataset_view"]
    data.save_to_disk(
        parent_dir=PROJECT_ROOT / "data",
    )

    print(data.hf_dataset)

    data = HFDatasetView.load_from_disk(path=PROJECT_ROOT / "data" / "trec")

    print(data.hf_dataset)
