import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from enum import auto
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from datasets import DatasetDict
from torch import nn

from latentis.data import DATA_DIR
from latentis.disk_index import DiskIndex
from latentis.io_utils import load_json, save_json
from latentis.space import LatentSpace
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
        self._encoding_index = DiskIndex(self._root_dir / "encodings", item_class=LatentSpace)

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

    def get_available_encodings(self, **properties) -> Mapping[str, Mapping[str, Any]]:
        """Scans the encodings directory and returns a mapping from feature to available encodings.

        Returns:
            Mapping[str, Mapping[str, Any]]: Mapping from feature to split to available encodings.

        """
        # TODO:use the index
        raise NotImplementedError

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
        raise NotImplementedError

    def add_encoding(
        self,
        space: LatentSpace,
        save_source_model: bool,
    ) -> LatentSpace:
        try:
            existing_space = self._encoding_index.load_item(**space.properties())
            existing_space.add_vectors(vectors=space.vectors, keys=space.keys)

            target_path = self._encoding_index.get_item_path(**space.properties())

            existing_space.save_to_disk(
                target_path,
                save_vector_source=True,
                save_info=False,
                save_source_model=save_source_model,
                save_decoders=False,
            )
        except KeyError:
            return self._encoding_index.add_item(
                item=space,
                save_args={
                    "save_vector_source": True,
                    "save_info": True,
                    "save_source_model": save_source_model,
                    "save_decoders": True,
                },
            )

    def save_to_disk(self):
        target_path = self._root_dir
        if target_path.exists():
            raise FileExistsError(f"Destination {self._root_dir} is not empty!")

        self._hf_dataset.save_to_disk(target_path / "hf_dataset")

        # Copy the encodings directory, if needed
        if (target_path / "encodings").exists():
            shutil.copy(self._root_dir / "encodings", target_path / "encodings")

        save_json(self.metadata, target_path / self._METADATA_FILE_NAME, indent=4)

        self._encoding_index.save_to_disk()

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
        dataset._encoding_index = DiskIndex.load_from_disk(path=path / "encodings")

        return dataset
