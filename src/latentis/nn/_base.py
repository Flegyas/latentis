from __future__ import annotations

import copy
from abc import abstractmethod
from enum import auto
from pathlib import Path
from typing import Mapping, Optional, Sequence

from lightning import LightningModule
from torch import nn
from torch.utils.data import DataLoader, default_collate

from latentis.serialize.io_utils import SerializableMixin, load_json, load_model, save_json, save_model
from latentis.types import Properties, StrEnum


class _LatentisModuleMetadata(StrEnum):
    _VERSION = auto()
    _TYPE = auto()


_PROPERTIES_FILE_NAME = "properties.json"


class LatentisModule(LightningModule, SerializableMixin):
    def __init__(self, properties: Optional[Properties] = None):
        super().__init__()

        properties = properties or {}
        # metadata[SpaceMetadata._NAME] = self._name
        properties[_LatentisModuleMetadata._VERSION] = self.version

        # TODO: store also the module to use for deserialization,
        # removing this info from the index
        properties[_LatentisModuleMetadata._TYPE] = LatentisModule.__name__

        self._properties = properties.copy()

    def save_to_disk(self, target_path: Path) -> None:
        target_path.mkdir(parents=True, exist_ok=True)
        save_json(self.properties, target_path / _PROPERTIES_FILE_NAME)
        save_model(model=self, target_path=target_path / "model.pt", version=self.version)

    @classmethod
    def load_from_disk(cls, path: Path) -> LatentisModule:
        properties = cls.load_properties(path)
        # TODO: if the save is changed, the properties should be injected here
        return load_model(path / "model.pt", version=properties[_LatentisModuleMetadata._VERSION])

    @property
    def version(cls) -> int:
        return 0

    @classmethod
    def load_properties(cls, space_path: Path) -> Properties:
        metadata = load_json(space_path / _PROPERTIES_FILE_NAME)
        return metadata

    @property
    def properties(self) -> Properties:
        return copy.deepcopy(self._properties)

    @abstractmethod
    def fit(self, train_dataloader: DataLoader) -> LatentisModule:
        raise NotImplementedError

    @abstractmethod
    def score(self, test_dataloader: DataLoader) -> Mapping[str, float]:
        raise NotImplementedError

    def pre_encode(self, samples: Sequence, feature: str):
        return default_collate({feature: [sample[feature] for sample in samples]})

    @abstractmethod
    def encode(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def post_encode(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def pre_decode(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def decode(self, *args, **kwargs):
        raise NotImplementedError


class WrappedModule(LatentisModule):
    def __init__(
        self,
        model: nn.Module,
        encode_fn: Optional[str] = None,
        decode_fn: Optional[str] = None,
        properties: Optional[Properties] = None,
    ):
        super().__init__(properties=properties)
        self.model = model
        self.encode_fn = encode_fn
        self.decode_fn = decode_fn

    def encode(self, *args, **kwargs):
        if self.encode_fn is None:
            raise NotImplementedError
        return getattr(self.model, self.encode_fn)(*args, **kwargs)

    def decode(self, *args, **kwargs):
        if self.decode_fn is None:
            raise NotImplementedError
        return getattr(self.model, self.decode_fn)(*args, **kwargs)


class PooledModel(WrappedModule):
    def __init__(
        self,
        model: nn.Module,
        pooler: nn.Module,
        encode_fn: str = "encode",
        decode_fn: str = "decode",
        properties: Optional[Properties] = None,
    ):
        super().__init__(
            model=model,
            encode_fn=encode_fn,
            decode_fn=decode_fn,
            properties={
                **properties,
                **{"pooler": self.pooler.name if hasattr(self.pooler, "name") else self.pooler.__class__.__name__},
            },
        )
        self.pooler = pooler

    def encode(self, *args, **kwargs):
        return self.pooler(super().encode(*args, **kwargs))


# TODO: this must be a LatentisModule
class StitchedModel(nn.Module):
    def __init__(self, encoding_model: LatentisModule, decoding_model: LatentisModule):
        super().__init__()
        self.encoding_model = encoding_model
        self.decoding_model = decoding_model

    def encode(self, *args, **kwargs):
        return self.encoding_model.encode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.decoding_model.decode(*args, **kwargs)
