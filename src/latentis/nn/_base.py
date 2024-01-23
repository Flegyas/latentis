from __future__ import annotations

from abc import abstractmethod
from typing import Mapping, Optional, Sequence

from lightning import LightningModule
from torch import nn
from torch.utils.data import DataLoader, default_collate


class LatentisModule(LightningModule):
    def __init__(self, model_key: str):
        super().__init__()
        self.model_key = model_key

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
    def decode(self, *args, **kwargs):
        raise NotImplementedError

    @property
    @abstractmethod
    def key(self) -> str:
        raise NotImplementedError


class WrappedModule(LatentisModule):
    def __init__(
        self,
        model_key: str,
        model: nn.Module,
        encode_fn: Optional[str] = None,
        decode_fn: Optional[str] = None,
    ):
        super().__init__(model_key=model_key)
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

    @property
    def key(self) -> str:
        return self.model_key


class PooledModel(WrappedModule):
    def __init__(
        self, model_key: str, model: nn.Module, pooler: nn.Module, encode_fn: str = "encode", decode_fn: str = "decode"
    ):
        super().__init__(model_key=model_key, model=model, encode_fn=encode_fn, decode_fn=decode_fn)
        self.pooler = pooler

    def encode(self, *args, **kwargs):
        return self.pooler(super().encode(*args, **kwargs))

    @property
    def key(self) -> str:
        return f"{super().key}_{self.pooler.name if hasattr(self.pooler, 'name') else self.pooler.__class__.__name__}"


class StitchedModel(nn.Module):
    def __init__(self, encoding_model: LatentisModule, decoding_model: LatentisModule):
        super().__init__()
        self.encoding_model = encoding_model
        self.decoding_model = decoding_model

    def encode(self, *args, **kwargs):
        return self.encoding_model.encode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.decoding_model.decode(*args, **kwargs)
