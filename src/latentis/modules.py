from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F
from PIL.Image import Image
from torch import nn
from torch.utils.data import default_collate
from transformers import AutoFeatureExtractor, AutoModel, AutoTokenizer, BatchEncoding, PreTrainedModel


class SVCModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return F.one_hot(torch.as_tensor(self.model.predict(x.cpu().numpy()))).to(x.device)


class Decoder(nn.Module):
    pass


class LatentisModule(nn.Module):
    def __init__(
        self, model_key: str, model: nn.Module, encode_fn: Optional[str] = None, decode_fn: Optional[str] = None
    ):
        super().__init__()
        self.model_key = model_key
        self.model = model
        self.encode_fn = encode_fn
        self.decode_fn = decode_fn

    def pre_encode(self, samples: Sequence, feature: str):
        return default_collate({feature: [sample[feature] for sample in samples]})

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

    # def __getattr__(self, item):
    #     if item in self.__dict__:
    #         return getattr(self, item)

    #     return getattr(self.model, item)

    # def __getattribute__(self, item):
    #     if item in self.__dict__:
    #         return getattr(self, item)

    #     return getattr(self.model, item)


class PooledModel(LatentisModule):
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


class HFEncoder(LatentisModule):
    def __init__(
        self, model_name: str, requires_grad: bool, encode_fn: Optional[str] = None, decode_fn: Optional[str] = None
    ):
        hf_model: PreTrainedModel = (
            AutoModel.from_pretrained(model_name, output_hidden_states=True, return_dict=True)
            .eval()
            .requires_grad_(requires_grad)
        )
        super().__init__(model_key=model_name, model=hf_model, encode_fn=encode_fn, decode_fn=decode_fn)


class TextHFEncoder(HFEncoder):
    def __init__(
        self,
        model_name: str,
        requires_grad: bool = False,
        truncation: bool = True,
        padding: bool = True,
        max_length: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(model_name, requires_grad, encode_fn=None, decode_fn=None)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        max_length = max_length or self.model.config.max_length

        self.pre_encode_kwargs = {
            "truncation": truncation,
            "padding": padding,
            "max_length": max_length,
            **kwargs,
        }

    @torch.no_grad()
    def pre_encode(self, samples: Sequence, feature: str) -> BatchEncoding:
        is_clip: bool = self.model_key.startswith("openai/clip")

        tok_out: BatchEncoding = self.tokenizer(
            [sample[feature] for sample in samples],
            return_special_tokens_mask=True,
            return_token_type_ids=not is_clip,
            return_tensors="pt",
            **self.pre_encode_kwargs,
        )

        return BatchEncoding(dict(tok_out=tok_out))

    def encode(self, x: BatchEncoding):
        tok_out = x["tok_out"]

        mask = tok_out["attention_mask"] * tok_out["special_tokens_mask"].bool().logical_not()
        del tok_out["special_tokens_mask"]

        if self.model_key.startswith("openai/clip"):
            # TODO: fix this
            encodings = [self.model.text_model(**tok_out, return_dict=True)["last_hidden_state"]]
        else:
            encodings = self.model(**tok_out)["hidden_states"]

        return {"x": encodings, "mask": mask}


class ImageHFEncoder(HFEncoder):
    def __init__(self, model_name: str, requires_grad: bool = False):
        super().__init__(model_name, requires_grad)
        self.extractor = AutoFeatureExtractor.from_pretrained(self.model_name)

    @torch.no_grad()
    def pre_encode(self, samples: Sequence, feature: str, **kwargs):
        is_clip: bool = self.model_key.startswith("openai/clip")

        images = [sample[feature] for sample in samples]

        kwargs["return_tensors"] = "pt"

        if is_clip:
            return self._clip_image_encode(images=images, **kwargs)
        else:
            return self._image_encode(images=images, **kwargs)

    @torch.no_grad()
    def _image_encode(self, images: Sequence[Image], **kwargs):
        images: List[torch.Tensor] = [self.extractor(image["image"].convert("RGB"), **kwargs) for image in images]
        images: torch.Tensor = torch.stack(images, dim=0)

        return {"image": images}

    @torch.no_grad()
    def _clip_image_encode(self, images: Sequence[Image], **kwargs):
        images = [image["image"].convert("RGB") for image in images]
        images = self.extractor(images=images, **kwargs)

        return {"image": images}
