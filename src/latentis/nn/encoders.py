from typing import List, Optional, Sequence

import torch
from PIL.Image import Image
from transformers import AutoFeatureExtractor, AutoModel, AutoTokenizer, BatchEncoding, PreTrainedModel

from latentis.nn._base import WrappedModule
from latentis.types import Properties


class HFEncoder(WrappedModule):
    def __init__(
        self,
        hf_name: str,
        requires_grad: bool,
        encode_fn: Optional[str] = None,
        decode_fn: Optional[str] = None,
        properties: Optional[Properties] = None,
    ):
        hf_model: PreTrainedModel = (
            AutoModel.from_pretrained(hf_name, output_hidden_states=True, return_dict=True)
            .eval()
            .requires_grad_(requires_grad)
        )
        self.hf_name = hf_name
        super().__init__(
            model=hf_model,
            encode_fn=encode_fn,
            decode_fn=decode_fn,
            properties={**(properties or {}), "hf_name": hf_name},
        )


class TextHFEncoder(HFEncoder):
    def __init__(
        self,
        hf_name: str,
        requires_grad: bool = False,
        truncation: bool = True,
        padding: bool = True,
        max_length: Optional[int] = None,
        properties: Optional[Properties] = None,
        **kwargs,
    ):
        super().__init__(hf_name, requires_grad, encode_fn=None, decode_fn=None, properties=properties)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_name)

        max_length = max_length or self.model.config.max_length

        self.pre_encode_kwargs = {
            "truncation": truncation,
            "padding": padding,
            "max_length": max_length,
            **kwargs,
        }

    @torch.no_grad()
    def pre_encode(self, samples: Sequence, feature: str) -> BatchEncoding:
        is_clip: bool = self.hf_name.startswith("openai/clip")

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

        if self.hf_name.startswith("openai/clip"):
            # TODO: fix this
            encodings = [self.model.text_model(**tok_out, return_dict=True)["last_hidden_state"]]
        else:
            encodings = self.model(**tok_out)["hidden_states"]

        return {"x": encodings, "mask": mask}


class ImageHFEncoder(HFEncoder):
    def __init__(self, hf_name: str, requires_grad: bool = False, properties: Optional[Properties] = None):
        super().__init__(hf_name, requires_grad, properties=properties)
        self.extractor = AutoFeatureExtractor.from_pretrained(self.hf_name)

    @torch.no_grad()
    def pre_encode(self, samples: Sequence, feature: str, **kwargs):
        is_clip: bool = self.hf_name.startswith("openai/clip")

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
