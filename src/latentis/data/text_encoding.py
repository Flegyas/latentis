import logging
from enum import auto
from typing import Sequence

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from latentis.types import StrEnum

pylogger = logging.getLogger(__name__)


class EncodeMode(StrEnum):
    MEAN = auto()
    TOKEN = auto()
    RAW = auto()
    SUM = auto()
    CLS = auto()


@torch.no_grad()
def batch_encode(
    encoding,
    encoder: PreTrainedModel,
    prefix: str,
    modes: Sequence[EncodeMode] = EncodeMode.MEAN,
    only_last: bool = False,
    return_tensors: str = "pt",
):
    modes = set(modes)

    mask = encoding["attention_mask"] * encoding["special_tokens_mask"].bool().logical_not()
    del encoding["special_tokens_mask"]

    encoding = encoding.to(encoder.device)

    if prefix.startswith("openai/clip"):
        encodings = [encoder.text_model(**encoding, return_dict=True)["last_hidden_state"]]
    else:
        encodings = encoder(**encoding)["hidden_states"]

    raw_encodings = {
        f"{prefix}_raw_encoding_{i_layer}": layer_encoding for i_layer, layer_encoding in enumerate(encodings)
    }

    result = {} if EncodeMode.RAW not in modes else raw_encodings
    if EncodeMode.TOKEN in modes:
        token_encodings = {
            f"{prefix}_token_encoding_{i_layer}": [
                sample_encoding[sample_mask].cpu().numpy() for sample_encoding, sample_mask in zip(layer_encoding, mask)
            ]
            for i_layer, layer_encoding in enumerate(encodings)
        }
        result.update(**token_encodings)

    if EncodeMode.MEAN in modes:
        pooled_encodings = {
            f"{prefix}_mean_encoding_{i_layer}": (
                torch.stack(
                    [
                        sample_encoding[sample_mask].mean(dim=0)
                        for sample_encoding, sample_mask in zip(layer_encoding, mask)
                    ],
                    dim=0,
                )
            )
            for i_layer, layer_encoding in enumerate(raw_encodings.values())
        }
        if only_last:
            pooled_encodings = {
                f"{prefix}_mean_encoding": list(pooled_encodings.values())[-1].clone()
            }  # the standard encoding is set to be the one from the last layer
        else:
            pooled_encodings[f"{prefix}_mean_encoding"] = list(pooled_encodings.values())[
                -1
            ].clone()  # the standard encoding is set to be the one from the last layer

        result.update(**pooled_encodings)

    if EncodeMode.SUM in modes:
        pooled_encodings = {
            f"{prefix}_sum_encoding_{i_layer}": (
                torch.stack(
                    [
                        sample_encoding[sample_mask].sum(dim=0)
                        for sample_encoding, sample_mask in zip(layer_encoding, mask)
                    ],
                    dim=0,
                )
            )
            for i_layer, layer_encoding in enumerate(raw_encodings.values())
        }

        if only_last:
            pooled_encodings = {
                f"{prefix}_sum_encoding": list(pooled_encodings.values())[-1].clone()
            }  # the standard encoding is set to be the one from the last layer
        else:
            pooled_encodings[f"{prefix}_sum_encoding"] = list(pooled_encodings.values())[
                -1
            ].clone()  # the standard encoding is set to be the one from the last layer

        result.update(**pooled_encodings)

    if EncodeMode.CLS in modes:
        pooled_encodings = {
            f"{prefix}_cls_encoding_{i_layer}": layer_encoding[
                :, 0, :
            ]  # TODO: adapt to encoders without CLS as first token
            for i_layer, layer_encoding in enumerate(raw_encodings.values())
        }

        if only_last:
            pooled_encodings = {
                f"{prefix}_cls_encoding": list(pooled_encodings.values())[-1].clone()
            }  # the standard encoding is set to be the one from the last layer
        else:
            pooled_encodings[f"{prefix}_cls_encoding"] = list(pooled_encodings.values())[
                -1
            ].clone()  # the standard encoding is set to be the one from the last layer

        result.update(**pooled_encodings)

    if return_tensors == "numpy":
        result = {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in result.items()}
    elif return_tensors == "pt":
        pass
    else:
        raise NotImplementedError

    return result


@torch.no_grad()
def sample_encode(
    batch,
    tokenizer: PreTrainedTokenizer,
    encoder: PreTrainedModel,
    prefix: str,
    modes: Sequence[EncodeMode] = EncodeMode.MEAN,
    only_last: bool = False,
    return_tensors: str = "pt",
    data_key: str = "data",
):
    modes = set(modes)

    encoding = tokenizer(
        batch[data_key],
        return_special_tokens_mask=True,
        return_token_type_ids=True,
        return_tensors="pt",
        truncation=True,
        max_length=encoder.config.max_length,
        padding=True,
    ).to(encoder.device)

    mask = encoding["attention_mask"] * encoding["special_tokens_mask"].bool().logical_not()
    del encoding["special_tokens_mask"]

    encoding = encoding.to(encoder.device)

    encodings = encoder(**encoding)["hidden_states"]

    raw_encodings = {
        f"{prefix}_raw_encoding_{i_layer}": layer_encoding for i_layer, layer_encoding in enumerate(encodings)
    }

    result = {} if EncodeMode.RAW not in modes else raw_encodings
    if EncodeMode.TOKEN in modes:
        token_encodings = {
            f"{prefix}_token_encoding_{i_layer}": [
                sample_encoding[sample_mask].cpu().numpy() for sample_encoding, sample_mask in zip(layer_encoding, mask)
            ]
            for i_layer, layer_encoding in enumerate(encodings)
        }
        result.update(**token_encodings)

    if EncodeMode.MEAN in modes:
        pooled_encodings = {
            f"{prefix}_mean_encoding_{i_layer}": (
                torch.stack(
                    [
                        sample_encoding[sample_mask].mean(dim=0)
                        for sample_encoding, sample_mask in zip(layer_encoding, mask)
                    ],
                    dim=0,
                )
            )
            for i_layer, layer_encoding in enumerate(raw_encodings.values())
        }
        if only_last:
            pooled_encodings = {
                f"{prefix}_mean_encoding": list(pooled_encodings.values())[-1].clone()
            }  # the standard encoding is set to be the one from the last layer
        else:
            pooled_encodings[f"{prefix}_mean_encoding"] = list(pooled_encodings.values())[
                -1
            ].clone()  # the standard encoding is set to be the one from the last layer

        result.update(**pooled_encodings)

    if EncodeMode.SUM in modes:
        pooled_encodings = {
            f"{prefix}_sum_encoding_{i_layer}": (
                torch.stack(
                    [
                        sample_encoding[sample_mask].sum(dim=0)
                        for sample_encoding, sample_mask in zip(layer_encoding, mask)
                    ],
                    dim=0,
                )
            )
            for i_layer, layer_encoding in enumerate(raw_encodings.values())
        }

        if only_last:
            pooled_encodings = {
                f"{prefix}_sum_encoding": list(pooled_encodings.values())[-1].clone()
            }  # the standard encoding is set to be the one from the last layer
        else:
            pooled_encodings[f"{prefix}_sum_encoding"] = list(pooled_encodings.values())[
                -1
            ].clone()  # the standard encoding is set to be the one from the last layer

        result.update(**pooled_encodings)

    if EncodeMode.CLS in modes:
        pooled_encodings = {
            f"{prefix}_cls_encoding_{i_layer}": layer_encoding[
                :, 0, :
            ]  # TODO: adapt to encoders without CLS as first token
            for i_layer, layer_encoding in enumerate(raw_encodings.values())
        }

        if only_last:
            pooled_encodings = {
                f"{prefix}_cls_encoding": list(pooled_encodings.values())[-1].clone()
            }  # the standard encoding is set to be the one from the last layer
        else:
            pooled_encodings[f"{prefix}_cls_encoding"] = list(pooled_encodings.values())[
                -1
            ].clone()  # the standard encoding is set to be the one from the last layer

        result.update(**pooled_encodings)

    if return_tensors == "numpy":
        result = {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in result.items()}
    elif return_tensors == "pt":
        pass
    else:
        raise NotImplementedError

    return result
