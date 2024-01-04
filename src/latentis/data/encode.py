import functools
import logging
from collections import defaultdict
from typing import Mapping, Sequence, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from transformers.feature_extraction_utils import PreTrainedFeatureExtractor

from latentis.data.dataset import DataType, Feature
from latentis.data.processor import LatentisDataset
from latentis.data.text_encoding import EncodeMode, batch_encode

pylogger = logging.getLogger(__name__)

DEFAULT_ENCODERS = {
    "text": [
        "bert-base-cased",
        "bert-base-uncased",
        "google/electra-base-discriminator",
        "roberta-base",
        "albert-base-v2",
        "xlm-roberta-base",
        "openai/clip-vit-base-patch32",
    ],
    "image": [
        "rexnet_100",
        "vit_base_patch16_224",
        "vit_base_patch16_384",
        "vit_base_resnet50_384",
        "vit_small_patch16_224",
        "openai/clip-vit-base-patch32",
        "efficientnet_b1_pruned",
        "regnety_002",
        "cspresnext50",
        "cspdarknet53",
    ],
}


def load_text_encoder(model_name: str) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    encoder = AutoModel.from_pretrained(model_name, output_hidden_states=True, return_dict=True)
    encoder.requires_grad_(False).eval()
    return encoder, AutoTokenizer.from_pretrained(model_name)


def load_image_encoder(model_name: str) -> Tuple[PreTrainedModel, PreTrainedFeatureExtractor]:
    encoder = AutoModel.from_pretrained(model_name, output_hidden_states=True, return_dict=True)
    encoder.requires_grad_(False).eval()
    return encoder, AutoFeatureExtractor.from_pretrained(model_name)


@torch.no_grad()
def collate_fn_text(batch, tokenizer, max_length: int, data_key: str, encoder_name: str, id_column: str):
    is_clip: bool = encoder_name.startswith("openai/clip")
    tokenizer_result = tokenizer(
        [sample[data_key] for sample in batch],
        return_special_tokens_mask=True,
        return_token_type_ids=not is_clip,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    )

    sample_ids = [sample[id_column] for sample in batch]

    return {
        "tokenizer_result": tokenizer_result,
        id_column: sample_ids,
    }


@torch.no_grad()
def collate_fn_image(batch, processor, data_key: str, encoder_name: str, id_column: str):
    pass


@torch.no_grad()
def encode_image(dataset, transform, encoder, encoder_name: str, batch_size=64):
    raise NotImplementedError
    # embeddings = []
    # loader = DataLoader(
    #     dataset,
    #     batch_size=batch_size,
    #     pin_memory=True,
    #     shuffle=False,
    #     num_workers=4,
    #     collate_fn=functools.partial(
    #         clip_image_encode if encoder_name.startswith("openai/clip") else image_encode,
    #         transform=transform,
    #     ),
    # )

    # encoding_module = encoder.vision_model if encoder_name.startswith("openai/clip") else encoder
    # for batch in tqdm(loader, desc=f"Embedding samples"):
    #     embeddings.extend(encoding_module(batch["image"].to(DEVICE)).cpu().tolist())

    # return embeddings


def encode_feature(
    dataset: LatentisDataset,
    feature: Feature,
    collate_fn: callable,
    encode_fn: callable,
    encoder_name: str,
    encoder: PreTrainedModel,
    encoding_batch_size: int,
    store_every: int,
    num_workers: int,
    device: torch.device,
):
    for split, split_data in dataset.dataset.items():
        loader = DataLoader(
            split_data,
            batch_size=encoding_batch_size,
            pin_memory=device != torch.device("cpu"),
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

        result = defaultdict(list)
        for batch_index, batch in tqdm(
            enumerate(loader), desc=f"Encoding samples for feature {feature} using {encoder_name}"
        ):
            batch_encoding = encode_fn(batch=batch, encoder=encoder, encoder_name=encoder_name)

            for k, v in batch_encoding.items():
                result[k].append(v)

            result[dataset.id_column].extend(batch[dataset.id_column])

            if (batch_index + 1) % store_every == 0 or batch_index == len(loader) - 1:
                for encoder_name, encoding in result.items():
                    if encoder_name == dataset.id_column:
                        continue
                    encoding = torch.cat(encoding, dim=0).cpu()

                    dataset.add_encoding(
                        feature=feature,
                        encoding_key=encoder_name,
                        split=split,
                        id2encoding=dict(zip(result[dataset.id_column], encoding)),
                    )

                result = defaultdict(list)

        assert len(result) == 0


def encode_dataset(
    dataset: LatentisDataset,
    encoding_batch_size: int,
    store_every: int,
    num_workers: int,
    force_recompute: bool,
    device: torch.device,
    features: Sequence[Feature] = None,
    data_type2encoders: Mapping[str, Sequence[str]] = DEFAULT_ENCODERS,
    # modality2collate_fn: Mapping[str, callable] = DEFAULT_COLLATES,
):
    """Encode the dataset using the specified encoders.

    Args:
        dataset (LatentisDataset): Dataset to encode.
        encoding_batch_size (int): Batch size for the encoding.
        store_every (int): Store the encodings every `store_every` batches.
        num_workers (int): Number of workers for the encoding.
        force_recompute (bool): Whether to recompute the encodings even if they are already present.
        device (torch.device): Device to use for the encoding.
        features (Sequence[Feature], optional): Features to encode. Defaults to None. If None, all the features are encoded.
        data_type2encoders (Mapping[str, Sequence[str]], optional): Encoders to use for each modality. Defaults to DEFAULT_ENCODERS.
    """
    if features is None:
        features = dataset.features
    else:
        assert all(feature in dataset.features for feature in features)

    pylogger.info(f"Encoding {len(features)} features for dataset {dataset.name}: {features}")
    print(features)
    for feature in features:
        print(feature)
        feature_type = feature.data_type
        if feature_type not in data_type2encoders:
            pylogger.warning(f"Missing encoders for data type `{feature_type}`. Skipping feature `{feature}`.")
            continue

        missing_encoders = [
            encoder
            for encoder in data_type2encoders[feature_type]
            if force_recompute or encoder not in dataset.get_available_encodings(features=feature)
        ]

        for encoder_name in tqdm(missing_encoders, desc=f"{missing_encoders}"):
            if feature_type == DataType.TEXT:
                encoder, tokenizer = load_text_encoder(encoder_name)
                encoder = encoder.to(device)
                collate_fn = functools.partial(
                    collate_fn_text,
                    tokenizer=tokenizer,
                    max_length=encoder.config.max_length,
                    data_key=feature.col_name,
                    encoder_name=encoder_name,
                    id_column=dataset.id_column,
                )
                encode_fn = functools.partial(
                    batch_encode,
                    encoder=encoder,
                    encoder_name=encoder_name,
                    modes=[EncodeMode.MEAN],
                    only_last=True,
                )

            elif feature_type == DataType.IMAGE:
                encoder, processor = load_image_encoder(encoder_name)
                collate_fn = functools.partial(
                    collate_fn_image,
                    processor=processor,
                    data_key=feature.col_name,
                    encoder_name=encoder_name,
                    id_column=dataset.id_column,
                )
                encode_fn = functools.partial(
                    encode_image,
                    transform=processor,
                    encoder=encoder,
                    encoder_name=encoder_name,
                )
            else:
                raise ValueError(f"Unknown feature modality {feature_type}")

            encoder.to(device)

            encode_feature(
                dataset=dataset,
                feature=feature,
                collate_fn=collate_fn,
                encode_fn=encode_fn,
                encoder_name=encoder_name,
                encoder=encoder,
                encoding_batch_size=encoding_batch_size,
                store_every=store_every,
                num_workers=num_workers,
                device=device,
            )

            encoder.cpu()


if __name__ == "__main__":
    dataset = LatentisDataset.load_from_disk("imdb", perc=0.01)
    print(dataset.dataset)

    encode_dataset(
        dataset=dataset,
        encoding_batch_size=32,
        store_every=10,
        num_workers=4,
        force_recompute=True,
        device=torch.device("cpu"),
        data_type2encoders={DataType.TEXT: ["bert-base-cased"]},
    )

# if __name__ == "__main__":
#     dataset = LatentisDataset.load_from_disk("imdb", 0.01)
#     print("original dataset", dataset.dataset)
#     dataset = dataset.with_encodings(["bert-base-cased_mean_encoding"])
#     print(dataset)
