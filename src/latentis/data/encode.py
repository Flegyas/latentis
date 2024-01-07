import functools
from collections import defaultdict
from typing import Sequence, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from transformers.feature_extraction_utils import PreTrainedFeatureExtractor

from latentis.data.processor import LatentisDataset
from latentis.data.text_encoding import EncodeMode, batch_encode

BASE_TEXT_ENCODERS = [
    "bert-base-cased",
    "bert-base-uncased",
    "google/electra-base-discriminator",
    "roberta-base",
    "albert-base-v2",
    "xlm-roberta-base",
    "openai/clip-vit-base-patch32",
]

BASE_IMAGE_ENCODERS = [
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
]


def load_text_encoder(model_name: str) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    encoder = AutoModel.from_pretrained(model_name, output_hidden_states=True, return_dict=True)
    encoder.requires_grad_(False).eval()
    return encoder, AutoTokenizer.from_pretrained(model_name)


def load_image_encoder(model_name: str) -> Tuple[PreTrainedModel, PreTrainedFeatureExtractor]:
    encoder = AutoModel.from_pretrained(model_name, output_hidden_states=True, return_dict=True)
    encoder.requires_grad_(False).eval()
    return encoder, AutoFeatureExtractor.from_pretrained(model_name)


@torch.no_grad()
def collate_fn(batch, tokenizer, max_length: int, data_key: str, encoder_name: str, id_column: str):
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


def encode_text(
    dataset: LatentisDataset,
    encoding_batch_size: int,
    store_every: int,
    num_workers: int,
    force_recompute: bool,
    encoders: Sequence[str],
    device: torch.device,
):
    missing_encoders = [
        encoder for encoder in encoders if force_recompute or encoder not in dataset.get_available_encodings()
    ]

    for encoder_name in tqdm(missing_encoders, desc=f"{missing_encoders}"):
        encoder, tokenizer = load_text_encoder(encoder_name)
        encoder = encoder.to(device)

        for split, split_data in dataset.dataset.items():
            loader = DataLoader(
                split_data,
                batch_size=encoding_batch_size,
                pin_memory=True,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=functools.partial(
                    collate_fn,
                    tokenizer=tokenizer,
                    max_length=encoder.config.max_length,
                    data_key=dataset.features[0].col_name,
                    encoder_name=encoder_name,
                    id_column=dataset.id_column,
                ),
            )

            result = defaultdict(list)
            for batch_index, batch in tqdm(enumerate(loader), desc="Encoding samples"):
                if batch_index % store_every == 0:
                    for encoding_key, encoding in result.items():
                        if encoding_key == dataset.id_column:
                            continue
                        encoding = torch.cat(encoding, dim=0).cpu()

                        dataset.add_encoding(
                            encoding_key=encoding_key,
                            split=split,
                            id2encoding=dict(zip(result[dataset.id_column], encoding)),
                        )

                    result = defaultdict(list)

                batch_encoding = batch_encode(
                    encoding=batch["tokenizer_result"].to(device),
                    encoder=encoder,
                    prefix=encoder_name,
                    modes=[EncodeMode.MEAN],
                    only_last=True,
                )

                for k, v in batch_encoding.items():
                    result[k].append(v)

                result[dataset.id_column].extend(batch[dataset.id_column])

            for encoding_key, encoding in result.items():
                if encoding_key == dataset.id_column:
                    continue
                encoding = torch.cat(encoding, dim=0).cpu()

                dataset.add_encoding(
                    encoding_key=encoding_key,
                    split=split,
                    id2encoding=dict(zip(result[dataset.id_column], encoding)),
                )

        encoder.cpu()


if __name__ == "__main__":
    dataset = LatentisDataset.load_from_disk("imdb", perc=0.01)
    print(dataset.dataset)

    encode_text(
        dataset=dataset,
        encoding_batch_size=32,
        store_every=10,
        num_workers=4,
        force_recompute=True,
        encoders=["bert-base-cased"],
        device=torch.device("cpu"),
    )

if __name__ == "__main__":
    dataset = LatentisDataset.load_from_disk("imdb", 0.01)
    print("original dataset", dataset.dataset)
    dataset = dataset.with_encodings(["bert-base-cased_mean_encoding"])
    print(dataset)

# if __name__ == "__main__":
#     mnist = MNIST()
#     print(mnist)

#     dataset = mnist.process(0.01)

#     print(dataset.dataset["train"][0]["image"].shape)
