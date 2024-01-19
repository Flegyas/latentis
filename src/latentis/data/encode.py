import functools
import logging
import shutil
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from latentis.data import DATA_DIR
from latentis.data.dataset import Feature
from latentis.data.processor import LatentisDataset
from latentis.data.text_encoding import HFPooling, cls_pool, mean_pool, sum_pool
from latentis.data.utils import default_collate
from latentis.modules import LatentisModule, PooledModel, TextHFEncoder

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


# encoding_module = encoder.vision_model if encoder_name.startswith("openai/clip") else encoder
# for batch in tqdm(loader, desc=f"Embedding samples"):
#     embeddings.extend(encoding_module(batch["image"].to(DEVICE)).cpu().tolist())

# return embeddings


@dataclass
class EncodingSpec:
    model: LatentisModule
    collate_fn: callable
    encoding_batch_size: int
    store_every: int
    num_workers: int
    save_source_model: bool = False
    poolers: Sequence[nn.Module] = field(default_factory=list)


def encode_feature(
    dataset: LatentisDataset,
    feature: Feature,
    encoding_spec: EncodingSpec,
    device: torch.device,
):
    for split, split_data in dataset._hf_dataset.items():
        model: LatentisModule = encoding_spec.model

        loader = DataLoader(
            split_data,
            batch_size=encoding_spec.encoding_batch_size,
            pin_memory=device != torch.device("cpu"),
            shuffle=False,
            num_workers=encoding_spec.num_workers,
            collate_fn=functools.partial(encoding_spec.collate_fn, model=model, feature=feature.col_name),
        )

        for batch in tqdm(loader, desc=f"Encoding `{split}` samples for feature {feature} using {model.key}"):
            raw_encoding = model.encode(batch)

            if len(encoding_spec.poolers) == 0:
                assert isinstance(raw_encoding, torch.Tensor)
                dataset.add_encoding(
                    feature=feature,
                    encoder_key=model.key,
                    encoding_key="no_pooling",
                    split=split,
                    vectors=raw_encoding,
                    keys=batch[dataset._id_column],
                    model=model,
                    save=encoding_spec.save_source_model,
                )
            else:
                for pooler in encoding_spec.poolers:
                    encodings = pooler(**raw_encoding)

                    for encoding_key, vectors in encodings.items():
                        dataset.add_encoding(
                            feature=feature,
                            encoder_key=model.key,
                            encoding_key=encoding_key,
                            split=split,
                            vectors=vectors,
                            keys=batch[dataset._id_column],
                            model=PooledModel(model_key=model.key, model=model, pooler=pooler),
                            save_source_model=encoding_spec.save_source_model,
                        )


def encode_dataset(
    dataset: LatentisDataset,
    feature2encoding_specs: Mapping[str, Sequence[EncodingSpec]],
    force_recompute: bool,
    device: torch.device,
):
    """Encode the dataset using the specified encoders.

    Args:
        dataset (LatentisDataset): Dataset to encode.
        feature2encoding_specs (Mapping[str, Sequence[EncodingSpec]]): Mapping from feature name to encoding specs.
        force_recompute (bool): Whether to recompute the encodings even if they already exist.
        device (torch.device): The torch device (e.g., CPU or GPU) to perform calculations on.
    """
    assert len(feature2encoding_specs) > 0, "No feature to encode specified"
    feature2encoding_specs = {
        dataset.get_feature(feature_name): encoding_specs
        for feature_name, encoding_specs in feature2encoding_specs.items()
    }

    if force_recompute:
        encodings_dir = dataset.root_dir / "encodings"
        if encodings_dir.exists():
            # TODO: delete only the encodings specified and not the whole dataset
            pylogger.warning(f"Overwriting existing encodings at {encodings_dir}")

            shutil.rmtree(encodings_dir)

    pylogger.info(
        f"Encoding {len(feature2encoding_specs)} features for dataset {dataset._name}: {feature2encoding_specs.keys()}"
    )

    for feature, encoding_specs in feature2encoding_specs.items():
        feature: Feature

        for encoding_spec in tqdm(encoding_specs, desc=f"Encoding feature {feature}"):
            # model: Union[Encoder, EncoderDecoder] = encoding_spec.resolve_model()

            # model.to(device)

            encode_feature(
                dataset=dataset,
                feature=feature,
                encoding_spec=encoding_spec,
                device=device,
            )

            # model.cpu()


if __name__ == "__main__":
    dataset = LatentisDataset.load_from_disk(DATA_DIR / "imdb")
    print(dataset.hf_dataset)

    encode_dataset(
        dataset=dataset,
        force_recompute=True,
        device=torch.device("cpu"),
        feature2encoding_specs={
            "text": [
                EncodingSpec(
                    model=TextHFEncoder("bert-base-cased"),
                    collate_fn=default_collate,
                    encoding_batch_size=32,
                    store_every=10,
                    num_workers=0,
                    save_source_model=False,
                    poolers=[
                        HFPooling(layers=-1, pooling_fn=cls_pool),
                        HFPooling(layers=-1, pooling_fn=mean_pool),
                    ],
                ),
                EncodingSpec(
                    model=TextHFEncoder("bert-base-uncased"),
                    collate_fn=default_collate,
                    encoding_batch_size=32,
                    store_every=7,
                    num_workers=0,
                    save_source_model=True,
                    poolers=[
                        HFPooling(layers=-1, pooling_fn=sum_pool),
                    ],
                ),
            ]
        },
    )
