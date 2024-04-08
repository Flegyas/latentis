import functools
import itertools
import logging
from typing import Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from latentis.benchmark.task import Task
from latentis.data import DATA_DIR
from latentis.data.dataset import Feature, HFDatasetView
from latentis.data.processor import DatasetView
from latentis.data.text_encoding import HFPooler, Pooler, cls_pool
from latentis.data.utils import default_collate
from latentis.nn import LatentisModule
from latentis.nn.encoders import TextHFEncoder
from latentis.space import Space
from latentis.space.vector_source import HDF5Source

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


class IdentityPooling(Pooler):
    def __init__(self, output_dim: int):
        super().__init__(name="no_pooling", output_dim=output_dim)

    def forward(self, x):
        return [(x, {})]


class EncodeTask(Task):
    def __init__(
        self,
        dataset: DatasetView,
        feature: str,
        model: LatentisModule,
        collate_fn: callable,
        encoding_batch_size: int,
        num_workers: int,
        device: torch.device,
        save_source_model: bool = False,
        poolers: Sequence[nn.Module] = None,
    ):
        super().__init__()
        self.dataset = dataset
        self.feature: Feature = dataset.get_feature(feature)

        self.model = model
        self.collate_fn = collate_fn
        self.encoding_batch_size = encoding_batch_size
        self.num_workers = num_workers
        self.device = device
        self.save_source_model = save_source_model

        if poolers is None or len(poolers) == 0:
            poolers = [IdentityPooling(output_dim=self.model.output_dim)]
        self.poolers = poolers

    def _run(
        self,
    ):
        """Encode the dataset using the specified encoders.

        Args:
            dataset (LatentisDataset): Dataset to encode
            feature (str): Name of the feature to encode
            model (LatentisModule): Model to use for encoding
            collate_fn (callable): Collate function to use for the dataset
            encoding_batch_size (int): Batch size for encoding
            store_every (int): Store encodings every `store_every` batches
            num_workers (int): Number of workers to use for encoding
            device (torch.device): Device to use for encoding
            save_source_model (bool, optional): Save the source model. Defaults to False.
            poolers (Sequence[nn.Module], optional): Poolers to use for encoding. Defaults to None.

        """
        pylogger.info(f"Encoding {self.feature} for dataset {dataset._name}")

        model = self.model.to(self.device)

        for split, split_data in dataset.hf_dataset.items():
            pooler2space = {
                pooler: Space(
                    vector_source=HDF5Source(
                        num_elements=len(split_data),
                        dimension=pooler.output_dim,
                        root_dir=DATA_DIR / dataset.name / "encodings" / split / self.feature.name / pooler.name,
                    ),
                    properties={
                        **{f"model/{key}": value for key, value in model.properties.items()},
                        "feature": self.feature.name,
                        "split": split,
                        "dataset": dataset.name,
                        "pooler": pooler.__class__.__name__,
                    },
                )
                for pooler in self.poolers
            }

            loader = DataLoader(
                split_data,
                batch_size=self.encoding_batch_size,
                pin_memory=self.device != torch.device("cpu"),
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=functools.partial(self.collate_fn, model=model, feature=self.feature.name),
            )

            for batch in tqdm(
                loader, desc=f"Encoding `{split}` samples for feature {self.feature.name} using {model.item_id[:8]}"
            ):
                raw_encoding = model.encode(batch.to(self.device))

                for pooler in self.poolers:
                    encoding2pooler_properties = pooler(**raw_encoding)

                    for encoding, pooler_properties in encoding2pooler_properties:
                        pooler2space[pooler].add_vectors(
                            vectors=encoding, keys=batch[dataset._id_column].cpu().tolist(), write=True
                        )

        model.cpu()


if __name__ == "__main__":
    for dataset_name, hf_encoder in itertools.product(["trec"], ["bert-base-cased"]):
        dataset = HFDatasetView.load_from_disk(DATA_DIR / dataset_name)

        space = EncodeTask(
            dataset=dataset,
            feature="text",
            model=TextHFEncoder(hf_encoder),
            collate_fn=default_collate,
            encoding_batch_size=256,
            num_workers=2,
            save_source_model=False,
            poolers=[
                HFPooler(layers=[12], pooling_fn=cls_pool, output_dim=768),
            ],
            device=torch.device("cpu"),
        ).run()
