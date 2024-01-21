from abc import abstractmethod

import torch

from latentis.data.dataset import LatentisDataset
from latentis.space._base import EncodingKey

# WikimatrixCorrespondence(source_dataset="en", target_dataset="fr", source_id=0)


# IdentityCorrespondence(dataset)(source_id=0)
class Correspondence:
    def __init__(self, x_dataset: LatentisDataset, y_dataset: LatentisDataset):
        self.x_dataset = x_dataset
        self.y_dataset = y_dataset

    @abstractmethod
    def get_fit_vectors(self, encoding_key: EncodingKey) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_test_vectors(self, encoding_key: EncodingKey) -> torch.Tensor:
        raise NotImplementedError


class SameDatasetCorrespondence(Correspondence):
    def __init__(self, dataset: LatentisDataset, perc: float, seed: int):
        super().__init__(x_dataset=dataset, y_dataset=dataset)
        assert 0 < perc <= 1, f"perc must be in (0, 1], got {perc}"

        self.perc = perc
        self.seed = seed
        train_size: int = len(dataset.hf_dataset["train"])
        self.fit_ids = torch.randperm(train_size, generator=torch.Generator().manual_seed(seed))[
            : int(train_size * perc)
        ]

    def get_fit_vectors(self, encoding_key: EncodingKey) -> torch.Tensor:
        assert (
            encoding_key.dataset == self.x_dataset._name
        ), f"Dataset mismatch in encoding key {encoding_key.dataset} and {self.x_dataset.name}"

        space = encoding_key.to_space(data_root=self.x_dataset.root_dir.parent)

        return space[self.fit_ids]

    def get_test_vectors(self, encoding_key: EncodingKey) -> torch.Tensor:
        assert (
            encoding_key.dataset == self.x_dataset.name
        ), f"Dataset mismatch in encoding key {encoding_key.dataset} and {self.x_dataset.name}"

        space = encoding_key.to_space(data_root=self.x_dataset.root_dir.parent)

        return space.vectors
