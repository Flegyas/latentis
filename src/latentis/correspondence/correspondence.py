from __future__ import annotations

import operator
import random
from itertools import groupby
from typing import Mapping, Sequence, Union

import torch

from latentis.correspondence import Correspondence
from latentis.correspondence._base import PI

# WikimatrixCorrespondence(source_dataset="en", target_dataset="fr", source_id=0)


# IdentityCorrespondence(dataset)(source_id=0)


class TensorCorrespondence(Correspondence):
    def __init__(self, x2y: torch.Tensor):
        super().__init__()
        if x2y.ndim != 2:
            raise ValueError(f"Expected x2y to have ndim=2, got {x2y.ndim}")
        if x2y.shape[1] != 2:
            raise ValueError(f"Expected x2y to be (n, 2), got {x2y.shape}")

        self.x2y: torch.Tensor = x2y

    # def get_x_ids(self) -> torch.Tensor:
    #     return self.x2y[:, 0]

    # def get_y_ids(self) -> torch.Tensor:
    #     return self.x2y[:, 1]


class SameKeyCorrespondence(Correspondence):
    def __init__(self):
        super().__init__()

    def subset(self, x_keys: Sequence[str], y_keys: Sequence[str], size: int, seed: int = 42) -> PI:
        if len(x_keys) != len(y_keys):
            raise ValueError(f"Expected x_keys and y_keys to have the same length, got {len(x_keys)} and {len(y_keys)}")
        p = torch.randperm(len(x_keys), generator=torch.Generator().manual_seed(seed))[:size]

        return PI(
            x_indices=p,
            y_indices=p,
        )

    def match(self, x_keys: Sequence[str], y_keys: Sequence[str]) -> Union[bool, torch.BoolTensor]:
        single: bool = isinstance(x_keys, str) and isinstance(y_keys, str)

        if isinstance(x_keys, (str, int)):
            x_keys = [x_keys]
        if isinstance(y_keys, (str, int)):
            y_keys = [y_keys]

        if single:
            return x_keys[0] == y_keys[0]

        result = torch.zeros(len(x_keys), len(y_keys), dtype=torch.bool)
        for i, x_key in enumerate(x_keys):
            for j, y_key in enumerate(y_keys):
                result[i, j] = x_key == y_key

        return result


class ImageNetToTextCorrespondence(Correspondence):
    def match(self, x_keys: Sequence[str], y_keys: Sequence[str]) -> Union[bool, torch.BoolTensor]:
        single: bool = isinstance(x_keys, str) and isinstance(y_keys, str)

        if single:
            x_keys = [x_keys]
            y_keys = [y_keys]

        x_synsets = [key.split("_")[0] for key in x_keys]
        y_synsets = [key.split("_")[0] for key in y_keys]

        if single:
            return x_synsets[0] == y_synsets[0]

        result = torch.zeros(len(x_synsets), len(y_synsets), dtype=torch.bool)
        for i, x_synset in enumerate(x_synsets):
            for j, y_synset in enumerate(y_synsets):
                result[i, j] = x_synset == y_synset

        return result

    def subset(
        self, x_keys: Sequence[str], y_keys: Sequence[str], size: int, seed: int = 42
    ) -> Mapping[str, Sequence[int]]:
        y_synsets = [key.split("_")[0] for key in y_keys]

        y_synset2indices = groupby(enumerate(y_synsets), key=operator.itemgetter(1))

        x_indices = torch.randperm(len(x_keys), generator=torch.Generator().manual_seed(seed))[:size]
        x_synsets = [x_keys[i].split("_")[0] for i in x_indices]

        random.seed(42)

        y_indices = [random.choice(y_synset2indices[x_synset]) for x_synset in x_synsets]

        return PI(
            x_indices=torch.as_tensor(x_indices),
            y_indices=torch.as_tensor(y_indices),
        )
