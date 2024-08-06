from __future__ import annotations

from typing import Mapping, Sequence, Union

import torch

from latentis.correspondence import Correspondence
from latentis.correspondence._base import PI

# WikimatrixCorrespondence(source_dataset="en", target_dataset="fr", source_id=0)


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

    def match(self, x_keys: Sequence[str], y_keys: Sequence[str], mode: str = "first") -> Union[bool, torch.BoolTensor]:
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
                if x_key == y_key:
                    result[i, j] = True
                    if mode == "first":
                        break

        return result


class ImageNetToTextCorrespondence(Correspondence):
    def __init__(self, max_captions: int):
        super().__init__()
        self.max_captions: int = max_captions

    def match(self, x_keys: Sequence[str], y_keys: Sequence[str], mode="first") -> Union[bool, torch.BoolTensor]:
        if isinstance(x_keys, (str, int)):
            x_keys = [x_keys]
        if isinstance(y_keys, (str, int)):
            y_keys = [y_keys]

        x_synsets = [key.split("_")[0] for key in x_keys]
        y_synsets = [key.split("_")[0] for key in y_keys]

        if len(x_synsets) == 1 and len(y_synsets) == 1:
            return x_synsets[0] == y_synsets[0]

        result = torch.zeros(len(x_synsets), len(y_synsets), dtype=torch.bool)
        for i, x_synset in enumerate(x_synsets):
            for j, y_synset in enumerate(y_synsets):
                if x_synset == y_synset:
                    result[i, j] = True
                    if mode == "first":
                        break
        return result.bool()  # .squeeze(dim=(0, 1))

    def subset1(
        self, x_keys: Sequence[str], y_keys: Sequence[str], size: int, seed: int = 42
    ) -> Mapping[str, Sequence[int]]:
        x_indices = torch.randperm(len(x_keys), generator=torch.Generator().manual_seed(seed))[:size]

        x_synsets = [x_keys[i].split("_")[0] for i in x_indices]
        y_synsets = [key.split("_")[0] for key in y_keys]

        synset2y_index = {synset: i for i, synset in enumerate(y_synsets)}

        y_indices = torch.as_tensor([synset2y_index[synset] for synset in x_synsets])

        return PI(
            x_indices=x_indices,
            y_indices=y_indices,
        )

    def subset(
        self, x_keys: Sequence[str], y_keys: Sequence[str], size: int, seed: int = 42
    ) -> Mapping[str, Sequence[int]]:
        y_synsets = [key.split("_")[0] for key in y_keys]

        y_synset2indices = {key: [] for key in set(y_synsets)}
        for i, synset_id in enumerate(y_synsets):
            y_synset2indices[synset_id].append(i)

        y_synset2indices = {key: torch.as_tensor(indices) for key, indices in y_synset2indices.items()}
        # TODO: reproducibility
        y_synset2indices = {
            key: indices[torch.randperm(len(indices))][: self.max_captions] for key, indices in y_synset2indices.items()
        }

        x_indices = torch.randperm(len(x_keys), generator=torch.Generator().manual_seed(seed))[:size]
        x_synsets = [x_keys[i].split("_")[0] for i in x_indices]

        synset2y_occurrences = torch.as_tensor([len(y_synset2indices[synset]) for synset in x_synsets])

        # random.seed(42)
        # y_indices = [random.choice(y_synset2indices[x_synset]) for x_synset in x_synsets]

        y_indices = torch.as_tensor([index for x_synset in x_synsets for index in y_synset2indices[x_synset]])

        return PI(
            x_indices=x_indices.repeat_interleave(synset2y_occurrences),
            y_indices=y_indices,
        )
