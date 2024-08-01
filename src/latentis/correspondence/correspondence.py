from __future__ import annotations

from typing import Sequence, Union

import torch

from latentis.correspondence import Correspondence

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

        x_keys = [key.split("_")[-1] for key in x_keys]
        y_keys = [key.split("_")[-1] for key in y_keys]

        if single:
            return x_keys[0] == y_keys[0]

        result = torch.zeros(len(x_keys), len(y_keys), dtype=torch.bool)
        for i, x_key in enumerate(x_keys):
            for j, y_key in enumerate(y_keys):
                result[i, j] = x_key == y_key

        return result
