from __future__ import annotations

import torch
from torch._tensor import Tensor

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

    def get_x_ids(self) -> torch.Tensor:
        return self.x2y[:, 0]

    def get_y_ids(self) -> torch.Tensor:
        return self.x2y[:, 1]


class IdentityCorrespondence(Correspondence):
    def __init__(self, n_samples: int):
        super().__init__()
        self.n_samples: int = n_samples

    def get_x_ids(self) -> Tensor:
        return torch.arange(self.n_samples)

    def get_y_ids(self) -> Tensor:
        return torch.arange(self.n_samples)


if __name__ == "__main__":
    pass
