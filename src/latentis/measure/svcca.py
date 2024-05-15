from typing import Union

import torch

from latentis.measure._metrics import Metric
from latentis.measure.functional.svcca import robust_svcca, svcca
from latentis.space import LatentSpace


class SVCCA(Metric):
    """A class for computing the Singular Vector Canonical Correlation Analysis (MCCA) between two matrices.

    Paper: https://arxiv.org/abs/1706.05806

    Attributes:
        device (torch.device): The torch device (e.g., CPU or GPU) to perform calculations on.
    """

    def __init__(
        self,
        robust: bool = True,
        variance_percentage: float = 0.98,
        device: Union[str, torch.device] = None,
        tolerance: float = 1e-4,
        epsilon: float = 1e-6,
    ):
        """Initialize the SVCCA instance with a specific mode and torch device."""
        super().__init__(SVCCA, device)

        self.device = device if device else torch.device("cpu")

        self.robust = robust

        self.tolerance = tolerance
        self.epsilon = epsilon

        self.variance_percentage = variance_percentage

    def _forward(self, space1: torch.Tensor, space2: torch.Tensor):
        """Compute the Singular Vector CCA between two spaces space1 and space2.

        Args:
            space1: shape (N, D), first embedding matrix.
            space2: shape (N, D'), second embedding matrix.

        Returns:
            float: The SVCCA similarity between space1 and space2.
        """
        if isinstance(space1, torch.Tensor) and isinstance(space2, torch.Tensor):
            space1 = space1.to(self.device)
            space2 = space2.to(self.device)

        if isinstance(space1, LatentSpace) and isinstance(space2, LatentSpace):
            space1 = LatentSpace.like(space1, vector_source=space1.vectors.to(self.device))
            space2 = LatentSpace.like(space2, vector_source=space2.vectors.to(self.device))

        svcca_fn = robust_svcca if self.robust else svcca

        return svcca_fn(
            space1=space1, space2=space2, variance_percentage=self.variance_percentage, epsilon=self.epsilon
        )
