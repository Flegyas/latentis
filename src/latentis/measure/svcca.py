import torch

from latentis.measure._metrics import Metric
from latentis.measure.functional.svcca import robust_svcca, svcca


class SVCCA(Metric):
    """A class for computing the Singular Vector Canonical Correlation Analysis (MCCA) between two matrices.

    Paper: https://arxiv.org/abs/1706.05806
    """

    def __init__(
        self,
        robust: bool = True,
        variance_percentage: float = 0.98,
        tolerance: float = 1e-4,
        epsilon: float = 1e-6,
    ):
        """Initialize the SVCCA instance with a specific mode and torch device."""
        super().__init__(SVCCA)
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
        if space1.device != space2.device:
            raise ValueError(
                f"space1 and space2 must be on the same device. Found {space1.device} and {space2.device}"
            )

        svcca_fn = robust_svcca if self.robust else svcca

        return svcca_fn(
            space1=space1,
            space2=space2,
            variance_percentage=self.variance_percentage,
            epsilon=self.epsilon,
            tolerance=self.tolerance,
        )
