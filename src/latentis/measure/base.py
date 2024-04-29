import functools
from enum import auto

import torch

from latentis.measure._abstract import PairwiseMetric
from latentis.measure.functional.cka import cka, kernel_hsic, linear_hsic
from latentis.space import Space
from latentis.types import StrEnum


class CKAMode(StrEnum):
    """Modes for Centered Kernel Alignment (CKA).

    Attributes:
        LINEAR: linear mode CKA.
        RBF: Radial Basis Function (RBF) mode CKA.
    """

    LINEAR = auto()
    RBF = auto()


class CKA(PairwiseMetric):
    """A class for computing Centered Kernel Alignment (CKA) between two matrices.

    Paper https://arxiv.org/abs/1905.00414

    This class supports both linear and RBF kernel methods for computing CKA.

    Attributes:
        mode (CKAMode): The mode of CKA (linear or RBF).
        device (torch.device): The torch device (e.g., CPU or GPU) to perform calculations on.
    """

    def __init__(self, mode: CKAMode, device: torch.device = None, tolerance: float = 1e-6, sigma=None):
        """Initialize the CKA instance with a specific mode and torch device."""
        super().__init__(CKA)

        self.mode = mode
        if self.mode == CKAMode.LINEAR:
            self.hsic = linear_hsic
        elif self.mode == CKAMode.RBF:
            self.hsic = functools.partial(kernel_hsic, sigma=sigma)
        else:
            raise NotImplementedError(f"CKA mode {self.mode} is not implemented.")

        self.device = device if device else torch.device("cpu")

        # to avoid numerical issues in the assertions
        self.tolerance = tolerance

        self.sigma = sigma

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """Compute the CKA between two spaces x and y.

        Depending on the mode, it either computes linear or RBF kernel based CKA.

        Args:
            x: shape (N, D), first embedding matrix.
            y: shape (N, D'), second embedding matrix.
            sigma: Optional parameter for RBF kernel.

        Returns:
            Computed CKA value.
        """
        if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
            x = x.to(self.device)
            y = y.to(self.device)
        if isinstance(x, Space) and isinstance(y, Space):
            x = Space.like(x, vector_source=x.vectors.to(self.device))
            y = Space.like(y, vector_source=y.vectors.to(self.device))

        return cka(x=x, y=y, hsic=self.hsic, tolerance=self.tolerance)

    def to(self, device):
        """Move the CKA instance to a specific torch device.

        Args:
            device: The torch device (e.g., CPU or GPU) to move the instance to.

        Returns:
            The CKA instance on the specified device.
        """
        self.device = device
        return self
