import latentis
from latentis.measure._metrics import Metric
import torch 
import math
from enum import auto

from latentis.measure.functional import cka, kernel_hsic, linear_hsic

try:
    # be ready for 3.10 when it drops
    from enum import StrEnum
except ImportError:
    from backports.strenum import StrEnum

from enum import auto
import torch
import math

class CKAMode(StrEnum):
    """
    Modes for Centered Kernel Alignment (CKA).

    Attributes:
        LINEAR: linear mode CKA.
        RBF: Radial Basis Function (RBF) mode CKA.
    """
    LINEAR = auto()
    RBF = auto()

class CKA(Metric):
    """
    A class for computing Centered Kernel Alignment (CKA) between two matrices.
    Paper https://arxiv.org/abs/1905.00414

    This class supports both linear and RBF kernel methods for computing CKA.

    Attributes:
        mode (CKAMode): The mode of CKA (linear or RBF).
        device (torch.device): The torch device (e.g., CPU or GPU) to perform calculations on.
    """

    def __init__(self, mode: CKAMode, device: torch.device = None):
        """Initialize the CKA instance with a specific mode and torch device."""

        super().__init__(CKA)
        
        self.mode = mode 
        if self.mode == CKAMode.LINEAR:
            self.hsic = linear_hsic
        elif self.mode == CKAMode.RBF:
            self.hsic = kernel_hsic
        else:
            raise NotImplementedError(f"No such mode {self.mode}")

        self.device = device

        # to avoid numerical issues in the assertions
        self.tolerance = 1e-6

    def _forward(self, space1: torch.Tensor, space2: torch.Tensor, sigma=None):
        """
        Compute the CKA between two spaces space1 and space2.

        Depending on the mode, it either computes linear or RBF kernel based CKA.

        Args:
            space1: shape (N, D), first embedding matrix.
            space2: shape (N, D'), second embedding matrix.
            sigma: Optional parameter for RBF kernel.

        Returns:
            Computed CKA value.
        """

        return cka(space1=space1, space2=space2, hsic=self.hsic, sigma=sigma, device=self.device, tolerance=self.tolerance)
    
    def to(self, device):
        """
        Move the CKA instance to a specific torch device.

        Args:
            device: The torch device (e.g., CPU or GPU) to move the instance to.

        Returns:
            The CKA instance on the specified device.
        """
        self.device = device
        return self




