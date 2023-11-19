import latentis
from latentis.measure._metrics import Metric
import torch 
import math
from enum import auto

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
            self.hsic = self.linear_HSIC
        elif self.mode == CKAMode.RBF:
            self.hsic = self.kernel_HSIC
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
            X: shape (N, D), first embedding matrix.
            Y: shape (N, D'), second embedding matrix.
            sigma: Optional parameter for RBF kernel.

        Returns:
            Computed CKA value.
        """

        if isinstance(space1, latentis.LatentSpace):
            space1 = space1.vectors

        if isinstance(space2, latentis.LatentSpace):
            space2 = space2.vectors

        assert space1.shape[0] == space2.shape[0], "X and Y must have the same number of samples."
        
        space1 = space1.to(self.device)
        space2 = space2.to(self.device)

        hsic = self.hsic(space1, space2, sigma)

        var1 = torch.sqrt(self.hsic(space1, space1, sigma))
        var2 = torch.sqrt(self.hsic(space2, space2, sigma))

        cka_result = hsic / (var1 * var2)
        
        assert 0 - self.tolerance <= cka_result <= 1 + self.tolerance , "CKA value must be between 0 and 1."

        return cka_result
        
    def linear_HSIC(self, X: torch.Tensor, Y: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Compute HSIC for linear kernels.

        This method is used in the computation of linear CKA.

        Args:
            X, Y: Input matrices.

        Returns:
            The computed HSIC value.
        """

        # inter-sample similarity matrices for both spaces ~(N, N)
        L_X = X @ X.T
        L_Y = Y @ Y.T

        return torch.sum(self.centering(L_X) * self.centering(L_Y))
    
    def kernel_HSIC(self, X: torch.Tensor, Y:torch.Tensor, sigma):
        """
        Compute HSIC (Hilbert-Schmidt Independence Criterion) for RBF kernels.

        This is used in the computation of kernel CKA.

        Args:
            X, Y: Input matrices.
            sigma: The RBF kernel width.

        Returns:
            The computed HSIC value.
        """
        return torch.sum(self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma)))

    def centering(self, K: torch.Tensor) -> torch.Tensor:
        """
        Center the kernel matrix K using the centering matrix H = I_n - (1/n) 1 * 1^T. (Eq. 3 in the paper)

        This method is used in the calculation of HSIC.

        Args:
            K: The kernel matrix to be centered.

        Returns:
            The centered kernel matrix.
        """
        n = K.shape[0]
        unit = torch.ones([n, n]).type_as(K) 
        identity_mat = torch.eye(n).type_as(K)
        H = identity_mat - unit / n

        return H @ K @ H

    def rbf(self, X: torch.Tensor, sigma=None):
        """
        Compute the RBF (Radial Basis Function) kernel for a matrix X.

        If sigma is not provided, it is computed based on the median distance.

        Args:
            X: The input matrix.
            sigma: Optional parameter to specify the RBF kernel width.

        Returns:
            The RBF kernel matrix.
        """

        GX = X @ X.T
        KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
        
        if sigma is None:
            mdist = torch.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        
        KX *= -0.5 / (sigma * sigma)
        KX = torch.exp(KX)

        return KX

def CKAFn(space1: torch.Tensor, space2: torch.Tensor, mode: CKAMode, device: torch.device = None, sigma=None):
    """
    Compute the Centered Kernel Alignment (CKA) between two spaces.

    Args:
        space1: First embedding matrix or LatentSpace object.
        space2: Second embedding matrix or LatentSpace object.
        mode: The mode of CKA (CKAMode.LINEAR or CKAMode.RBF).
        device: The torch device (e.g., CPU or GPU) to perform calculations on. Defaults to None.
        sigma: Optional parameter for RBF kernel. Only used when mode is CKAMode.RBF.

    Returns:
        The computed CKA value.
    """
    cka = CKA(mode, device)

    return cka._forward(space1, space2, sigma)
