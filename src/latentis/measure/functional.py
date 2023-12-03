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


def linear_cka(space1: torch.Tensor, space2: torch.Tensor):
    return cka(space1, space2, hsic=linear_hsic)

def rbf_cka(space1: torch.Tensor, space2: torch.Tensor, sigma:float=None):
    return cka(space1, space2, hsic=kernel_hsic, sigma=sigma)

def cka(space1: torch.Tensor, space2: torch.Tensor, hsic: callable, sigma:float=None, device=None, tolerance=1e-6):

    if isinstance(space1, latentis.LatentSpace):
        space1 = space1.vectors

    if isinstance(space2, latentis.LatentSpace):
        space2 = space2.vectors

    assert space1.shape[0] == space2.shape[0], "X and Y must have the same number of samples."

    space1 = space1.to(device)
    space2 = space2.to(device)

    numerator = hsic(space1, space2, sigma)

    var1 = torch.sqrt(hsic(space1, space1, sigma))
    var2 = torch.sqrt(hsic(space2, space2, sigma))

    cka_result = numerator / (var1 * var2)

    assert 0 - tolerance <= cka_result <= 1 + tolerance , "CKA value must be between 0 and 1."

    return cka_result

def linear_hsic(X: torch.Tensor, Y: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """
    Compute HSIC for linear kernels.

    This method is used in the computation of linear CKA.

    Args:
        X: shape (N, D), first embedding matrix.
        Y: shape (N, D'), second embedding matrix.

    Returns:
        The computed HSIC value.
    """

    # inter-sample similarity matrices for both spaces ~(N, N)
    L_X = X @ X.T
    L_Y = Y @ Y.T

    return torch.sum(center_kernel_matrix(L_X) * center_kernel_matrix(L_Y))

def kernel_hsic(X: torch.Tensor, Y:torch.Tensor, sigma):
    """
    Compute HSIC (Hilbert-Schmidt Independence Criterion) for RBF kernels.

    This is used in the computation of kernel CKA.

    Args:
        X: shape (N, D), first embedding matrix.
        Y: shape (N, D'), second embedding matrix.
        sigma: The RBF kernel width.

    Returns:
        The computed HSIC value.
    """
    return torch.sum(center_kernel_matrix(rbf(X, sigma)) * center_kernel_matrix(rbf(Y, sigma)))

def center_kernel_matrix(K: torch.Tensor) -> torch.Tensor:
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

def rbf(X: torch.Tensor, sigma=None):
    """
    Compute the RBF (Radial Basis Function) kernel for a matrix X.

    If sigma is not provided, it is computed based on the median distance.

    Args:
        X: The input matrix (num_samples, embedding_dim).
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
