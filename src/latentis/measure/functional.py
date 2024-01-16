import functools
import math

import torch

import latentis


def linear_cka(x: torch.Tensor, y: torch.Tensor):
    return cka(x, y, hsic=linear_hsic)


def rbf_cka(x: torch.Tensor, y: torch.Tensor, *, sigma: float = None):
    return cka(x, y, hsic=functools.partial(kernel_hsic, sigma=sigma))


def cka(x: torch.Tensor, y: torch.Tensor, *, hsic: callable, tolerance=1e-6):
    if isinstance(x, latentis.LatentSpace):
        x = x.vectors

    if isinstance(y, latentis.LatentSpace):
        y = y.vectors

    assert x.shape[0] == y.shape[0], "X and Y must have the same number of samples."

    numerator = hsic(x, y)

    var1 = torch.sqrt(hsic(x, x))
    var2 = torch.sqrt(hsic(y, y))

    cka_result = numerator / (var1 * var2)

    assert 0 - tolerance <= cka_result <= 1 + tolerance, "CKA value must be between 0 and 1."

    return cka_result


def linear_hsic(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute HSIC for linear kernels.

    This method is used in the computation of linear CKA.

    Args:
        X: shape (N, D), first embedding matrix.
        Y: shape (N, D'), second embedding matrix.

    Returns:
        The computed HSIC value.
    """
    # inter-sample similarity matrices for both spaces ~(N, N)
    L_X = x @ x.T
    L_Y = y @ y.T

    return torch.sum(center_kernel_matrix(L_X) * center_kernel_matrix(L_Y))


def kernel_hsic(x: torch.Tensor, y: torch.Tensor, *, sigma):
    """Compute HSIC (Hilbert-Schmidt Independence Criterion) for RBF kernels.

    This is used in the computation of kernel CKA.

    Args:
        X: shape (N, D), first embedding matrix.
        Y: shape (N, D'), second embedding matrix.
        sigma: The RBF kernel width.

    Returns:
        The computed HSIC value.
    """
    return torch.sum(center_kernel_matrix(rbf(x, sigma=sigma)) * center_kernel_matrix(rbf(y, sigma=sigma)))


def center_kernel_matrix(k: torch.Tensor) -> torch.Tensor:
    """Center the kernel matrix K using the centering matrix H = I_n - (1/n) 1 * 1^T. (Eq. 3 in the paper).

    This method is used in the calculation of HSIC.

    Args:
        K: The kernel matrix to be centered.

    Returns:
        The centered kernel matrix.
    """
    n = k.shape[0]
    unit = torch.ones([n, n]).type_as(k)
    identity_mat = torch.eye(n).type_as(k)
    H = identity_mat - unit / n

    return H @ k @ H


def rbf(x: torch.Tensor, *, sigma=None):
    """Compute the RBF (Radial Basis Function) kernel for a matrix X.

    If sigma is not provided, it is computed based on the median distance.

    Args:
        X: The input matrix (num_samples, embedding_dim).
        sigma: Optional parameter to specify the RBF kernel width.

    Returns:
        The RBF kernel matrix.
    """
    GX = x @ x.T
    KX = torch.diag(GX).type_as(x) - GX + (torch.diag(GX) - GX).T

    if sigma is None:
        mdist = torch.median(KX[KX != 0])
        sigma = math.sqrt(mdist)

    KX *= -0.5 / (sigma * sigma)
    KX = torch.exp(KX)

    return KX
