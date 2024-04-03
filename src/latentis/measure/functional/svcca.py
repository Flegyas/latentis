# Code in this file is adapted from https://github.com/google/svcca/blob/master/cca_core.py
from typing import Dict, Tuple, Union
import torch
import numpy as np
from latentis.measure._metrics import preprocess_latent_space_args


@preprocess_latent_space_args
def robust_svcca(
    space1: torch.Tensor,
    space2: torch.Tensor,
    variance_percentage: float = 0.98,
    epsilon: float = 1e-4,
    num_trials: float = 5,
    tolerance: float = 1e-4,
) -> Union[float, None]:
    """Calls svcca multiple times while adding noise.

    Robust version of svcca that should be called when the latter doesn't converge for a pair of inputs.
    In practice, this function adds some noise to the activations to help convergence.

    Args:
        space1: tensor of shape (N, D) containing D-dimensional embeddings for N samples.
        space2: tensor of shape (N, D'), containing D'-dimensional embeddings for N samples.
        epsilon: small float quantifying the tolerance in the inequalities and preventing numerical issues.
        variance_percentage: float between 0, 1 used to get rid of trailing zeros in the cca correlation coefficients to output more accurate summary statistics of correlations.

    Returns:
        svcca_similarity: float quantifying the SVCCA similarity between X and Y.
    """

    result = None
    for _ in range(num_trials):

        try:
            result = svcca(
                space1, space2, variance_percentage=variance_percentage, epsilon=epsilon, tolerance=tolerance
            )

            break

        except torch.linalg.LinAlgError:
            space1 = space1 * 1e-1 + np.random.normal(size=space1.shape) * epsilon
            space2 = space2 * 1e-1 + np.random.normal(size=space2.shape) * epsilon

    return result


@preprocess_latent_space_args
def svcca(
    space1: torch.Tensor,
    space2: torch.Tensor,
    variance_percentage: float = 0.98,
    tolerance: float = 1e-4,
    epsilon: float = 1e-6,
) -> float:
    """
    Args:
        space1: tensor of shape (N, D) containing D-dimensional embeddings for N samples.
        space2: tensor of shape (N, D'), containing D'-dimensional embeddings for N samples.
        variance_percentage: float between 0, 1 used to get rid of trailing zeros in the cca correlation coefficients to output more accurate summary statistics of correlations.
        epsilon: small float used in divisions to prevent numerical issues.
        tolerance: small float quantifying the tolerance in the inequalities

    Returns:
        svcca_similarity: float quantifying the SVCCA similarity between X and Y.

    """

    assert space1.shape[0] == space2.shape[0], "space1 and space2 must have the same number of samples."

    space1, space2 = space1.T, space2.T

    num_neurons_x, num_neurons_y = space1.shape[0], space2.shape[0]

    # compute the covariance between the neurons in space1 and the neurons in space2, treating the activation for each sample as an observation
    # shape (m, m), where m = num_neurons_x + num_neurons_y
    covariance = torch.cov(torch.cat((space1, space2)))

    covariances = _decompose_and_normalize_covariance_matrix(covariance, num_neurons_x, num_neurons_y)

    (svd_decomposition, kept_indices) = _compute_ccas(covariances, epsilon=epsilon)

    if (not torch.any(kept_indices["x"])) or (not torch.any(kept_indices["y"])):
        return 0

    singular_values = svd_decomposition["s"]

    # only keep the singular values explaining threshold% of the variance
    last_significant_direction = _get_last_most_important_direction(singular_values, variance_percentage)

    singular_values_to_keep = singular_values[:last_significant_direction]

    svcca_similarity = torch.mean(singular_values_to_keep)

    assert (
        0 - tolerance <= svcca_similarity <= 1 + tolerance
    ), f"SVCCA value must be between 0 and 1, got {svcca_similarity} and tolerance is {tolerance}."

    return svcca_similarity


def _decompose_and_normalize_covariance_matrix(covariance: torch.Tensor, n_x: int, n_y: int) -> Dict[str, torch.Tensor]:
    """
    Decomposes and normalizes the covariance matrix.

    The function first decomposes the covariance matrix between X and Y as cov(X, X), cov(Y, Y) and cross-cov(X, Y), cross-cov(Y, X) and then
    properly normalizes each one to prevent numerical instability in the subsequent computations.

    Args:
        covariance: tensor of shape (m, m), where m = num_neurons_x + num_neurons_y
        n_x: number of neurons in X
        n_y: number of neurons in Y

    Returns:
        Dict containing each of the normalized components of the covariance matrix.
    """

    cov_xx, cov_yy = covariance[:n_x, :n_x], covariance[n_x:, n_x:]
    cov_xy, cov_yx = covariance[:n_x, n_x:], covariance[n_x:, :n_x]

    assert cov_xx.shape == (n_x, n_x) and cov_yy.shape == (n_y, n_y), "Covariance matrix has incorrect shape."

    xmax = torch.max(torch.abs(cov_xx)) + 1e-6
    ymax = torch.max(torch.abs(cov_yy)) + 1e-6

    cov_xx /= xmax
    cov_yy /= ymax
    cov_xy /= torch.sqrt(xmax * ymax)
    cov_yx /= torch.sqrt(xmax * ymax)

    return {"xx": cov_xx, "yy": cov_yy, "xy": cov_xy, "yx": cov_yx}


def _compute_ccas(covariances: Dict[str, torch.Tensor], epsilon: float):
    """Main cca computation function, takes in variances and crossvariances.

    This function takes in the covariances and cross covariances of X, Y,
    preprocesses them (removing small magnitudes) and outputs the raw results of
    the cca computation, including cca directions in a rotated space, and the
    cca correlation coefficient values.

    Args:
        covariance: dict containing cov_xx, cov_yy, cross-cov_xy and cross-cov_yx.
        epsilon: small float to help with stabilizing computations.

    Returns:
        svd_results: Singular Value Decomposition of sqrt(inv(cov_XX)) @ cov_XY @ sqrt(inv(cov_YY))
        kept_indices: indices of dimensions to keep
    """

    ref_tensor = covariances["xy"]

    pruned_covariances, kept_indices = _prune_small_covariances(covariances, epsilon)
    cov_xx, cov_yy = pruned_covariances["xx"], pruned_covariances["yy"]

    num_neurons_x, num_neurons_y = cov_xx.shape[0], cov_yy.shape[0]

    if num_neurons_x == 0 or num_neurons_y == 0:
        return (None, None)

    cov_xx += torch.eye(num_neurons_x).type_as(ref_tensor) * epsilon
    cov_yy += torch.eye(num_neurons_y).type_as(ref_tensor) * epsilon

    inv_xx = torch.linalg.pinv(cov_xx)
    inv_yy = torch.linalg.pinv(cov_yy)

    invsqrt_xx = matrix_sqrt_positive_definitive(inv_xx)
    invsqrt_yy = matrix_sqrt_positive_definitive(inv_yy)

    arr = invsqrt_xx @ pruned_covariances["xy"] @ invsqrt_yy

    u, s, v = torch.linalg.svd(arr)

    svd_results = {"u": u, "s": torch.abs(s), "v": v}

    return svd_results, kept_indices


def matrix_sqrt_positive_definitive(X: torch.Tensor) -> torch.Tensor:
    """Stable method for computing matrix square roots, supports complex matrices.

    Args:
        X: possibly complex valued tensor that is a positive definite symmetric (or hermitian) matrix.

    Returns:
        sqrt_x: The matrix square root of X.
    """
    w, v = torch.linalg.eigh(X)

    wsqrt = torch.sqrt(w)
    sqrt_X = v @ torch.diag(wsqrt) @ torch.conj(v).T

    return sqrt_X


def _prune_small_covariances(
    covariances: Dict[str, torch.Tensor], epsilon: float
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Takes covariance between X, Y, and removes values of small magnitude.

    Args:
        covariance: dict containing cov_xx, cov_yy, cross-cov_xy and cross-cov_yx.
        epsilon: cutoff value for norm below which directions are thrown away.

    Returns:
        pruned_covariances: dict containing pruned cov_xx, cov_yy, cross-cov_xy and cross-cov_yx.
        kept_indices: indices of dimensions that were not pruned.
    """

    x_diag, y_diag = torch.abs(torch.diagonal(covariances["xx"])), torch.abs(torch.diagonal(covariances["yy"]))

    x_idxs, y_idxs = x_diag >= epsilon, y_diag >= epsilon

    sigma_xx_crop, sigma_yy_crop = covariances["xx"][x_idxs][:, x_idxs], covariances["yy"][y_idxs][:, y_idxs]
    sigma_xy_crop, sigma_yx_crop = covariances["xy"][x_idxs][:, y_idxs], covariances["yx"][y_idxs][:, x_idxs]

    pruned_covariances = {
        "xx": sigma_xx_crop,
        "yy": sigma_yy_crop,
        "xy": sigma_xy_crop,
        "yx": sigma_yx_crop,
    }

    kept_indices = {"x": x_idxs, "y": y_idxs}

    return pruned_covariances, kept_indices


def _get_last_most_important_direction(array: torch.Tensor, percentage: float):
    """Returns the index i at which the sum of array[:i] amounts to percentage*total mass of the array.

    This function takes in a decreasing array of nonnegative floats, and a
    percentage. It returns the index i at which the sum of the array up to i is percentage*total mass of the array.

    Args:
        array: a 1D tensor of decreasing nonnegative floats
        percentage: float between 0 and 1

    Returns:
        i: index at which the cumulative sum is greater than threshold
    """
    assert (percentage >= 0) and (percentage <= 1), "threshold should be a percentage"

    for i in range(len(array)):
        if torch.sum(array[:i]) / torch.sum(array) >= percentage:
            return i
