import torch


def pca(
    x: torch.Tensor,
    k: int,
    return_weights: bool = False,
    return_recon: bool = False,
    return_variance: bool = False,
):
    """Perform Principal Component Analysis (PCA) on the input data.

    Args:
        x (torch.Tensor): Input data of shape (n_samples, dim).
        k (int): Number of components.
        return_weights (bool, optional): Whether to return the singular values (weights). Defaults to False.
        return_recon (bool, optional): Whether to return the reconstruction of the input data. Defaults to False.
        return_variance (bool, optional): Whether to return the explained variance and explained variance ratio. Defaults to False.

    Returns:
        dict: Dictionary containing the components, weights, explained variance, explained variance ratio, and reconstruction of the input data.
    """
    # Center the data by subtracting the mean of each dimension
    x_centered = x - torch.mean(x, dim=0)

    # Compute the SVD of the centered data
    U, S, Vt = torch.linalg.svd(x_centered, full_matrices=False)

    # Select the number of components we want to keep
    components = Vt[:k]

    result = {}
    result["components"] = components

    # Optionally return the singular values (weights)
    if return_weights:
        weights = S[:k]
        result["weights"] = weights

    # Optionally return the explained variance and explained variance ratio
    if return_variance:
        # Compute explained variance
        n_samples = x.shape[0]
        explained_variance = (S**2) / (n_samples - 1)
        explained_variance_ratio = explained_variance / torch.sum(explained_variance)

        result["explained_variance"] = explained_variance[:k]
        result["explained_variance_ratio"] = explained_variance_ratio[:k]

    # Optionally return the reconstruction of the input data
    if return_recon:
        recon = x_centered @ components.T @ components + torch.mean(x, dim=0)
        result["recon"] = recon

    return result
