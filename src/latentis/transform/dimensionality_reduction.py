import torch


def pca(
    x: torch.Tensor, k: int, return_weights: bool = False, return_recon: bool = False
):
    """Perform Principal Component Analysis (PCA) on the input data.

    Args:
        x (torch.Tensor): Input data of shape (n_samples, dim).
        k (int): Number of components.

    Returns:
        dict: Dictionary containing the components, and optionally the weights and the reconstruction.ƒ

    """
    # Center the data by subtracting the mean of each dimension
    x_centered = x - torch.mean(x, dim=0)

    # Compute the SVD of the centered data
    U, S, Vt = torch.linalg.svd(x_centered, full_matrices=False)

    # Select the number of components we want to keep
    components = Vt[:k]

    result = {}

    result["components"] = components
    if return_weights:
        weights = S[:k]
        result["weights"] = weights
    if return_recon:
        recon = x_centered @ components.T @ components + torch.mean(x, dim=0)
        result["recon"] = recon

    return result
