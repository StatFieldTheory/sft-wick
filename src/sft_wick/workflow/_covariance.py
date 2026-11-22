"""Numerical validation shared by initial laws and Markov embeddings."""

import numpy as np


def is_covariance(matrix: np.ndarray) -> bool:
    """Test positivity of a symmetric matrix at each component's scale."""
    if not np.all(np.isfinite(matrix)):
        return False
    variance = np.diag(matrix)
    active = variance > 0.0
    if np.any(variance < 0.0) or np.any(matrix[~active]):
        return False
    if not np.any(active):
        return True
    std = np.sqrt(variance[active])
    largest = np.maximum(std[:, None], std[None, :])
    smallest = np.minimum(std[:, None], std[None, :])
    with np.errstate(over="ignore"):
        correlation = matrix[np.ix_(active, active)] / largest / smallest
    return bool(
        np.all(np.isfinite(correlation)) and np.linalg.eigvalsh(correlation)[0] >= -1e-12
    )
