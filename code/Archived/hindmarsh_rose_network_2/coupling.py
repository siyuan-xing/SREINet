"""Simple diffusive coupling helper."""

from __future__ import annotations

import numpy as np


def diffusive_coupling(values: np.ndarray, edge_list: np.ndarray) -> np.ndarray:
    """
    Compute the Laplacian coupling (pairwise diffusive) for a scalar field.
    """

    coupling = np.zeros_like(values)
    for i, j in edge_list:
        diff = values[j] - values[i]
        coupling[i] += diff
        coupling[j] -= diff
    return coupling

