"""
Hindmarsh–Rose neuron dynamics.
"""

import numpy as np


def hindmarsh_rose_network(
    t: float,
    state: np.ndarray,
    edge_list: np.ndarray,
    coupling_strength: float = 0.05,
    a: float = 1.0,
    b: float = 3.0,
    c: float = 1.0,
    d: float = 5.0,
    r: float = 0.05,
    s: float = 4.0,
    x0: float = -1.618,
    I: float = 3.2,
) -> np.ndarray:
    """
    Coupled Hindmarsh–Rose neurons with pairwise diffusive coupling on x.
    """

    n_nodes = state.size // 3
    x = state[:n_nodes]
    y = state[n_nodes : 2 * n_nodes]
    z = state[2 * n_nodes :]

    coupling = diffusive_coupling(x, edge_list)

    dx = y - a * x**3 + b * x**2 - z + I + coupling_strength * coupling
    dy = c - d * x**2 - y
    dz = r * (s * (x - x0) - z)

    return np.concatenate([dx, dy, dz])


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