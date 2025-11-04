"""
Lorenz network dynamics with pairwise diffusive coupling on the x coordinate.
"""

from __future__ import annotations

import numpy as np

from coupling import diffusive_coupling

def lorenz_network(
    t: float,
    state: np.ndarray,
    edge_list: np.ndarray,
    coupling_strength: float = 1e-1,
) -> np.ndarray:
    """
    Coupled Lorenz oscillators over an undirected network.

    Parameters
    ----------
    t:
        Time variable (required by `solve_ivp`, not used explicitly).
    state:
        Concatenated state vector with layout [x_0..x_{N-1}, y_0.., z_0..].
    edge_list:
        0-based undirected edge list describing the network topology.
    coupling_strength:
        Diffusive coupling coefficient applied to the x variable.

    Returns
    -------
    derivatives:
        Time derivatives d/dt of the state vector.
    """

    n_nodes = state.size // 3

    x = state[:n_nodes]
    y = state[n_nodes : 2 * n_nodes]
    z = state[2 * n_nodes :]

    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0

    coupling = diffusive_coupling(x, edge_list)

    dx = sigma * (y - x) + coupling_strength * coupling
    dy = x * (rho - z) - y
    dz = x * y - beta * z

    return np.concatenate([dx, dy, dz])
