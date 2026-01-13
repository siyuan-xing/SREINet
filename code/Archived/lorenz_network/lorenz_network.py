"""
Lorenz network dynamics with pairwise diffusive coupling on the x coordinate.
"""

from __future__ import annotations

import numpy as np


def lorenz_network(
    t,
    state,
    edge_list,
    coupling_strength = 1e-1,
    coupling_type = 'diffusive',
):
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

    if coupling_type == 'diffusive':
        coupling = diffusive_coupling(x, edge_list)
    elif coupling_type == 'phase':
        coupling = phase_coupling(x, edge_list)
    elif coupling_type == 'non-diffusive':
        coupling = non_diffusive_coupling(x, edge_list)
    elif coupling_type == 'quadratic':
        coupling = quadratic_coupling(x, edge_list)
    else:
        raise ValueError(f"Invalid coupling type: {coupling_type}")

    dx = sigma * (y - x) + coupling_strength * coupling
    dy = x * (rho - z) - y
    dz = x * y - beta * z

    return np.concatenate([dx, dy, dz])



def diffusive_coupling(values, edge_list):
    """
    Compute diffusive coupling for a scalar variable on each node.

    The returned vector corresponds to (A - D) @ values, which is the
    action of the graph Laplacian on the state.
    """

    coupling = np.zeros_like(values)
    for i, j in edge_list:
        diff = values[j] - values[i]
        coupling[i] += diff
        coupling[j] -= diff
    return coupling

#add phase coupling
def phase_coupling(values, edge_list):
    """
    Compute phase coupling for a scalar variable on each node.
    """
    coupling = np.zeros_like(values)
    for i, j in edge_list:
        diff = np.sin(values[j] - values[i])
        coupling[i] += diff
        coupling[j] -= diff
    return coupling

#add non-diffusive coupling
def non_diffusive_coupling(values, edge_list):
    """
    Compute non-diffusive coupling for a scalar variable on each node.
    """
    coupling = np.zeros_like(values)
    for i, j in edge_list:
        coupling[i] += np.sin(values[j])
        coupling[j] += np.sin(values[i])
    return coupling

def quadratic_coupling(values, edge_list):
    """
    Compute quadratic coupling for a scalar variable on each node.
    """
    coupling = np.zeros_like(values)
    for i, j in edge_list:
        coupling[i] += (values[j]-values[i])**2
        coupling[j] += (values[i]-values[j])**2
    return coupling