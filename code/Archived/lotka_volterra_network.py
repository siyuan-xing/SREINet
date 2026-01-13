"""
Lotka–Volterra predator–prey network with diffusive coupling.
"""

from __future__ import annotations

import numpy as np

from coupling import diffusive_coupling


def lotka_volterra_network(
    t,
    state,
    edge_list,
    coupling_strength_prey = 1e-1,
    coupling_strength_predator = 1e-1,
    alpha = 1.5,
    beta = 1.0,
    delta = 1.0,
    gamma = 3.0,
    coupling_type = 'diffusive',
):
    """
    Coupled Lotka–Volterra oscillators over an undirected network.

    Parameters
    ----------
    t:
        Time variable (unused, present for `solve_ivp` compatibility).
    state:
        Concatenated state vector [prey_0..prey_{N-1}, predator_0..].
    edge_list:
        0-based undirected edge list describing the network topology.
    coupling_strength_prey:
        Diffusive coupling coefficient for the prey variable.
    coupling_strength_predator:
        Diffusive coupling coefficient for the predator variable.
    alpha, beta, delta, gamma:
        Local Lotka–Volterra parameters.

    Returns
    -------
    derivatives:
        Time derivatives of the concatenated state vector.
    """

    n_nodes = state.size // 2

    prey = state[:n_nodes]
    predator = state[n_nodes:]

    prey_coupling = diffusive_coupling(prey, edge_list)
    predator_coupling = diffusive_coupling(predator, edge_list)

    dprey = alpha * prey - beta * prey * predator + coupling_strength_prey * prey_coupling
    dpredator = delta * prey * predator - gamma * predator + coupling_strength_predator * predator_coupling

    return np.concatenate([dprey, dpredator])

