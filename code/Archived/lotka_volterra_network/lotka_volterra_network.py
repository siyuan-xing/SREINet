"""
Lotka–Volterra predator–prey network with diffusive coupling.
"""

from __future__ import annotations

import numpy as np

from coupling import diffusive_coupling, higher_order_diffusive_coupling


def lotka_volterra_network(
    t: float,
    state: np.ndarray,
    edge_list: np.ndarray,
    triangle_list: np.ndarray | None = None,
    higher_order_coupling_strength: float = 0.0,
    alpha: float = 1.5,
    beta: float = 1.0,
    delta: float = 1.0,
    gamma: float = 3.0,
) -> np.ndarray:
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
    triangle_list:
        Optional array of shape (m, 3) listing higher-order interactions.
    coupling_strength:
        Diffusive coupling coefficient applied to both prey and predator variables.
    higher_order_coupling_strength:
        Coupling coefficient for the three-body (triangle-based) interactions.
    alpha, beta, delta, gamma:
        Local Lotka–Volterra parameters.

    Returns
    -------
    derivatives:
        Time derivatives of the concatenated state vector.
    """

    n_species = 2
    if state.size % n_species != 0:
        raise ValueError("State vector length must be divisible by the number of species (2).")

    n_nodes = state.size // n_species
    populations = state.reshape(n_species, n_nodes)

    # f_i(x_i) part in Eq. (5): species-specific linear growth/decay.
    local_rates = np.array([alpha, -gamma], dtype=populations.dtype).reshape(n_species, 1)
    local_dynamics = local_rates * populations

    # a_ij^{(1)} x_i x_j part in Eq. (5): pairwise Lotka–Volterra interactions.
    interaction_matrix = np.array(
        [
            [0.0, -beta],  # prey loses mass to predator encounters
            [delta, 0.0],  # predator gains mass from prey encounters
        ],
        dtype=populations.dtype,
    )
    pairwise_interactions = np.zeros_like(populations)
    for i in range(n_species):
        for j in range(n_species):
            pairwise_interactions[i] += interaction_matrix[i, j] * populations[i] * populations[j]


    # Higher-order (three-body) coupling via simplices, if provided.
    higher_order_terms = np.vstack(
        [higher_order_diffusive_coupling(populations[i], triangle_list) for i in range(n_species)]
    )

    derivatives = (
        local_dynamics
        + pairwise_interactions
        + higher_order_coupling_strength * higher_order_terms
    )
    return derivatives.reshape(-1)
