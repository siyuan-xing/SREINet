"""
Helpers for sampling higher-order (simplicial) network structures.
"""

from __future__ import annotations

import itertools
from typing import Dict, Tuple

import numpy as np

from network_generator import generate_erdos_renyi_network, build_adjacency_matrix


def generate_erdos_renyi_simplicial(
    n_nodes: int,
    edge_probability: float,
    triangle_probability: float,
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, int]]:
    """
    Sample an Erdős–Rényi graph and augment it with random 3-body interactions.

    Parameters
    ----------
    n_nodes:
        Number of nodes in the graph.
    edge_probability:
        Probability of each pairwise edge.
    triangle_probability:
        Probability of each triple being marked as a higher-order interaction.
    seed:
        Optional RNG seed for reproducibility.

    Returns
    -------
    edge_list:
        Array of shape (m, 2) with pairwise edges.
    triangle_list:
        Array of shape (r, 3) with node indices forming 3-body interactions.
    adjacency:
        Dense adjacency matrix of the pairwise network.
    degrees:
        Degree dictionary for the pairwise network.
    """

    rng = np.random.default_rng(seed)
    edge_list, adjacency, degrees = generate_erdos_renyi_network(
        n_nodes=n_nodes,
        edge_probability=edge_probability,
        seed=seed,
    )

    triangles = []
    for combo in itertools.combinations(range(n_nodes), 3):
        if rng.random() < triangle_probability:
            triangles.append(combo)

    triangle_list = np.asarray(triangles, dtype=int) if triangles else np.empty((0, 3), dtype=int)

    return edge_list, triangle_list, adjacency, degrees


def build_triangle_incidence(triangle_list: np.ndarray, n_nodes: int) -> np.ndarray:
    """
    Construct a node-by-triangle incidence matrix.
    """

    incidence = np.zeros((n_nodes, len(triangle_list)), dtype=int)
    for tri_idx, (i, j, k) in enumerate(triangle_list):
        incidence[i, tri_idx] = 1
        incidence[j, tri_idx] = 1
        incidence[k, tri_idx] = 1
    return incidence

