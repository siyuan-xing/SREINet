"""
Network helpers for Hindmarsh–Rose simulations.
"""

from __future__ import annotations

import numpy as np
import networkx as nx


def generate_erdos_renyi_network(
    n_nodes: int,
    edge_probability: float,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    """
    Sample a connected Erdős–Rényi (G(n, p)) graph.
    """

    rng = np.random.default_rng(seed)
    next_seed = None if seed is None else int(rng.integers(low=0, high=2**32 - 1))

    graph = nx.erdos_renyi_graph(n_nodes, edge_probability, seed=next_seed, directed=False)

    if not nx.is_connected(graph):
        attempt = 1
        while not nx.is_connected(graph):
            next_seed = None if seed is None else int(rng.integers(low=0, high=2**32 - 1))
            graph = nx.erdos_renyi_graph(n_nodes, edge_probability, seed=next_seed, directed=False)
            attempt += 1
            if attempt > 50:
                raise RuntimeError(
                    "Failed to sample a connected Erdős–Rényi graph after 50 attempts. "
                    "Increase `edge_probability`."
                )

    edge_list = np.asarray(graph.edges(), dtype=int)
    adjacency = nx.to_numpy_array(graph)
    degrees = dict(graph.degree())

    return edge_list, adjacency, degrees


def build_adjacency_matrix(edge_list: np.ndarray, n_nodes: int) -> np.ndarray:
    """
    Build an adjacency matrix from a 0-based edge list.
    """

    adjacency = np.zeros((n_nodes, n_nodes))
    for i, j in edge_list:
        adjacency[i, j] = 1.0
        adjacency[j, i] = 1.0
    return adjacency

