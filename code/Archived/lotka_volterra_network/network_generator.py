"""
Utilities for generating and visualizing homogeneous random networks.

The Rossler example shipped with the repo loads a fixed higher-order
network from disk.  For the Lorenz and Lotka–Volterra variants we work
with Erdős–Rényi (ER) networks instead, so this module centralises the
network generation logic.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, Tuple


def generate_erdos_renyi_network(
    n_nodes: int,
    edge_probability: float,
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, int]]:
    """
    Generate an undirected Erdős–Rényi (G(n, p)) network.

    Parameters
    ----------
    n_nodes:
        Number of nodes in the network.
    edge_probability:
        Probability of each possible edge being included.
    seed:
        Optional random seed for reproducibility.

    Returns
    -------
    edge_list:
        Array with shape (m, 2) describing the undirected edges.
    adjacency:
        Dense adjacency matrix with shape (n_nodes, n_nodes).
    degree_dict:
        Dictionary mapping node index to degree.
    """

    rng = np.random.default_rng(seed)
    # NetworkX consumes numpy random bit generators only via integers, so
    # sample a seed beforehand to keep behaviour reproducible.
    nx_seed = None if seed is None else int(rng.integers(low=0, high=2**32 - 1))

    graph = nx.erdos_renyi_graph(n_nodes, edge_probability, seed=nx_seed, directed=False)

    if not nx.is_connected(graph):
        # The downstream simulations rely on each node being influenced by others.
        # If the sampled graph is disconnected, resample until we obtain a connected instance.
        attempt = 1
        while not nx.is_connected(graph):
            nx_seed = None if seed is None else int(rng.integers(low=0, high=2**32 - 1))
            graph = nx.erdos_renyi_graph(n_nodes, edge_probability, seed=nx_seed, directed=False)
            attempt += 1
            if attempt > 50:
                raise RuntimeError(
                    "Failed to sample a connected Erdős–Rényi graph after 50 attempts. "
                    "Try increasing `edge_probability`."
                )

    edge_list = np.array(graph.edges())
    adjacency = nx.to_numpy_array(graph)
    degree_dict = dict(graph.degree())

    return edge_list, adjacency, degree_dict


def build_adjacency_matrix(edge_list: np.ndarray, n_nodes: int) -> np.ndarray:
    """Construct an adjacency matrix from a 0-based edge list."""
    adjacency = np.zeros((n_nodes, n_nodes))
    for i, j in edge_list:
        adjacency[i, j] = 1.0
        adjacency[j, i] = 1.0
    return adjacency


def visualize_network(edge_list: np.ndarray, n_nodes: int, ax: plt.Axes | None = None) -> plt.Axes:
    """
    Visualize an ER network using a spring layout.

    Parameters
    ----------
    edge_list:
        0-based edge list.
    n_nodes:
        Number of nodes in the network.
    ax:
        Optional Matplotlib axes to draw on.

    Returns
    -------
    ax:
        The axes containing the drawn network.
    """

    graph = nx.Graph()
    graph.add_nodes_from(range(n_nodes))
    graph.add_edges_from(edge_list.tolist())

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    pos = nx.spring_layout(graph, seed=42)
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color="tab:blue", node_size=200, alpha=0.85)
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color="lightgray", width=1.5, alpha=0.7)
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=8, font_color="white")

    ax.set_title(f"Connected Erdős–Rényi network (n={n_nodes}, m={len(edge_list)})")
    ax.set_axis_off()

    return ax
