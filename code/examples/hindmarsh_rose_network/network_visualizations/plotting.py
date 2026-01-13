"""
Reusable NetworkX visualizations for the SREINet example notebooks.
"""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


CUSTOM_COLORS = ["#B8DBB3", "#719AAC"]


def _normalize_edge_list(edge_list: Iterable[Iterable[int]]) -> np.ndarray:
    array = np.asarray(list(edge_list), dtype=int)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("edge_list must be convertible to an (m, 2) integer array.")
    return array


def _draw_graph(
    graph: nx.Graph,
    pos: dict[int, tuple[float, float]],
    ax: plt.Axes | None = None,
    node_labels: dict[int, str] | None = None,
    *,
    node_size: int = 100,
    node_color: Iterable[str] | str = "tab:blue",
    node_cmap: str | plt.Colormap | None = None,
    edge_width: float = 1,
    edge_alpha: float = 0.4,
    edge_color: str = "lightgray",
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    labels = node_labels if node_labels is not None else {node: str(node) for node in graph.nodes}
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_color=node_color,
        node_size=node_size,
        alpha=0.85,
        cmap=node_cmap,
    )
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color=edge_color, width=edge_width, alpha=edge_alpha)
    nx.draw_networkx_labels(
        graph,
        pos,
        labels=labels,
        ax=ax,
        font_size=8,
        font_color="0.25",
        font_family="Helvetica",
    )

    ax.set_axis_off()
    return ax


def build_complete_edge_list(n_nodes: int) -> np.ndarray:
    """
    Return a 0-based edge list for an undirected complete graph K_n.
    """

    if n_nodes < 2:
        raise ValueError("n_nodes must be at least 2 for a connected graph.")

    edge_list = [(i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes)]
    return np.asarray(edge_list, dtype=int)


def plot_hindmarsh_network(
    edge_list: Iterable[Iterable[int]],
    n_nodes: int,
    ax: plt.Axes | None = None,
    layout_seed: int = 42,
) -> plt.Axes:
    """
    Draw the Hindmarsh–Rose network using the spring layout from the example notebook.
    """

    normalized_edges = _normalize_edge_list(edge_list)
    graph = nx.Graph()
    graph.add_nodes_from(range(n_nodes))
    graph.add_edges_from(normalized_edges.tolist())

    pos = nx.spring_layout(graph, seed=layout_seed, k=0.75)
    return _draw_graph(
        graph,
        pos,
        ax=ax,
        node_color=CUSTOM_COLORS[0],
        node_cmap=None,
        node_size=120,
        edge_width=0.6,
        edge_alpha=0.8,
    )


def plot_kuramoto_complete_network(
    n_nodes: int,
    ax: plt.Axes | None = None,
    node_order: Iterable[int] | None = None,
    label_increment: int = 4,
    label_start: int = 0,
) -> plt.Axes:
    """
    Draw the fully connected Kuramoto network on a circle.
    """

    if label_increment < 1:
        raise ValueError("label_increment must be >= 1.")

    edge_list = build_complete_edge_list(n_nodes)
    graph = nx.Graph()
    graph.add_nodes_from(range(n_nodes))
    graph.add_edges_from(edge_list.tolist())

    if node_order is not None:
        order_list = list(node_order)
        if len(order_list) != n_nodes:
            raise ValueError("node_order must have exactly n_nodes entries.")
        if set(order_list) != set(graph.nodes):
            raise ValueError("node_order must contain each node exactly once.")
        pos = nx.circular_layout(order_list)
    else:
        pos = nx.circular_layout(graph)

    current_label = label_start
    label_map = {}
    for node in graph.nodes:
        label_map[node] = str(current_label)
        current_label += label_increment
    return _draw_graph(
        graph,
        pos,
        ax=ax,
        node_color=CUSTOM_COLORS[1],
        node_labels=label_map,
        node_size=120,
        edge_width=0.6,
        edge_alpha=0.4,
        edge_color="#999999",
    )
