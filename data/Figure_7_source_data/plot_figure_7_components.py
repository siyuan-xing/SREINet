"""Reproduce the three component panels used to assemble manuscript Figure 7."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "reproduced"
FS_SMALL = 10
FS_MEDIUM = 12
FS_LARGE = 14

matplotlib.rcParams.update(
    {
        "image.origin": "lower",
        "image.interpolation": "nearest",
        "axes.grid": False,
        "axes.labelsize": FS_MEDIUM,
        "axes.titlesize": FS_LARGE,
        "font.size": FS_SMALL,
        "legend.fontsize": FS_SMALL,
        "xtick.labelsize": FS_SMALL,
        "ytick.labelsize": FS_SMALL,
        "font.family": "Helvetica",
    }
)


def draw_network(graph: nx.Graph, positions, axis, color, labels, *, edge_alpha: float, edge_color: str) -> None:
    nx.draw_networkx_nodes(graph, positions, ax=axis, node_color=color, node_size=120, alpha=0.85)
    nx.draw_networkx_edges(graph, positions, ax=axis, edge_color=edge_color, width=0.6, alpha=edge_alpha)
    nx.draw_networkx_labels(graph, positions, labels=labels, ax=axis, font_size=10, font_color="0.25", font_family="Helvetica")
    axis.set_axis_off()


def plot_network_panel() -> None:
    with np.load(HERE / "hindmarsh_rose_plot_data.npz") as data:
        edges = data["edge_list"]
        hindmarsh_nodes = int(data["network_nodes"])
    kuramoto_nodes = 15
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.15), constrained_layout=True)

    graph = nx.Graph()
    graph.add_nodes_from(range(hindmarsh_nodes))
    graph.add_edges_from(edges.tolist())
    positions = nx.spring_layout(graph, seed=42, k=0.75)
    draw_network(
        graph,
        positions,
        axes[0],
        "#B8DBB3",
        {node: str(node) for node in graph.nodes},
        edge_alpha=0.8,
        edge_color="lightgray",
    )

    complete = nx.complete_graph(kuramoto_nodes)
    positions = nx.circular_layout(complete)
    labels = {node: str(node * 4) for node in complete.nodes}
    draw_network(complete, positions, axes[1], "#719AAC", labels, edge_alpha=0.4, edge_color="#999999")

    output = OUTPUT / "network_panel.png"
    fig.savefig(output, dpi=600)
    plt.close(fig)
    print(f"Wrote {output.relative_to(HERE)}")


if __name__ == "__main__":
    OUTPUT.mkdir(exist_ok=True)
    from plot_hindmarsh_rose_panel import main as plot_hindmarsh_rose
    from plot_kuramoto_panel import main as plot_kuramoto

    plot_hindmarsh_rose()
    print("Wrote reproduced/HindmarshRose_75D_state_comparison.png")
    plot_kuramoto()
    print("Wrote reproduced/Kuramoto_60D_state_comparison.png")
    plot_network_panel()
