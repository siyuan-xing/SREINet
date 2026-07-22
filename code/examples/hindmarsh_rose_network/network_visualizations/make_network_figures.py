"""
Generate publication-ready network visualizations for the Hindmarsh–Rose
and Kuramoto examples.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

FS_SMALL = 10
FS_MEDIUM = 12
FS_LARGE = 14

plt.rcParams.update(
    {
        "font.family": "Helvetica",
        "font.size": FS_SMALL,
        "axes.labelsize": FS_MEDIUM,
        "axes.titlesize": FS_MEDIUM,
        "axes.titleweight": "bold",
        "xtick.labelsize": FS_SMALL,
        "ytick.labelsize": FS_SMALL,
        "legend.fontsize": FS_SMALL,
    }
)

EXAMPLES_DIR = Path(__file__).resolve().parents[2]
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.append(str(EXAMPLES_DIR))

from hindmarsh_rose_network.network_generator import generate_erdos_renyi_network  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from plotting import (  # noqa: E402
    plot_hindmarsh_network,
    plot_kuramoto_complete_network,
)


def create_figure(
    hindmarsh_nodes: int = 25,
    hindmarsh_edge_probability: float = 0.2,
    hindmarsh_seed: int = 42,
    kuramoto_nodes: int = 15,
    figsize: tuple[float, float] = (7.0, 2.15),
) -> tuple[plt.Figure, list[plt.Axes]]:
    """
    Build a side-by-side figure (each subplot 4x3 inches) for the manuscript.
    """

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    edge_list, _, _ = generate_erdos_renyi_network(
        hindmarsh_nodes,
        hindmarsh_edge_probability,
        seed=hindmarsh_seed,
    )
    plot_hindmarsh_network(edge_list, hindmarsh_nodes, ax=axes[0])

    plot_kuramoto_complete_network(kuramoto_nodes, ax=axes[1])

    return fig, list(axes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "network_panel.png",
        help="Where to save the figure (default: network_panel.png next to this script).",
    )
    parser.add_argument("--dpi", type=int, default=600, help="Output resolution.")
    parser.add_argument("--hindmarsh-nodes", type=int, default=25, help="Number of Hindmarsh–Rose nodes.")
    parser.add_argument("--hindmarsh-edge-prob", type=float, default=0.2, help="Edge probability for ER graph.")
    parser.add_argument("--hindmarsh-seed", type=int, default=42, help="Seed controlling the ER graph.")
    parser.add_argument("--kuramoto-nodes", type=int, default=15, help="Number of Kuramoto oscillators.")
    args = parser.parse_args()

    fig, _ = create_figure(
        hindmarsh_nodes=args.hindmarsh_nodes,
        hindmarsh_edge_probability=args.hindmarsh_edge_prob,
        hindmarsh_seed=args.hindmarsh_seed,
        kuramoto_nodes=args.kuramoto_nodes,
        figsize=(7.0, 2.15),
    )
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi)
    print(f"Wrote {output_path} (dpi={args.dpi})")


if __name__ == "__main__":
    main()
