"""
Command-line entry point for Hindmarsh–Rose network simulations.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import json

from .simulation import simulate_hindmarsh_rose_network, plot_overview
from .network_generator import build_adjacency_matrix


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate coupled Hindmarsh–Rose neurons on a random network.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-nodes", type=int, default=20, help="Number of neurons.")
    parser.add_argument("--edge-probability", type=float, default=0.25, help="Erdős–Rényi edge probability.")
    parser.add_argument("--seed", type=int, default=101, help="Random seed.")
    parser.add_argument("--tmax", type=float, default=80.0, help="Simulation horizon.")
    parser.add_argument("--dt", type=float, default=0.02, help="Sampling interval.")
    parser.add_argument("--coupling-strength", type=float, default=0.05, help="Diffusive coupling coefficient.")
    parser.add_argument("--external-current", type=float, default=3.2, help="Constant input current I.")
    parser.add_argument("--initial-noise", type=float, default=0.5, help="Noise scale for initial conditions.")
    parser.add_argument(
        "--custom-edges",
        type=Path,
        help="Optional JSON file containing an edge list [[i, j], ...] to override random sampling.",
    )
    parser.add_argument("--visualize", action="store_true", help="Display diagnostic plots after integration.")
    return parser.parse_args(argv)


def load_edges(path: Path) -> np.ndarray:
    edges = json.loads(path.read_text())
    edge_array = np.asarray(edges, dtype=int)
    if edge_array.size == 0:
        return np.empty((0, 2), dtype=int)
    if edge_array.ndim != 2 or edge_array.shape[1] != 2:
        raise ValueError("Edge list must consist of [i, j] pairs.")
    return edge_array


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    if args.custom_edges is not None:
        edge_list = load_edges(args.custom_edges)
        n_nodes = args.n_nodes if args.n_nodes is not None else (
            int(edge_list.max()) + 1 if edge_list.size else 0
        )
        adjacency = build_adjacency_matrix(edge_list, n_nodes)
        degrees = {node: int(adjacency[node].sum()) for node in range(n_nodes)}
    else:
        edge_list = None
        adjacency = None
        degrees = None

    result = simulate_hindmarsh_rose_network(
        n_nodes=args.n_nodes,
        edge_probability=args.edge_probability,
        seed=args.seed,
        tmax=args.tmax,
        dt=args.dt,
        coupling_strength=args.coupling_strength,
        I=args.external_current,
        initial_noise=args.initial_noise,
        edge_list=edge_list,
        adjacency=adjacency,
        degrees=degrees,
        visualize=args.visualize,
    )

    print(
        f"Simulation complete. {result.n_nodes} nodes, "
        f"{len(result.edge_list)} edges. Trajectories shape: {result.state.shape}."
    )

    if args.visualize:
        plot_overview(result)


if __name__ == "__main__":
    main()

