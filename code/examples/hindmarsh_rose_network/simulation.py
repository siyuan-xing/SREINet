"""
Simulation helpers for Hindmarsh–Rose networks with pairwise coupling.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from .hindmarsh_rose_model import hindmarsh_rose_network
from .network_generator import generate_erdos_renyi_network, build_adjacency_matrix


@dataclass
class SimulationResult:
    time: np.ndarray
    state: np.ndarray
    derivatives: np.ndarray
    edge_list: np.ndarray
    adjacency: np.ndarray
    degrees: Dict[int, int]
    n_nodes: int

    def to_dict(self) -> Dict[str, np.ndarray]:
        return {
            "T": self.time,
            "X": self.state,
            "dX": self.derivatives,
            "edge_list": self.edge_list,
            "adjacency": self.adjacency,
            "n_nodes": self.n_nodes,
        }


def simulate_hindmarsh_rose_network(
    n_nodes: Optional[int] = 20,
    edge_probability: float = 0.25,
    seed: Optional[int] = 101,
    tmax: float = 80.0,
    dt: float = 0.02,
    coupling_strength: float = 5e-2,
    I: float = 3.2,
    initial_noise: float = 0.5,
    visualize: bool = False,
    edge_list: Optional[np.ndarray] = None,
    adjacency: Optional[np.ndarray] = None,
    degrees: Optional[Dict[int, int]] = None,
) -> SimulationResult:
    """
    Run a Hindmarsh–Rose network simulation with diffusive coupling.
    """

    if edge_list is None:
        if n_nodes is None:
            raise ValueError("`n_nodes` must be provided when sampling a network.")
        edge_list, adjacency, degrees = generate_erdos_renyi_network(
            n_nodes=n_nodes,
            edge_probability=edge_probability,
            seed=seed,
        )
    else:
        edge_list = np.asarray(edge_list, dtype=int)
        if edge_list.ndim == 1:
            edge_list = edge_list.reshape(-1, 2)
        if adjacency is None:
            if n_nodes is None:
                n_nodes = int(edge_list.max()) + 1 if edge_list.size else 0
            adjacency = build_adjacency_matrix(edge_list, n_nodes)
        if degrees is None:
            degrees = {node: int(adjacency[node].sum()) for node in range(adjacency.shape[0])}

    if adjacency is None:
        raise ValueError("Adjacency could not be determined.")

    n_nodes = adjacency.shape[0]

    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-1.0, 1.0, size=n_nodes)
    y0 = rng.uniform(-10.0, 0.0, size=n_nodes)
    z0 = rng.uniform(1.0, 2.0, size=n_nodes)
    initial_state = np.concatenate([x0, y0, z0]) + rng.normal(scale=initial_noise, size=3 * n_nodes)

    time_grid = np.arange(0.0, tmax + dt, dt)
    start = time.perf_counter()
    solution = solve_ivp(
        lambda t, state: hindmarsh_rose_network(
            t,
            state,
            edge_list,
            coupling_strength=coupling_strength,
            I=I,
        ),
        t_span=(0.0, tmax),
        y0=initial_state,
        t_eval=time_grid,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
    )
    if not solution.success:
        raise RuntimeError(f"Integration failed: {solution.message}")

    elapsed = time.perf_counter() - start
    print(f"Integration finished in {elapsed:.2f} seconds ({len(solution.t)} steps).")

    trajectories = solution.y.T
    derivatives = np.zeros_like(trajectories)
    for idx, t in enumerate(solution.t):
        derivatives[idx] = hindmarsh_rose_network(
            t,
            trajectories[idx],
            edge_list,
            coupling_strength=coupling_strength,
            I=I,
        )

    result = SimulationResult(
        time=solution.t,
        state=trajectories,
        derivatives=derivatives,
        edge_list=edge_list,
        adjacency=adjacency,
        degrees=degrees,
        n_nodes=n_nodes,
    )

    if visualize:
        plot_overview(result)

    return result


def plot_overview(result: SimulationResult) -> None:
    time_values = result.time
    state = result.state
    derivatives = result.derivatives
    n_nodes = result.n_nodes

    x = state[:, :n_nodes]
    y = state[:, n_nodes:2 * n_nodes]
    z = state[:, 2 * n_nodes:]

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    for idx in range(min(4, n_nodes)):
        axes[0].plot(time_values, x[:, idx], label=f"x_{idx}")
    axes[0].set_ylabel("x")
    axes[0].set_title("Membrane potentials")
    axes[0].legend()

    for idx in range(min(3, n_nodes)):
        axes[1].plot(time_values, y[:, idx], label=f"y_{idx}")
    axes[1].set_ylabel("y")
    axes[1].set_title("Fast recovery variables")
    axes[1].legend()

    axes[2].plot(time_values, z[:, 0], label="z_0")
    axes[2].plot(time_values, derivatives[:, 0], label="dx/dt_0", alpha=0.7)
    axes[2].set_xlabel("time")
    axes[2].set_ylabel("value")
    axes[2].set_title("Slow variable and derivative (node 0)")
    axes[2].legend()

    plt.tight_layout()
    plt.show()

