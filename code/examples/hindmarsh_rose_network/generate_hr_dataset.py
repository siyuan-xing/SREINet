#!/usr/bin/env python3
"""Generate Hindmarsh–Rose network data identical to the notebook setup."""

from __future__ import annotations

import os
import sys
from typing import Iterable

import numpy as np
from scipy.io import savemat

sys.path.append(os.path.abspath("../../utilities"))

import DataGenerator as DG  # noqa: E402
from hindmarsh_rose_model import hindmarsh_rose_network  # noqa: E402
from network_generator import generate_erdos_renyi_network  # noqa: E402


def build_initial_conditions(
    num_trajectories: int,
    n_nodes: int,
    seed: int,
) -> list[np.ndarray]:
    """Match the notebook's initial condition sampling."""
    np.random.seed(seed)
    conditions: list[np.ndarray] = []
    for _ in range(num_trajectories):
        x0 = np.random.uniform(-1.0, 1.0, size=n_nodes)
        y0 = np.random.uniform(-5.0, 5.0, size=n_nodes)
        z0 = np.random.uniform(1.0, 2.0, size=n_nodes)
        conditions.append(np.concatenate([x0, y0, z0]))
    return conditions


def make_time_column(t_arr: Iterable[float], repeats: int) -> np.ndarray:
    """Tile the 1D time array once for each trajectory."""
    tiled = np.tile(np.asarray(t_arr), repeats)
    return tiled.reshape(-1, 1)


def main() -> None:
    seed = 42

    # Network & simulation parameters (kept in sync with the notebook)
    n_nodes = 25
    edge_probability = 0.2
    num_traj = 15
    data_T = 30.0
    data_dt = 0.01

    output_dir = "output"
    train_path = os.path.join(output_dir, "train_data.mat")
    val_path = os.path.join(output_dir, "validate_data.mat")

    os.makedirs(output_dir, exist_ok=True)

    edge_list, adjacency, degrees = generate_erdos_renyi_network(
        n_nodes,
        edge_probability,
        seed=seed,
    )

    initial_conditions = build_initial_conditions(num_traj, n_nodes, seed)
    generator = DG.DataGenerator(
        initial_conditions,
        T=data_T,
        dt=data_dt,
        derivative_mode="exact",
    )

    def hr_vector_field(t: float, state: np.ndarray, edges: np.ndarray) -> np.ndarray:
        return hindmarsh_rose_network(
            t,
            state,
            edges,
        )

    t_arr, x_data, dx_data = generator.generate_dataset_by_custom_ODEs(
        hr_vector_field,
        edge_list,
        method="RK45",
    )

    time_column = make_time_column(t_arr, len(initial_conditions))

    # Stack time, state, and derivatives to match notebook loading:
    # scipy.io.loadmat(... )["data"]
    data_matrix = np.concatenate(
        [time_column, x_data, dx_data],
        axis=1,
    )

    payload = {
        "data": data_matrix,
        "t": time_column,
        "x": x_data,
        "dx": dx_data,
        "edge_list": edge_list,
        "adjacency": adjacency,
        "degrees": np.asarray(sorted(degrees.items()), dtype=float),
        "initial_conditions": np.asarray(initial_conditions),
        "dt": np.asarray(data_dt),
        "T": np.asarray(data_T),
    }

    data_path = os.path.join(output_dir, "data.mat")
    savemat(data_path, payload)

    print(
        f"Saved full dataset to {data_path} ({data_matrix.shape[0]} samples, "
        f"matrix shape {data_matrix.shape})"
    )


if __name__ == "__main__":
    main()
