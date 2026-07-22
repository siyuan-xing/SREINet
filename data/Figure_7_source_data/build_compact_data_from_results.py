"""Extract the state trajectories and graph used in manuscript Figure 7.

The source ``*.mat`` files are generated intermediates and stay under the
Git-ignored ``results/`` tree.  This script solves the recovered systems once
and saves the much smaller arrays needed to redraw the published panels.
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.io import loadmat


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def load_function(path: Path, function_name: str):
    spec = spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, function_name)


def hindmarsh_rose() -> None:
    source = REPO / "results/Hindmarsh_rose_network-75D"
    mat = loadmat(source / "data.mat")
    data = mat["data"]
    time = data[:, 0]
    resets = np.where(np.diff(time) <= 0)[0]
    end = resets[0] + 1 if resets.size else len(time)
    time = time[:end]
    truth = data[:end, 1:76]
    ode = load_function(source / "hindmarsh_rose_ode.py", "hindmarsh_rose_ode")
    solution = solve_ivp(ode, (time[0], time[-1]), truth[0], t_eval=time, method="BDF")
    if not solution.success or solution.y.shape[1] != time.size:
        raise RuntimeError(f"Hindmarsh-Rose integration failed: {solution.message}")
    output = HERE / "hindmarsh_rose_plot_data.npz"
    np.savez_compressed(
        output,
        time=time - time[0],
        truth=truth,
        prediction=solution.y.T,
        edge_list=np.asarray(mat["edge_list"], dtype=int),
        n_nodes=np.asarray(75),
        network_nodes=np.asarray(25),
        model=np.asarray("Hindmarsh-Rose network"),
    )
    print(f"Wrote {output.relative_to(REPO)} ({output.stat().st_size / 1024**2:.1f} MiB)")


def kuramoto() -> None:
    source = REPO / "results/Kuramoto_60D"
    data = loadmat(source / "train_data.mat")["data"]
    dim, n, end_time, dt = 60, 1_000, 100.0, 0.1
    time = data[:n, 0]
    truth = data[:n, 1 : 1 + dim]
    ode = load_function(source / "kuramoto_ode.py", "kuramoto_ode")
    t_eval = np.arange(0.0, end_time, dt)
    solution = solve_ivp(ode, (0.0, end_time), truth[0], t_eval=t_eval, method="BDF")
    if not solution.success or solution.y.shape[1] != time.size:
        raise RuntimeError(f"Kuramoto integration failed: {solution.message}")
    output = HERE / "kuramoto_plot_data.npz"
    np.savez_compressed(
        output,
        time=time,
        truth=truth,
        prediction=solution.y.T,
        n_nodes=np.asarray(dim),
        displayed_network_nodes=np.asarray(15),
        model=np.asarray("Kuramoto network"),
    )
    print(f"Wrote {output.relative_to(REPO)} ({output.stat().st_size / 1024**2:.1f} MiB)")


if __name__ == "__main__":
    hindmarsh_rose()
    kuramoto()
