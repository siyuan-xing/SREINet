"""Extract only the numerical arrays plotted in manuscript Figure 6.

The original ``results/*/*.mat`` files are generated intermediates and are not
committed.  Run this script from a working checkout that contains those files
to rebuild the compact, Git-friendly ``*.npz`` plotting inputs.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def save(name: str, truth: np.ndarray, prediction: np.ndarray, time: np.ndarray,
         *, model: str, quantity: str, extent: tuple[float, float, float, float]) -> None:
    output = HERE / f"{name}_plot_data.npz"
    np.savez_compressed(
        output,
        truth=np.asarray(truth),
        prediction=np.asarray(prediction),
        time=np.asarray(time),
        model=np.asarray(model),
        quantity=np.asarray(quantity),
        extent=np.asarray(extent, dtype=float),
    )
    print(f"Wrote {output.relative_to(REPO)} ({output.stat().st_size / 1024**2:.1f} MiB)")


def lorenz96() -> None:
    source = REPO / "results/Lorenz_96_100D"
    train = loadmat(source / "train_data.mat")["data"]
    validation = loadmat(source / "validation_data.mat")["data"]
    dim = 100
    n = 10_000
    time = np.concatenate((train[:n, 0], validation[1:, 0]))[:-1]
    truth = np.concatenate(
        (train[:n, 1 + dim : 1 + 2 * dim], validation[1:, 1 + dim : 1 + 2 * dim])
    )[:-1].T
    prediction = np.concatenate(
        (train[:n, 1 + 2 * dim : 1 + 3 * dim], validation[1:, 1 + 2 * dim : 1 + 3 * dim])
    )[:-1].T
    save("lorenz96", truth, prediction, time, model="Lorenz-96", quantity="derivative", extent=(0, 20, 0, 100))


def phi4() -> None:
    data = loadmat(REPO / "results/Phi4/train_data.mat")["data"]
    dim, nodes, n = 100, 50, 10_000
    save(
        "phi4",
        data[:n, 1 + dim : 1 + dim + nodes].T,
        data[:n, 1 + 2 * dim : 1 + 2 * dim + nodes].T,
        data[:n, 0],
        model="discrete phi4",
        quantity="derivative",
        extent=(0, 500, -25, 25),
    )


def dnls() -> None:
    data = loadmat(REPO / "results/DNLS_100D/train_data.mat")["data"]
    dim, nodes, n = 100, 50, 10_000
    truth_derivative = data[:n, 1 + dim : 1 + 2 * dim]
    predicted_derivative = data[:n, 1 + 2 * dim : 1 + 3 * dim]
    truth = np.sqrt(truth_derivative[:, :nodes] ** 2 + truth_derivative[:, nodes : 2 * nodes] ** 2).T
    prediction = np.sqrt(
        predicted_derivative[:, :nodes] ** 2 + predicted_derivative[:, nodes : 2 * nodes] ** 2
    ).T
    save("dnls", truth, prediction, data[:n, 0], model="DNLS", quantity="derivative magnitude", extent=(0, 100, -25, 25))


def abl() -> None:
    data = loadmat(REPO / "results/AL_128D/train_data.mat")["data"]
    dim, nodes, n = 128, 64, 50_000
    truth_derivative = data[:n, 1 + dim : 1 + 2 * dim]
    predicted_derivative = data[:n, 1 + 2 * dim : 1 + 3 * dim]
    truth = np.sqrt(truth_derivative[:, :nodes] ** 2 + truth_derivative[:, nodes : 2 * nodes] ** 2).T
    prediction = np.sqrt(
        predicted_derivative[:, :nodes] ** 2 + predicted_derivative[:, nodes : 2 * nodes] ** 2
    ).T
    save("abl", truth, prediction, data[:n, 0], model="Ablowitz-Ladik", quantity="derivative magnitude", extent=(0, 500, -32, 32))


if __name__ == "__main__":
    lorenz96()
    phi4()
    dnls()
    abl()
