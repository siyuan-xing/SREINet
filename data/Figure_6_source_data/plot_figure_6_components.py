"""Reproduce the five component panels used to assemble manuscript Figure 6."""

from pathlib import Path
import subprocess
import sys

import cmocean  # noqa: F401  # registers ``cmo.delta``
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "reproduced"
FS_SMALL = 10
FS_MEDIUM = 12

matplotlib.rcParams.update(
    {
        "image.origin": "lower",
        "image.interpolation": "nearest",
        "axes.grid": False,
        "savefig.dpi": 600,
        "axes.labelsize": FS_MEDIUM,
        "axes.titlesize": FS_MEDIUM,
        "axes.titleweight": "bold",
        "font.size": FS_SMALL,
        "legend.fontsize": FS_SMALL,
        "xtick.labelsize": FS_SMALL,
        "ytick.labelsize": FS_SMALL,
        "text.usetex": False,
        "font.family": "Helvetica",
    }
)


CONFIGS = {
    "lorenz96": {
        "filename": "Lorenz_96_ground_truth_vs_identified.png",
        "true_title": r"$\dot{x}_i$",
        "pred_title": r"$\dot{\hat{x}}_i$",
        "yticks": [0, 100],
        "xticks": [0, 10, 20],
        "ylabel": r"Index, $i$",
        "left": 0.23,
        "fixed_error_exponent": -5,
    },
    "phi4": {
        "filename": "phi4_50_node_results.png",
        "true_title": r"$\dot{u}_i$",
        "pred_title": r"$\dot{\hat{u}}_i$",
        "yticks": [-25, 25],
        "xticks": [0, 250, 500],
        "left": 0.18,
    },
    "dnls": {
        "filename": "DNLS_50_node_results.png",
        "true_title": r"$|\dot{u}|_i$",
        "pred_title": r"$|\dot{\hat{u}}|_i$",
        "yticks": [-25, 25],
        "xticks": [0, 50, 100],
        "left": 0.18,
        # Retain the published script's top-panel x extent.
        "true_extent": (0, 80, -25, 25),
    },
    "abl": {
        "filename": "AL_64_node_results.png",
        "true_title": r"$|\dot{u}|_i$",
        "pred_title": r"$|\dot{\hat{u}}|_i$",
        "yticks": [-32, 32],
        "xticks": [0, 250, 500],
        "left": 0.18,
    },
}


def plot_heatmaps(name: str) -> None:
    config = CONFIGS[name]
    with np.load(HERE / f"{name}_plot_data.npz") as data:
        truth = data["truth"]
        prediction = data["prediction"]
        extent = tuple(data["extent"])

    fig = plt.figure(figsize=(7.0 / 4.0, 4.08))
    fig.patch.set_facecolor((0.95, 0.95, 0.95, 0.1))
    grid = fig.add_gridspec(3, 2, hspace=0.28, width_ratios=[20, 1])

    ax_true = fig.add_subplot(grid[0, 0])
    true_extent = config.get("true_extent", extent)
    true_kwargs = {} if name == "lorenz96" else {"extent": true_extent}
    ax_true.imshow(truth, aspect="auto", cmap="cmo.delta", **true_kwargs)
    ax_true.set_xticks([])
    ax_true.set_yticks(config["yticks"])
    ax_true.tick_params(axis="both", direction="in")
    if config.get("ylabel"):
        ax_true.set_ylabel(config["ylabel"], labelpad=-10)
    true_label = fig.add_subplot(grid[0, 1])
    true_label.set_title(config["true_title"], loc="center", fontsize=FS_SMALL, y=0.4)
    true_label.axis("off")

    ax_pred = fig.add_subplot(grid[1, 0])
    pred_kwargs = {} if name == "lorenz96" else {"extent": extent}
    ax_pred.imshow(prediction, aspect="auto", cmap="cmo.delta", **pred_kwargs)
    ax_pred.set_xticks([])
    ax_pred.set_yticks(config["yticks"])
    ax_pred.tick_params(axis="both", direction="in")
    if config.get("ylabel"):
        ax_pred.set_ylabel(config["ylabel"], labelpad=-10)
    pred_label = fig.add_subplot(grid[1, 1])
    pred_label.set_title(config["pred_title"], loc="center", fontsize=FS_SMALL, y=0.4)
    pred_label.axis("off")

    error = np.abs(truth - prediction)
    ax_error = fig.add_subplot(grid[2, 0])
    image = ax_error.imshow(error, extent=extent, aspect="auto", cmap="cmo.delta")
    colorbar_axis = fig.add_subplot(grid[2, 1])
    colorbar = fig.colorbar(image, cax=colorbar_axis, orientation="vertical")
    exponent = config.get("fixed_error_exponent")
    if exponent is None:
        maximum = np.nanmax(error)
        exponent = int(np.floor(np.log10(maximum))) if maximum > 0 else 0
        colorbar.locator = MaxNLocator(nbins=5, steps=[1, 2, 5, 10])
    scale = 10.0 ** exponent
    colorbar.formatter = FuncFormatter(lambda value, _: f"{value / scale:g}")
    colorbar.update_ticks()
    colorbar.ax.set_title(rf"$\times 10^{{{exponent}}}$", fontsize=FS_SMALL, pad=2)
    colorbar.ax.tick_params(direction="in", labelleft=False, labelright=True)
    ax_error.set_yticks(config["yticks"])
    ax_error.set_xticks(config["xticks"])
    ax_error.set_xlabel(r"Time, $t$", labelpad=0)
    ax_error.tick_params(axis="both", direction="in")
    if config.get("ylabel"):
        ax_error.set_ylabel(config["ylabel"], labelpad=-10)

    plt.subplots_adjust(left=config["left"], right=0.78)
    output = OUTPUT / config["filename"]
    fig.savefig(output, dpi=600)
    plt.close(fig)
    print(f"Wrote {output.relative_to(HERE)}")


if __name__ == "__main__":
    OUTPUT.mkdir(exist_ok=True)
    for system in CONFIGS:
        plot_heatmaps(system)
    subprocess.run(
        [sys.executable, str(HERE / "original_results_scripts/identification_error_plot.py")],
        cwd=OUTPUT,
        check=True,
    )
    print("Wrote reproduced/combined_identification_error_plot.png")
