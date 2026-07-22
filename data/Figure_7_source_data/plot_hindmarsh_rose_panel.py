from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

import cmocean
import matplotlib

SCRIPT_DIR = Path(__file__).resolve().parent

FS_SMALL = 10
FS_MEDIUM = 12
FS_LARGE = 14


def main() -> None:
    with np.load(SCRIPT_DIR / "hindmarsh_rose_plot_data.npz") as data:
        t_plot = data["time"]
        state_block = data["truth"]
        pred_state = data["prediction"]
    n_nodes = state_block.shape[1]
    T_plot = float(t_plot[-1])
    heat_map_x = state_block.T
    heat_map_pred_x = pred_state.T
    x0_true = state_block[:, 0]
    x0_pred = pred_state[:, 0]
    vmin = min(heat_map_x.min(), heat_map_pred_x.min())
    vmax = max(heat_map_x.max(), heat_map_pred_x.max())

    params = {
        "image.origin": "lower",
        "image.interpolation": "nearest",
        "axes.grid": False,
        "axes.labelsize": FS_MEDIUM,
        "axes.titlesize": FS_LARGE,
        "font.size": FS_SMALL,
        "legend.fontsize": FS_SMALL,
        "xtick.labelsize": FS_SMALL,
        "ytick.labelsize": FS_SMALL,
        #"text.usetex": True,
        "font.family": "Helvetica",
        #"text.latex.preamble": r"\usepackage{amsmath}",
        #"axes.labelweight": "bold",
    }
    matplotlib.rcParams.update(params)

    fig = plt.figure(figsize=(3.2, 4.2), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.0, h_pad=0.0, hspace=0.0, wspace=0.0)
    gs = fig.add_gridspec(4, 1, height_ratios=[1.0, 0.12, 1.0, 0.12], hspace=0.0)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_top_ts = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax_bottom = fig.add_subplot(gs[2, 0], sharex=ax_top)
    ax_bottom_ts = fig.add_subplot(gs[3, 0], sharex=ax_top)

    extent = [0, T_plot, 0, n_nodes]
    yticks = [0, 25, 50, 75]

    im_top = ax_top.imshow(heat_map_x, aspect="auto", extent=extent, cmap="cmo.haline", vmin=vmin, vmax=vmax)
    ax_top.set_yticks(yticks)
    ax_top.set_ylim([0, 25])
    ax_top.set_ylabel(r"Index, $i$", labelpad=-6)
    ax_top.tick_params(axis="both", direction="in", labelbottom=False)
    for label in ax_top.get_xticklabels():
        label.set_fontname("Helvetica")
    for label in ax_top.get_yticklabels():
        label.set_fontname("Helvetica")
    cbar_top = plt.colorbar(im_top, ax=ax_top, pad=0.002, fraction=0.04)
    cbar_top.set_label(r"$x_i$", rotation=0, labelpad=5)
    cbar_top.set_ticks([4, 0, -4, -8])
    cbar_top.formatter = ScalarFormatter(useMathText=False)
    cbar_top.formatter.set_scientific(False)
    cbar_top.formatter.set_powerlimits((0, 0))
    cbar_top.update_normal(im_top)
    cbar_top.ax.yaxis.get_offset_text().set_x(2.5)
    for label in cbar_top.ax.get_yticklabels():
        label.set_fontname("Helvetica")

    ax_top_ts.plot(t_plot, x0_true, color="k", lw=0.6)
    ax_top_ts.set_ylabel(r"$x_0$", labelpad=-2)
    ax_top_ts.set_ylim([-2,3])
    ax_top_ts.set_yticks([-2,2])
    ax_top_ts.tick_params(axis="both", direction="in", labelbottom=False)
    ax_top_ts.yaxis.set_label_coords(-0.10, 0.5)
    for label in ax_top_ts.get_yticklabels():
        label.set_fontname("Helvetica")

    im_bottom = ax_bottom.imshow(
        heat_map_pred_x,
        aspect="auto",
        extent=extent,
        cmap="cmo.haline",
        vmin=vmin,
        vmax=vmax,
    )
    ax_bottom.set_yticks(yticks)
    ax_bottom.set_ylim([0, 25])
    ax_bottom.set_xticks(np.linspace(0, T_plot, num=3))
    ax_bottom.set_ylabel(r"Index, $i$", labelpad=-6)
    ax_bottom.tick_params(axis="both", direction="in", labelbottom=False)
    for label in ax_bottom.get_xticklabels():
        label.set_fontname("Helvetica")
    for label in ax_bottom.get_yticklabels():
        label.set_fontname("Helvetica")
    cbar_bottom = plt.colorbar(im_bottom, ax=ax_bottom, pad=0.002, fraction=0.04)
    cbar_bottom.set_label(r"$\hat{x}_i$", rotation=0, labelpad=5)
    cbar_bottom.set_ticks([4, 0, -4, -8])
    cbar_bottom.formatter = ScalarFormatter(useMathText=False)
    cbar_bottom.formatter.set_scientific(False)
    cbar_bottom.formatter.set_powerlimits((0, 0))
    cbar_bottom.update_normal(im_bottom)
    cbar_bottom.ax.yaxis.get_offset_text().set_x(2.5)
    for label in cbar_bottom.ax.get_yticklabels():
        label.set_fontname("Helvetica")

    ax_bottom_ts.plot(t_plot, x0_pred, color="black", lw=0.6)
    ax_bottom_ts.set_xlabel(r"Time, $t$")
    ax_bottom_ts.set_ylabel(r"$\hat{x}_0$", labelpad=-3)
    ax_bottom_ts.tick_params(axis="both", direction="in", pad=1)
    ax_bottom_ts.yaxis.set_label_coords(-0.10, 0.5)
    ax_bottom_ts.set_ylim([-2,3])
    ax_bottom_ts.set_yticks([-2,2])
    for label in ax_bottom_ts.get_xticklabels():
        label.set_fontname("Helvetica")
    for label in ax_bottom_ts.get_yticklabels():
        label.set_fontname("Arial")

    output_dir = SCRIPT_DIR / "reproduced"
    output_dir.mkdir(exist_ok=True)
    fig.savefig(output_dir / "HindmarshRose_75D_state_comparison.png", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()
