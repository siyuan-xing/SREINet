from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter, FixedLocator

import cmocean
import matplotlib

SCRIPT_DIR = Path(__file__).resolve().parent

FS_SMALL = 10
FS_MEDIUM = 12
FS_LARGE = 14


def main() -> None:
    with np.load(SCRIPT_DIR / "kuramoto_plot_data.npz") as data:
        t_plot = data["time"]
        x_arr = data["truth"]
        pred_x_arr = data["prediction"]
    dim = x_arr.shape[1]

    # Only plot t=0 to T_plot
    node_idx = 30
    heat_map_x = x_arr.T
    heat_map_pred_x = pred_x_arr.T
    theta_node_true = x_arr[:, node_idx]
    theta_node_pred = pred_x_arr[:, node_idx]
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

    t_plot_end = t_plot[-1]
    extent = [0, t_plot_end, 0, dim]
    yticks = [0, 60]

    im_top = ax_top.imshow(heat_map_x, aspect="auto", extent=extent, cmap="cmo.deep", vmin=vmin, vmax=vmax)
    ax_top.set_yticks(yticks)
    ax_top.set_ylabel(r"Index, $i$", labelpad=-6)
    ax_top.tick_params(axis="both", direction="in", labelbottom=False)
    cbar_top = plt.colorbar(im_top, ax=ax_top, pad=0.002, fraction=0.04)
    cbar_top.set_label(r"$\theta_i$", rotation=0, labelpad=5)
    # Set scientific notation for colorbar
    cbar_top.formatter = ScalarFormatter(useMathText=True)
    cbar_top.formatter.set_powerlimits((0, 0))
    cbar_top.update_normal(im_top)
    cbar_top.ax.yaxis.get_offset_text().set_x(2.5)

    ax_top_ts.plot(t_plot, theta_node_true, color="k", lw=0.6)
    ax_top_ts.set_ylabel(r"$\theta_{30}$", labelpad=-3)
    ax_top_ts.tick_params(axis="both", direction="in", labelbottom=False)
    ax_top_ts.yaxis.set_label_coords(-0.10, 0.5)
   

    im_bottom = ax_bottom.imshow(
        heat_map_pred_x,
        aspect="auto",
        extent=extent,
        cmap="cmo.deep",
        vmin=vmin,
        vmax=vmax,
    )
    ax_bottom.set_xticks([0, 50, 100])
    ax_bottom.set_ylabel(r"Index, $i$", labelpad=-6)
    ax_bottom.set_yticks([2])
    ax_bottom.yaxis.set_major_locator(FixedLocator([2]))
    ax_bottom.tick_params(axis="both", direction="in", labelbottom=False)
    cbar_bottom = plt.colorbar(im_bottom, ax=ax_bottom, pad=0.002, fraction=0.04)
    cbar_bottom.set_label(r"$\hat{\theta}_i$", rotation=0, labelpad=5)
    # Set scientific notation for colorbar
    cbar_bottom.formatter = ScalarFormatter(useMathText=True)
    cbar_bottom.formatter.set_powerlimits((0, 0))
    cbar_bottom.update_normal(im_bottom)
    cbar_bottom.ax.yaxis.get_offset_text().set_x(2.5)

    ax_bottom_ts.plot(t_plot, theta_node_pred, color="k", lw=0.6)
    ax_bottom_ts.set_xlabel(r"Time, $t$")
    ax_bottom_ts.set_ylabel(r"$\hat{\theta}_{30}$", labelpad=-3)
    ax_bottom_ts.tick_params(axis="both", direction="in", pad=1)
    ax_bottom_ts.yaxis.set_label_coords(-0.10, 0.5)

    output_dir = SCRIPT_DIR / "reproduced"
    output_dir.mkdir(exist_ok=True)
    fig.savefig(output_dir / "Kuramoto_60D_state_comparison.png", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()
