import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from scipy.integrate import solve_ivp
from matplotlib.ticker import ScalarFormatter

import cmocean
import matplotlib

from hindmarsh_rose_ode import hindmarsh_rose_ode

FS_SMALL = 10
FS_MEDIUM = 12
FS_LARGE = 14


def main() -> None:
    train_data = scipy.io.loadmat("data.mat")["data"]

    t_arr = train_data[:, 0]
    state = train_data[:, 1:]
    dim = state.shape[1] // 2 # original data including x and dx
    n_nodes = dim  # 75D system; plot all dimensions

    # Split into the first continuous trajectory (t strictly increasing)
    resets = np.where(np.diff(t_arr) <= 0)[0]
    first_block_end = resets[0] + 1 if resets.size else len(t_arr)
    t_block = t_arr[:first_block_end]
    state_block = state[:first_block_end, :n_nodes]

    t_relative = t_block - t_block[0]
    T_plot = min(100.0, t_relative[-1])

    solution = solve_ivp(
        hindmarsh_rose_ode,
        (t_block[0], t_block[-1]),
        state_block[0],
        t_eval=t_block,
        method="BDF",
    )
    
    pred_state = solution.y.T

    plot_mask = t_relative <= T_plot + 1e-9
    t_plot = t_relative[plot_mask]
    heat_map_x = state_block[plot_mask, :n_nodes].T
    heat_map_pred_x = pred_state[plot_mask, :n_nodes].T
    x0_true = state_block[plot_mask, 0]
    x0_pred = pred_state[plot_mask, 0]
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

    fig.savefig("HindmarshRose_75D_state_comparison.png", dpi=600)


if __name__ == "__main__":
    main()
