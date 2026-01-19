import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from scipy.integrate import solve_ivp

import cmocean

from hindmarsh_rose_ode import hindmarsh_rose_ode


def main() -> None:
    mat = scipy.io.loadmat("data.mat")
    data = mat["data"]

    t_arr = data[:, 0]
    state = data[:, 1:]
    dim = state.shape[1] // 2
    x_arr = state[:, :dim]
    dx_arr = state[:, dim:]

    # Use the first continuous trajectory block.
    resets = np.where(np.diff(t_arr) <= 0)[0]
    first_block_end = resets[0] + 1 if resets.size else len(t_arr)
    t_block = t_arr[:first_block_end]
    x_block = x_arr[:first_block_end]
    dx_block = dx_arr[:first_block_end]

    initial_state = x_block[0]
    t_span = (t_block[0], t_block[-1])
    t_eval = t_block

    solution = solve_ivp(
        hindmarsh_rose_ode,
        t_span,
        initial_state,
        t_eval=t_eval,
        method="BDF",
    )
    pred_x_arr = solution.y.T

    t_relative = t_block - t_block[0]
    t_plot = min(50.0, t_relative[-1])
    plot_mask = t_relative <= t_plot + 1e-9

    heatmap_t_arr = t_relative[plot_mask]
    heat_map_dx = dx_block[plot_mask].T
    heat_map_x = x_block[plot_mask].T
    heat_map_pred_x = pred_x_arr[plot_mask].T

    params = {
        "image.origin": "lower",
        "image.interpolation": "nearest",
        "axes.grid": False,
        "savefig.dpi": 600,
        "axes.labelsize": 14,
        "axes.titlesize": 12,
        "font.size": 14,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "text.usetex": True,
        "font.family": "Helvetica",
        "text.latex.preamble": r"\usepackage{amsmath}",
        "axes.labelweight": "bold",
    }
    matplotlib.rcParams.update(params)

    my_size = 110 / 25.4
    fig = plt.figure(figsize=(my_size * 2, my_size))
    gs = fig.add_gridspec(10, 3, width_ratios=[20, 20, 1], hspace=0.1, wspace=0.2)

    ax1 = fig.add_subplot(gs[:4, 0])
    ax1.set_position(
        [
            ax1.get_position().x0,
            ax1.get_position().y0 + 0.02,
            ax1.get_position().width,
            ax1.get_position().height,
        ]
    )
    cax1 = ax1.imshow(heat_map_dx, aspect="auto", cmap="cmo.tempo")
    ax1.set_xticks([])
    ax1.set_yticks([0, 25, 50, 75])
    ax1.set_ylabel(r"State Index, $i$")
    ax1.set_title("Full Simulation")

    ax2 = fig.add_subplot(gs[4, 0])
    ax2.set_position(
        [
            ax2.get_position().x0,
            ax2.get_position().y0 + 0.02,
            ax2.get_position().width,
            ax2.get_position().height,
        ]
    )
    ax2.plot(heatmap_t_arr, heat_map_dx[0], "k", linewidth=2.0)
    ax2.set_xlim(0, t_plot)
    ax2.set_ylabel(r"$\dot{x}_1$")
    ax2.set_xticks([])

    ax3 = fig.add_subplot(gs[:4, 1])
    ax3.set_position(
        [
            ax3.get_position().x0,
            ax3.get_position().y0 + 0.02,
            ax3.get_position().width,
            ax3.get_position().height,
        ]
    )
    cax3 = ax3.imshow(heat_map_pred_x, aspect="auto", cmap="cmo.tempo")
    ax3.set_xticks([])
    ax3.set_yticks([0, 25, 50, 75])
    ax3.set_title("Identified System")

    ax4 = fig.add_subplot(gs[4, 1])
    ax4.set_position(
        [
            ax4.get_position().x0,
            ax4.get_position().y0 + 0.02,
            ax4.get_position().width,
            ax4.get_position().height,
        ]
    )
    ax4.plot(heatmap_t_arr, heat_map_pred_x[0], "k", linewidth=2.0)
    ax4.set_xlim(0, t_plot)
    ax4.set_xticks([])

    cbar1_ax = fig.add_subplot(gs[:4, 2])
    cbar1_ax.set_position(
        [
            cbar1_ax.get_position().x0,
            cbar1_ax.get_position().y0 + 0.02,
            cbar1_ax.get_position().width,
            cbar1_ax.get_position().height,
        ]
    )
    cbar = fig.colorbar(cax1, ax=ax1, orientation="vertical", cax=cbar1_ax)
    cbar.set_label(r"Derivative, $\dot{x}_i$")

    ax5 = fig.add_subplot(gs[5:-1, 0])
    ax5.set_position(
        [
            ax5.get_position().x0,
            ax5.get_position().y0 - 0.02,
            ax5.get_position().width,
            ax5.get_position().height,
        ]
    )
    cax5 = ax5.imshow(heat_map_x, aspect="auto", cmap="cmo.tempo")
    ax5.set_xticks([])
    ax5.set_yticks([0, 25, 50, 75])
    ax5.set_ylabel(r"State Index, $i$")

    ax6 = fig.add_subplot(gs[-1, 0])
    ax6.set_position(
        [
            ax6.get_position().x0,
            ax6.get_position().y0 - 0.02,
            ax6.get_position().width,
            ax6.get_position().height,
        ]
    )
    ax6.plot(heatmap_t_arr, heat_map_x[0], "k", linewidth=2.0)
    ax6.set_xlim(0, t_plot)
    ax6.set_xlabel(r"Time, $t$", labelpad=-0.5)
    ax6.set_ylabel(r"$x_1$")
    ax6.set_xticks([0, t_plot / 2, t_plot])

    ax7 = fig.add_subplot(gs[5:-1, 1])
    ax7.set_position(
        [
            ax7.get_position().x0,
            ax7.get_position().y0 - 0.02,
            ax7.get_position().width,
            ax7.get_position().height,
        ]
    )
    cax7 = ax7.imshow(heat_map_pred_x, aspect="auto", cmap="cmo.tempo")
    ax7.set_xticks([])
    ax7.set_yticks([0, 25, 50, 75])

    ax8 = fig.add_subplot(gs[-1, 1])
    ax8.set_position(
        [
            ax8.get_position().x0,
            ax8.get_position().y0 - 0.02,
            ax8.get_position().width,
            ax8.get_position().height,
        ]
    )
    ax8.plot(heatmap_t_arr, heat_map_pred_x[0], "k", linewidth=2.0)
    ax8.set_xlim(0, t_plot)
    ax8.set_xlabel(r"Time, $t$", labelpad=-0.5)
    ax8.set_xticks([0, t_plot / 2, t_plot])

    cbar2_ax = fig.add_subplot(gs[5:-1, 2])
    cbar2_ax.set_position(
        [
            cbar2_ax.get_position().x0,
            cbar2_ax.get_position().y0 - 0.02,
            cbar2_ax.get_position().width,
            cbar2_ax.get_position().height,
        ]
    )
    cbar = fig.colorbar(cax5, ax=ax5, orientation="vertical", cax=cbar2_ax)
    cbar.set_label(r"State, $x_i$")

    plt.tight_layout()
    plt.savefig("HindmarshRose_75D_simulation_vs_infer", dpi=600)

    error_map = np.abs(heat_map_x - heat_map_pred_x)
    error_vmax = error_map.max()

    fig2, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True, constrained_layout=True)
    slices = [(0, 25, "x"), (25, 50, "y"), (50, 75, "z")]

    for ax, (start, end, label) in zip(axes, slices):
        im = ax.imshow(
            error_map[start:end, :],
            extent=[0, t_plot, 0, end - start],
            aspect="auto",
            cmap="cmo.tempo",
            vmin=0.0,
            vmax=error_vmax,
        )
        ax.set_ylabel(rf"${label}$ index")
        ax.set_yticks([0, 12.5, 25])

    axes[-1].set_xlabel(r"Time, $t$")
    axes[-1].set_xticks([0, t_plot / 2, t_plot])
    cbar_2 = fig2.colorbar(im, ax=axes, location="right", shrink=0.95)
    cbar_2.set_label(r"Error")
    fig2.savefig("HindmarshRose_75D_error", dpi=600)

    fig3 = plt.figure(figsize=(8, 6))
    gs = fig3.add_gridspec(4, 2)

    full_point = len(heatmap_t_arr)
    plot_idx = [4, 9, 14, 19]

    ax11 = fig3.add_subplot(gs[0:2, 0])
    ax11.plot(heatmap_t_arr[:full_point], heat_map_x[plot_idx[0], :full_point], "k", label="True")
    ax11.plot(
        heatmap_t_arr[:full_point],
        heat_map_pred_x[plot_idx[0], :full_point],
        "r--",
        label="Predicted",
    )
    ax11.set_ylabel(r"$x_{5}$")
    ax11.set_yticks([-1, 0, 1, 2])

    ax12 = fig3.add_subplot(gs[0:2, 1])
    ax12.plot(heatmap_t_arr[:full_point], heat_map_x[plot_idx[1], :full_point], "k", label="True")
    ax12.plot(
        heatmap_t_arr[:full_point],
        heat_map_pred_x[plot_idx[1], :full_point],
        "r--",
        label="Predicted",
    )
    ax12.set_ylabel(r"$x_{10}$")
    ax12.set_yticks([-1, 0, 1, 2])

    ax13 = fig3.add_subplot(gs[2:4, 0])
    ax13.plot(heatmap_t_arr[:full_point], heat_map_x[plot_idx[2], :full_point], "k", label="True")
    ax13.plot(
        heatmap_t_arr[:full_point],
        heat_map_pred_x[plot_idx[2], :full_point],
        "r--",
        label="Predicted",
    )
    ax13.set_xlabel(r"Time, $t$")
    ax13.set_ylabel(r"$x_{15}$")
    ax13.set_yticks([-1, 0, 1, 2])

    ax14 = fig3.add_subplot(gs[2:4, 1])
    ax14.plot(heatmap_t_arr[:full_point], heat_map_x[plot_idx[3], :full_point], "k", label="True")
    ax14.plot(
        heatmap_t_arr[:full_point],
        heat_map_pred_x[plot_idx[3], :full_point],
        "r--",
        label="Predicted",
    )
    ax14.set_xlabel(r"Time, $t$")
    ax14.set_ylabel(r"$x_{20}$")
    ax14.set_yticks([-1, 0, 1, 2])

    plt.tight_layout()
    plt.savefig("HindmarshRose_75D_time_history", dpi=600)


if __name__ == "__main__":
    main()
