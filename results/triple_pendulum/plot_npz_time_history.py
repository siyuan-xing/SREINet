import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Styling consistent with simulate_and_compare.py
plt.rcParams.update(
    {
        "axes.grid": False,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "font.size": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "font.family": "Helvetica",
        "figure.facecolor": "none",
        "axes.facecolor": "white",
        "savefig.facecolor": "none",
    }
)
palette = sns.color_palette("Set2")
true_color = palette[0]
pred_color = palette[1]


def main() -> None:
    data = np.load("output/triple_pendulum_3layers_data.npz")

    t = data["SREINet_t"].ravel()
    y_pred = data["SREINet_y"]
    val_t = data["val_t_eval"].ravel()
    val_x = data["val_x"]
    pred_dx = data["pred_dx"]
    val_dd_theta = data["val_dd_theta"]

    y_pred = np.transpose(y_pred)

    # States: angles/velocities are first six entries
    theta_true = val_x[:, 0:3]
    dtheta_true = val_x[:, 3:6]
    theta_pred = y_pred[:, 0:3]
    dtheta_pred = y_pred[:, 3:6]

    # Accelerations: true from val_dd_theta (rows 0-2), predicted from pred_dx rows 3-5
    ddtheta_true = val_dd_theta[:, 0:3]
    ddtheta_pred = pred_dx[3:6, :]

    labels = [
        r"$\theta_1$",
        r"$\theta_2$",
        r"$\theta_3$",
        r"$\dot{\theta}_1$",
        r"$\dot{\theta}_2$",
        r"$\dot{\theta}_3$",
        r"$\ddot{\theta}_1$",
        r"$\ddot{\theta}_2$",
        r"$\ddot{\theta}_3$",
    ]

    fig, axes = plt.subplots(3, 3, figsize=(8, 4.5), sharex=False, facecolor="none")

    # Helper to configure one subplot
    def setup_panel(ax: plt.Axes, truth: np.ndarray, pred: np.ndarray, label: str, is_angle: bool) -> None:
        ax.set_facecolor("white")
        ax.plot(val_t, truth[: len(val_t)], label="True", color=true_color, lw=1.0)
        ax.plot(t, pred, label="Pred", color=pred_color, lw=1.0, ls="--")
        ax.set_ylabel(label)
        ax.tick_params(direction="in")
        # Leave limits/ticks to be set manually if desired

    # Row 1: angles
    setup_panel(axes[0, 0], theta_true[:, 0], theta_pred[:, 0], labels[0], is_angle=True)
    setup_panel(axes[0, 1], theta_true[:, 1], theta_pred[:, 1], labels[1], is_angle=True)
    setup_panel(axes[0, 2], theta_true[:, 2], theta_pred[:, 2], labels[2], is_angle=True)

    # Row 2: angular velocities
    setup_panel(axes[1, 0], dtheta_true[:, 0], dtheta_pred[:, 0], labels[3], is_angle=False)
    setup_panel(axes[1, 1], dtheta_true[:, 1], dtheta_pred[:, 1], labels[4], is_angle=False)
    setup_panel(axes[1, 2], dtheta_true[:, 2], dtheta_pred[:, 2], labels[5], is_angle=False)

    # Row 3: angular accelerations
    setup_panel(axes[2, 0], ddtheta_true[:, 0], ddtheta_pred[0, :], labels[6], is_angle=False)
    setup_panel(axes[2, 1], ddtheta_true[:, 1], ddtheta_pred[1, :], labels[7], is_angle=False)
    setup_panel(axes[2, 2], ddtheta_true[:, 2], ddtheta_pred[2, :], labels[8], is_angle=False)

    # Sync x-axis limits across columns (since we removed sharex)
    x_min = min(val_t.min(), t.min())
    x_max = max(val_t.max(), t.max())
    for col in range(3):
        for row in range(3):
            axes[row, col].set_xlim(x_min, x_max)

    # Hide x tick labels on upper rows
    for row in range(2):
        for col in range(3):
            axes[row, col].set_xticklabels([])

    # X labels only on bottom row
    axes[2, 0].set_xlabel("Time [s]")
    axes[2, 0].set_xticks([0, 5, 10])
    axes[2, 0].set_xticklabels(["0", "5", "10"])
    axes[2, 1].set_xlabel("Time [s]")
    axes[2, 1].set_xticks([0, 5, 10])
    axes[2, 1].set_xticklabels(["0", "5", "10"])
    axes[2, 2].set_xlabel("Time [s]")
    axes[2, 2].set_xticks([0, 5, 10])
    axes[2, 2].set_xticklabels(["0", "5", "10"])

    # Align y-labels per column
    fig.align_ylabels(axes[:, 0])
    fig.align_ylabels(axes[:, 1])
    fig.align_ylabels(axes[:, 2])

    # Legend outside
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="center left", bbox_to_anchor=(-0.02, 0.68), frameon=False)
    fig.tight_layout()
    fig.subplots_adjust(right=0.82)
    fig.savefig("triple_pendulum_npz_time_history.png", dpi=400, bbox_inches="tight", transparent=True)


if __name__ == "__main__":
    main()
