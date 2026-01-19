import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Styling kept consistent with plot_npz_time_history.py
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
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{helvet}\renewcommand{\familydefault}{\sfdefault}",
    }
)
palette = sns.color_palette("Set2")
true_color = palette[0]
pred_color = palette[1]


def load_layer_series(layer: int) -> dict:
    """Load true/predicted series for a given layer count."""
    data = np.load(f"output/triple_pendulum_{layer}layers_data.npz")

    t_pred = data["SREINet_t"].ravel()
    y_pred = np.transpose(data["SREINet_y"])
    val_t = data["val_t_eval"].ravel()
    val_x = data["val_x"]
    pred_dx = data["pred_dx"]
    val_dd_theta = data["val_dd_theta"]

    return {
        "t_pred": t_pred,
        "val_t": val_t,
        "theta_true": val_x[:, 0],
        "dtheta_true": val_x[:, 3],
        "ddtheta_true": val_dd_theta[:, 0],
        "theta_pred": y_pred[:, 0],
        "dtheta_pred": y_pred[:, 3],
        "ddtheta_pred": pred_dx[3, :],
    }


def setup_panel(
    ax: plt.Axes,
    val_t: np.ndarray,
    t_pred: np.ndarray,
    truth: np.ndarray,
    pred: np.ndarray,
    label: str,
) -> None:
    ax.set_facecolor("white")
    ax.plot(val_t, truth[: len(val_t)], label="True", color=true_color, lw=1.0)
    ax.plot(t_pred, pred, label="Pred", color=pred_color, lw=1.0, ls="--")
    ax.set_ylabel(label)
    ax.tick_params(direction="in")


def main() -> None:
    layers = [2, 3, 4]
    fig, axes = plt.subplots(
        3, len(layers), figsize=(9.0, 4.5), sharex=False, facecolor="none"
    )

    for col, layer in enumerate(layers):
        series = load_layer_series(layer)
        axes[0, col].set_title(f"{layer} layers")

        setup_panel(
            axes[0, col],
            series["val_t"],
            series["t_pred"],
            series["theta_true"],
            series["theta_pred"],
            r"$\theta_1$",
        )
        setup_panel(
            axes[1, col],
            series["val_t"],
            series["t_pred"],
            series["dtheta_true"],
            series["dtheta_pred"],
            r"$\dot{\theta}_1$",
        )
        setup_panel(
            axes[2, col],
            series["val_t"],
            series["t_pred"],
            series["ddtheta_true"],
            series["ddtheta_pred"],
            r"$\ddot{\theta}_1$",
        )

        # Sync limits per column using both time arrays
        x_min = min(series["val_t"].min(), series["t_pred"].min())
        x_max = max(series["val_t"].max(), series["t_pred"].max())
        for row in range(3):
            axes[row, col].set_xlim(x_min, x_max)
            if col > 0:
                axes[row, col].set_ylabel("")
                axes[row, col].set_yticklabels([])
                axes[row, col].tick_params(labelleft=False)

    # Hide x tick labels on upper rows
    for row in range(2):
        for col in range(len(layers)):
            axes[row, col].set_xticklabels([])

    # X labels only on bottom row
    for col in range(len(layers)):
        axes[2, col].set_xlabel("Time, $t$ [s]")
        axes[2, col].set_xticks([0, 5, 10])
        axes[2, col].set_xticklabels([r"$0$", r"$5$", r"$10$"])

    # Align y-labels per row
    fig.align_ylabels(axes[0, :])
    fig.align_ylabels(axes[1, :])
    fig.align_ylabels(axes[2, :])

    # Legend inside first subplot
    axes[0, 0].legend(
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=1.0,
    )
    fig.tight_layout()
    fig.savefig(
        "triple_pendulum_theta1_layers_time_history.png",
        dpi=400,
        bbox_inches="tight",
        transparent=False,
    )


if __name__ == "__main__":
    main()
