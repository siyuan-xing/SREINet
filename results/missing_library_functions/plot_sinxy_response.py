import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_phase(path: str):
    """
    Return predicted and true phase coordinates (x, dx) for the first channel.
    true_y stores the ground-truth derivatives in the saved files.
    """
    data = np.load(path)
    pred_x = data["val_x"][:, 0]
    pred_dx = data["val_dx"][:, 0]
    true_x = data["true_x"][:, 0]
    true_dx = data["true_y"][:, 0]
    return (pred_x, pred_dx), (true_x, true_dx)


def main():
    # High-contrast pairing for easy visual separation.
    palette = ["#000000", "#d62728"]
    sns.set_theme(
        context="paper",
        style="white",
        palette=palette,
        font="Arial",
        font_scale=0.9,
    )

    plt.rcParams.update(
        {
            "savefig.dpi": 600,
            "font.family": "Helvetica",
            "text.usetex": True,
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 9,
        }
    )

    cases = [
        ("Small IC", "sin(xy)_nonlinearity_small_IC.npz"),
        ("Large IC", "sin(xy)_nonlinearity_large_IC.npz"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7, 3), sharex=False, sharey=False)

    for idx, (ax, (title, path)) in enumerate(zip(axes, cases)):
        (pred_x, pred_dx), (true_x, true_dx) = load_phase(path)
        ax.plot(true_x, true_dx, label=r"$\mathrm{Ground~truth}$", linewidth=2.0, color=palette[0])
        ax.plot(
            pred_x,
            pred_dx,
            label=r"$\mathrm{Prediction}$",
            linewidth=2.0,
            linestyle="--",
            color=palette[1],
        )
        # Mark initial condition with a blue circle (no legend entry).
        ax.scatter(true_x[0], true_dx[0], color="#1f77b4", s=30, zorder=5, label="_nolegend_")
        ax.set_title(title)
        ax.set_xlabel(r"$x_1$")
        ax.grid(alpha=0.2)

    axes[0].set_ylabel(r"$\dot{x}_1$")
    axes[0].legend(frameon=False)
    axes[0].set_ylim([-0.4, 0.6])

    fig.tight_layout(w_pad=1.5)
    fig.savefig("sinxy_response.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
