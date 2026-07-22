#!/usr/bin/env python3
import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects


import seaborn as sns

# Styling consistent with simulate_and_compare.py
plt.rcParams.update(
    {
        "axes.grid": False,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "font.size": 10,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "font.family": "Helvetica",
        'axes.titleweight': 'bold',
    }
)
palette = sns.color_palette("Set2")


LINE_RE = re.compile(
    r"Epoch\s+(?P<epoch>\d+)\s+\(Period\s+(?P<period>\d+)\):\s+"
    r"Train Loss:\s+(?P<train>[0-9.eE+-]+),\s+"
    r"Validation Loss:\s+(?P<val>[0-9.eE+-]+)"
)

PLOT_TITLES = [r'$\phi_4$', 'Kuramoto', 'Hindmarsh-Rose']
X_TICKS = [None, None, [0,9,18,27,36]]
Y_TICKS = [None, None, None]


def parse_log(path: Path):
    epochs = []
    periods = []
    train_losses = []
    val_losses = []
    period_starts = []
    period_first_epoch = {}
    seen_periods = set()

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            match = LINE_RE.search(line)
            if not match:
                continue
            epoch = int(match.group("epoch"))
            period = int(match.group("period"))
            period_zero = period - 1
            train = float(match.group("train"))
            val = float(match.group("val"))

            epochs.append(epoch)
            periods.append(period_zero)
            train_losses.append(train)
            val_losses.append(val)

            if period_zero not in seen_periods:
                period_starts.append(period_zero)
                seen_periods.add(period_zero)
                period_first_epoch[period_zero] = epoch

    return epochs, periods, train_losses, val_losses, period_starts, period_first_epoch


def plot_files(paths, loss_mode, output_path, show_plot):

    fig, axes = plt.subplots(1, len(paths), figsize=(7.0, 2.7), sharey=False)
    if len(paths) == 1:
        axes = [axes]

    for idx, (ax, path) in enumerate(zip(axes, paths)):
        (
            epochs,
            periods,
            train_losses,
            val_losses,
            period_starts,
            period_first_epoch,
        ) = parse_log(path)

        if periods:
            period_counts = {}
            for period in periods:
                period_counts[period] = period_counts.get(period, 0) + 1

            period_x = [
                period + (epoch - period_first_epoch[period]) / period_counts[period]
                for epoch, period in zip(epochs, periods)
            ]
            if loss_mode in ("train", "both"):
                ax.plot(period_x, train_losses, label="Train", linewidth=1.2)
            if loss_mode in ("val", "both"):
                ax.plot(period_x, val_losses, label="Validation", linewidth=1.2)

            for period in period_starts:
                ax.axvline(period, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

            if loss_mode == "both" and idx == 0:
                ax.legend(frameon=False)
        else:
            ax.text(
                0.5,
                0.5,
                "No data parsed",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        ax.set_xlabel("Pruning Period")
        if ax is axes[0]:
            ax.set_ylabel("Loss")
        else:
            ax.set_ylabel("")
        ax.set_yscale("log")
        title = PLOT_TITLES[idx] if idx < len(PLOT_TITLES) else None
        title_artist = ax.set_title(
            title if title is not None else path.stem.replace("_", " "),
            fontweight='bold',
        )
        title_artist.set_path_effects(
            [path_effects.withStroke(linewidth=0.25, foreground='black')]
        )
        if idx < len(X_TICKS) and X_TICKS[idx] is not None:
            ax.set_xticks(X_TICKS[idx])
        if idx < len(Y_TICKS) and Y_TICKS[idx] is not None:
            ax.set_yticks(Y_TICKS[idx])

    fig.tight_layout()
    fig.savefig(output_path, dpi=600)
    if show_plot:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot loss vs. epoch for training logs."
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Paths to log .txt files (default: all .txt files in the folder).",
    )
    parser.add_argument(
        "--loss",
        choices=["train", "val", "both"],
        default="both",
        help="Which loss curves to plot.",
    )
    parser.add_argument(
        "--output",
        default="training_dynamics.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot window.",
    )
    args = parser.parse_args()

    if args.files:
        paths = [Path(p) for p in args.files]
    else:
        paths = sorted(Path(".").glob("*.txt"))

    preferred_order = [
        "Phi_4_log_sample",
        "Kuramoto_log_sample",
        "Hindmarsh_rose_network",
    ]
    order_index = {name: idx for idx, name in enumerate(preferred_order)}
    paths.sort(key=lambda p: order_index.get(p.stem, len(preferred_order)))

    plot_files(paths, args.loss, Path(args.output), args.show)
   


if __name__ == "__main__":
    main()
