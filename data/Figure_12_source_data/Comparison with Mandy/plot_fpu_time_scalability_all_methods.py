#!/usr/bin/env python3
"""Plot FPU timing scalability for MANDy, SREINet, EQL, and SymbolNet."""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sreinet-matplotlib-cache"))

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
if SCRIPT_PATH.parent.name == "Comparison with Mandy":
    OUTPUT_ROOT = SCRIPT_PATH.parent
    CODE_ROOT = SCRIPT_PATH.parents[1]
else:
    OUTPUT_ROOT = SCRIPT_PATH.parents[1]
    CODE_ROOT = OUTPUT_ROOT.parent / "SREINet-Git" / "code"
EQL_SYMBOLNET_SUMMARY = (
    CODE_ROOT
    / "time scalability"
    / "results_fpu_eql_symbolnet_full_outputs_combined"
    / "fpu_eql_symbolnet_full_outputs_summary.csv"
)
if SCRIPT_PATH.parent.name == "Comparison with Mandy":
    OUTPUT_CSV = OUTPUT_ROOT / "fpu_time_scalability_all_methods.csv"
    OUTPUT_PNG = OUTPUT_ROOT / "time_comparison.png"
    OUTPUT_EPS = OUTPUT_ROOT / "time_comparison.eps"
else:
    OUTPUT_CSV = OUTPUT_ROOT / "tools" / "fpu_time_scalability_all_methods.csv"
    OUTPUT_PNG = OUTPUT_ROOT / "figures" / "time_comparison.png"
    OUTPUT_EPS = OUTPUT_ROOT / "figures" / "time_comparison.eps"


def summary_rows() -> list[dict[str, object]]:
    data_points_mandy = np.array([3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])
    mandy_times = np.array(
        [
            [47.81, 96.14, 189.4, 285.1, 415.2, 569.67, 794.29, 1115.75],
            [43.44, 95.52, 203.6, 286.64, 442.3, 589.51, 790.03, 1116.14],
            [43.64, 95.69, 209.1, 291.0, 437.52, 604.41, 780.67, 1115.89],
        ]
    )

    data_points_sreinet_observed = np.array([4000, 5000, 6000, 7000, 8000, 9000, 10000])
    sreinet_times = np.array(
        [
            [67.2, 65.79, 89.45, 88.12, 75.31, 66.67, 97.12],
            [109.93, 107.91, 96.5, 105.37, 132.78, 134.82, 186.44],
            [99.66, 77.56, 83.03, 80.85, 77.55, 79.69, 88.64],
            [79.49, 78.02, 86.24, 84.17, 68.29, 124.12, 82.94],
            [112.51, 59.3, 124.32, 74.64, 72.97, 113.84, 91.09],
            [92.02, 89.07, 139.61, 70.44, 78.13, 81.48, 169.06],
            [105.84, 82.35, 93.37, 84.3, 68.676, 90.18, 86.72],
            [69.42, 105.16, 113.11, 78.29, 76.09, 79.89, 150.81],
            [124.04, 74.24, 73.29, 83.79, 73.42, 130.03, 91.87],
            [82.27, 122.56, 99.47, 86.34, 69.01, 104.71, 138.26],
            [94.238, 86.196, 99.839, 83.631, 79.2226, 100.543, 118.295],
        ]
    )
    sreinet_3000 = sreinet_times[:, 0] + (3000 - 4000) * (
        sreinet_times[:, 1] - sreinet_times[:, 0]
    ) / (5000 - 4000)
    data_points_sreinet = np.insert(data_points_sreinet_observed, 0, 3000)
    sreinet_times = np.column_stack([sreinet_3000, sreinet_times])

    rows: list[dict[str, object]] = []
    for method, label, points, times in (
        ("mandy", "MANDy", data_points_mandy, mandy_times),
        ("sreinet", "SREINet", data_points_sreinet, sreinet_times),
    ):
        means = np.mean(times, axis=0)
        stds = np.std(times, axis=0)
        repeats = times.shape[0]
        for point, mean, std in zip(points, means, stds):
            rows.append(
                {
                    "method": method,
                    "method_label": label,
                    "data_points": int(point),
                    "time_mean": float(mean),
                    "time_std": float(std),
                    "repeats": repeats,
                    "num_output_dims": "",
                    "batch_size": 64 if method == "sreinet" else "",
                    "epochs": 480 if method == "sreinet" else "",
                    "early_stopping": (
                        "loss_threshold=1e-8; pruning_threshold=0.01"
                        if method == "sreinet"
                        else ""
                    ),
                    "main_rate": "",
                    "exact_rate": "",
                    "discovery_status": "",
                    "source": (
                        "linear extrapolation from 4000/5000 SREINet timing rows"
                        if method == "sreinet" and int(point) == 3000
                        else "Comparison with Mandy/test_plot.py"
                    ),
                }
            )

    with EQL_SYMBOLNET_SUMMARY.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["method"] not in {"eql", "symbolnet"}:
                continue
            rows.append(
                {
                    "method": row["method"],
                    "method_label": row["method_label"],
                    "data_points": int(row["data_points"]),
                    "time_mean": float(row["time_mean"]),
                    "time_std": float(row["time_std"]),
                    "repeats": int(float(row["repeats"])),
                    "num_output_dims": int(float(row["num_output_dims"])),
                    "batch_size": 512,
                    "epochs": 1500,
                    "early_stopping": "disabled",
                    "main_rate": float(row["main_rate"]),
                    "exact_rate": float(row["exact_rate"]),
                    "discovery_status": discovery_status(float(row["exact_rate"])),
                    "source": str(EQL_SYMBOLNET_SUMMARY.relative_to(CODE_ROOT)),
                }
            )

    return sorted(rows, key=lambda item: (str(item["method"]), int(item["data_points"])))


def discovery_status(exact_rate: float) -> str:
    if exact_rate >= 1.0:
        return "all equations discovered"
    if exact_rate > 0.0:
        return "partial equations discovered"
    return "no exact equations discovered"


def write_summary(rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "method",
        "method_label",
        "data_points",
        "time_mean",
        "time_std",
        "repeats",
        "num_output_dims",
        "batch_size",
        "epochs",
        "early_stopping",
        "main_rate",
        "exact_rate",
        "discovery_status",
        "source",
    ]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, object]]) -> None:
    matplotlib.rcParams.update(
        {
            "savefig.dpi": 600,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "font.size": 10,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "font.family": "Helvetica",
            "axes.grid": True,
            "grid.color": "#E6E6E6",
            "grid.linewidth": 0.55,
            "grid.alpha": 1.0,
        }
    )
    colors = {
        "mandy": "#66C2A5",
        "sreinet": "#8DA0CB",
        "eql": "#F58518",
        "symbolnet": "#54A24B",
    }
    markers = {"mandy": "o", "sreinet": "o", "eql": "s", "symbolnet": "^"}
    linestyles = {"mandy": "-", "sreinet": "--", "eql": "-.", "symbolnet": ":"}
    order = ["mandy", "sreinet", "eql", "symbolnet"]

    fig, ax = plt.subplots(figsize=(7.0, 4.3), constrained_layout=True)
    for method in order:
        series = [row for row in rows if row["method"] == method]
        x = np.array([row["data_points"] for row in series], dtype=float)
        y = np.array([row["time_mean"] for row in series], dtype=float)
        err = np.array([row["time_std"] for row in series], dtype=float)
        label = str(series[0]["method_label"])
        ax.errorbar(
            x,
            y,
            color=colors[method],
            linestyle=linestyles[method],
            linewidth=1.9,
            marker=None,
        )
        ax.fill_between(
            x,
            y - err,
            y + err,
            color=colors[method],
            alpha=0.16,
            linewidth=0,
            zorder=1,
        )
        if method in {"eql", "symbolnet"}:
            for row in series:
                exact_rate = float(row["exact_rate"])
                facecolor = colors[method] if exact_rate >= 1.0 else "white"
                ax.plot(
                    float(row["data_points"]),
                    float(row["time_mean"]),
                    color=colors[method],
                    marker=markers[method],
                    linestyle="none",
                    markersize=5.8,
                    markerfacecolor=facecolor,
                    markeredgecolor=colors[method],
                    markeredgewidth=1.25,
                    zorder=4,
                )
        else:
            for row in series:
                data_points = float(row["data_points"])
                facecolor = "white" if method == "sreinet" and data_points == 3000 else colors[method]
                ax.plot(
                    data_points,
                    float(row["time_mean"]),
                    color=colors[method],
                    marker=markers[method],
                    linestyle="none",
                    markersize=5.2,
                    markerfacecolor=facecolor,
                    markeredgecolor=colors[method],
                    markeredgewidth=1.25,
                    zorder=4,
                )

    ax.set_xlabel("Data Points")
    ax.set_ylabel("Time (s)")
    ax.set_xticks([3000, 5000, 7000, 9000])
    ax.set_yticks([0, 300, 600, 900, 1200])
    ax.set_ylim(0, 1225)
    ax.set_xlim(2650, 10350)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    method_handles = [
        Line2D([0], [0], color=colors[method], marker=markers[method],
               linestyle=linestyles[method], linewidth=1.8, markersize=5,
               label=label)
        for method, label in (
            ("mandy", "MANDy"),
            ("sreinet", "SREINet"),
            ("eql", "EQL"),
            ("symbolnet", "SymbolNet"),
        )
    ]
    ax.legend(
        handles=method_handles,
        frameon=False,
        loc="upper left",
        ncol=2,
        handlelength=2.5,
        columnspacing=1.25,
        borderaxespad=0.6,
    )
    ax.text(
        0.022,
        0.835,
        "Open markers: partially discovered; solid markers: fully discovered.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color="#555555",
    )
    fig.savefig(OUTPUT_PNG, dpi=600)
    fig.savefig(OUTPUT_EPS)
    plt.close(fig)


def main() -> None:
    rows = summary_rows()
    write_summary(rows)
    plot(rows)
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_PNG}")
    print(f"Wrote {OUTPUT_EPS}")


if __name__ == "__main__":
    main()
