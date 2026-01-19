"""
Parse pruning logs and summarize last training losses, epochs, and training times per dimension.

Targets three files in the current directory:
    - hard_hreshold_log.txt
    - SINDy_library_pruning_log.txt
    - SREINet_library_pruning_log.txt

For each file, the script attempts to:
    * capture the last training loss seen for each dimension (lines like "Epoch ... Train Loss: ...")
    * capture the last epoch index for each dimension
    * capture the training time for each dimension (lines like "Training Stopped. Time elapsed: 4.89s")
Then it reports per-file values (one line per dimension) and averages across dimensions in that file.

Run with:  python parse_pruning_logs.py
"""

from __future__ import annotations

import re
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple


LOG_FILES = [
    "SINDy_hard_threshold_log.txt",
    "SINDy_library_pruning_log.txt",
    "SREINet_library_hard_threshold.txt",
    "SREINet_library_pruning_log.txt",
    
]

# Regex patterns to capture relevant values
DIM_START_PATTERN = re.compile(r"Training\s+dimension\s+(\d+)", re.IGNORECASE)
EPOCH_LINE_PATTERN = re.compile(
    r"Epoch\s+(\d+)\s*\([^)]*\):\s*Train\s+Loss:\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)
TIME_PATTERN = re.compile(
    r"Training\s+Stopped\.\s*Time\s+elapsed:\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)s",
    re.IGNORECASE,
)


def parse_log(path: Path) -> List[Dict[str, Optional[float]]]:
    """Parse a single log file and return a list of per-dimension summaries.

    Returns:
        List of dicts with keys: dim (int), loss (float|None), epochs (int|None), train_time (float|None)
    """
    summaries: List[Dict[str, Optional[float]]] = []
    if not path.exists():
        return summaries

    current_dim: Optional[int] = None
    current_loss: Optional[float] = None
    current_epoch: Optional[int] = None
    current_time: Optional[float] = None

    def flush_dim():
        nonlocal current_dim, current_loss, current_epoch, current_time
        if current_dim is not None:
            summaries.append(
                {
                    "dim": current_dim,
                    "loss": current_loss,
                    "epochs": current_epoch,
                    "train_time": current_time,
                }
            )
        current_dim = None
        current_loss = None
        current_epoch = None
        current_time = None

    for line in path.read_text().splitlines():
        dim_match = DIM_START_PATTERN.search(line)
        if dim_match:
            flush_dim()
            current_dim = int(dim_match.group(1))
            continue

        epoch_match = EPOCH_LINE_PATTERN.search(line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            current_loss = float(epoch_match.group(2))
            continue

        time_match = TIME_PATTERN.search(line)
        if time_match:
            current_time = float(time_match.group(1))
            continue

    flush_dim()
    return summaries


def summarize() -> None:
    file_summaries = []
    for fname in LOG_FILES:
        path = Path(fname)
        per_dim = parse_log(path)
        file_summaries.append({"file": fname, "per_dim": per_dim})

    print("Per-file per-dimension summary:")
    for summary in file_summaries:
        print(f"\n{summary['file']}:")
        if not summary["per_dim"]:
            print("  No dimension entries found.")
            continue
        for entry in sorted(summary["per_dim"], key=lambda d: d["dim"]):
            print(
                f"  dim {entry['dim']:>2}: last loss={entry['loss']}, "
                f"epochs={entry['epochs']}, time(s)={entry['train_time']}"
            )

    print("\nPer-file averages (across dimensions present in that file):")
    for summary in file_summaries:
        per_dim = summary["per_dim"]
        if not per_dim:
            print(f"- {summary['file']}: no data")
            continue
        losses = [d["loss"] for d in per_dim if d["loss"] is not None]
        epochs = [d["epochs"] for d in per_dim if d["epochs"] is not None]
        times = [d["train_time"] for d in per_dim if d["train_time"] is not None]

        loss_avg = f"{mean(losses):.6g}" if losses else "n/a"
        epoch_avg = f"{mean(epochs):.2f}" if epochs else "n/a"
        time_avg = f"{mean(times):.6g}" if times else "n/a"

        print(f"- {summary['file']}: loss={loss_avg}, epochs={epoch_avg}, time={time_avg}")


if __name__ == "__main__":
    summarize()
