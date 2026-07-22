# Figure 11 source data and reproduction

This directory contains the recovered-equation text outputs and the original
plotting script for manuscript Figure 11, which summarizes the combined effects
of intermittent noise and missing data on coefficient recovery.

Run from this directory so the script can find its original relative input
folder:

```bash
cd data/Figure_11_source_data
python intermittent_noise_incomplete_data.py
```

The script reads three recovered-equation outputs for each noise, pollution,
and missing-data setting under `Incomplete Noise/`, averages their coefficient
ratios, and writes `intermittent_noise_incomplete_data.png`. Hatched cells mark
settings for which the recovered equation structure is incorrect, matching the
manuscript convention.

The plotting code and text outputs are retained in their original layout.
Dependencies: Python 3, NumPy, Matplotlib, and seaborn.
