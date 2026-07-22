# Figure 14 source data and reproduction

This directory contains representative SREINet training logs and the plotting
script for manuscript Figure 14. The three panels show fast, moderate, and
robust convergence for the discrete phi4, Kuramoto, and Hindmarsh-Rose systems.

Run from this directory to use the three included logs in manuscript order:

```bash
cd data/Figure_14_source_data
python plot_training_dynamics.py
```

By default, the script plots both training and validation losses and writes
`training_dynamics.png`. Use `--loss train`, `--loss val`, or `--loss both` to
select the curves, and `--output PATH` to choose a different output filename.
Explicit log paths may also be supplied as positional arguments.

Dependencies: Python 3, Matplotlib, and seaborn.
