# Figure 7 source data and reproduction

This directory is a self-contained plotting package for manuscript Figure 7.
It contains the plotted truth/prediction trajectories for the Hindmarsh-Rose
and Kuramoto networks, the Hindmarsh-Rose edge list, plotting code, the exact
component images, the assembled manuscript figure, and its Illustrator source.

Run from the repository root:

```bash
python data/Figure_7_source_data/plot_figure_7_components.py
```

The three recreated panels are written to `reproduced/`.  The compact NPZ files
are sufficient for plotting and replace the much larger generated MAT files.
They retain all values visible in Figure 7, including the time-series traces.

To recreate the compact NPZ files in a checkout that has the ignored local
results, run:

```bash
python data/Figure_7_source_data/build_compact_data_from_results.py
```

That script uses:

- `results/Hindmarsh_rose_network-75D/data.mat` and `hindmarsh_rose_ode.py`
- `results/Kuramoto_60D/train_data.mat` and `kuramoto_ode.py`
- SciPy `solve_ivp(..., method="BDF")`, matching the original panel scripts

`components/` contains the exact panels used by the manuscript.  The original
results-side plotting scripts are retained in `original_results_scripts/` for
provenance, but depend on ignored generated files.  `Figure_7_composition.ai`
records the final manual assembly of those components.

Dependencies: Python 3, NumPy, Matplotlib, cmocean, and NetworkX.  Rebuilding
the NPZ files also requires SciPy.  The network layout uses seed 42.
