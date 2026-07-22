# Figure 6 source data and reproduction

This directory is a self-contained plotting package for manuscript Figure 6.
The large generated MATLAB intermediates under `results/` are intentionally not
copied into Git.  The four `*_plot_data.npz` files contain only the arrays that
are displayed in the paper: the reference and SREINet-predicted derivatives (or
derivative magnitudes), their plotting time coordinates, and panel metadata.

Run from the repository root:

```bash
python data/Figure_6_source_data/plot_figure_6_components.py
```

This writes the five panels to `reproduced/`.  `components/` contains the exact
panels used in the manuscript, `Figure_6.png` is the current assembled figure,
and `Figure_6_composition.ai` is its editable assembly source.  The summary loss
and coefficient-error values are also retained in `Figure_6_loss_data.xlsx`.

To rebuild the compact NPZ files from a local results tree, run:

```bash
python data/Figure_6_source_data/build_compact_data_from_results.py
```

Source mapping:

- Lorenz-96: `results/Lorenz_96_100D/{train_data,validation_data}.mat`
- discrete phi4: `results/Phi4/train_data.mat`
- DNLS: `results/DNLS_100D/train_data.mat`
- Ablowitz-Ladik: `results/AL_128D/train_data.mat`
- summary bars: `results/four_model_errors/identification_error_plot.py`

The pre-compaction scripts copied from those result directories are preserved
in `original_results_scripts/` for provenance.  They require the ignored MAT
files and other local result-directory modules; use the self-contained plotting
script above for normal reproduction.

Dependencies: Python 3, NumPy, Matplotlib, cmocean, and seaborn.  Rebuilding NPZ
files also requires SciPy.  No random seed is involved in these Figure 6 plots.
