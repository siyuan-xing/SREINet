# Figure 10 source data and reproduction

This directory contains the coefficient-recovery summaries and plotting code
for the continuous-Gaussian-noise experiment, the exact three component images,
the current assembled manuscript figure, and its Illustrator composition source.

All three plotting scripts are retained without modification.  Run each from
the directory expected by its original relative paths:

```bash
cd data/Figure_10_source_data
python MSE_vs_noise_level.py

cd "MSE_vs _dimensions"
python coefficient_error_vs_dim.py

cd ../phase_space_original/output
python continuous_noise_time_simu_state_space.py
```

The scripts write their component images in their respective current
directories:

- coefficient MSE versus system dimension;
- coefficient MSE versus noise level;
- phase-space trajectories for 1%, 5%, 10%, and 15% noise.

The first two panels are computed directly from the archived text outputs.  For
the third panel, `phase_space_original/` recreates the original relative layout:
the unchanged script, the unchanged utility modules, and the required recovered
equation logs.  The original script did not record the random seed for injected
noise, so exact noisy trajectories may vary between reruns; the exact manuscript
rendering is preserved in `components/`.

Generated MAT datasets are not needed by these plotting scripts and are not
included.  The removed local copies have byte-identical originals under
`code/noise/results_with_dimensions/` and can also be regenerated from Python.

Dependencies: Python 3, NumPy, SciPy, pandas, Matplotlib, seaborn, and cmocean.
