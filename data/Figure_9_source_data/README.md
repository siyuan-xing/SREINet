# Figure 9 source data and reproduction

This directory contains the four extrapolation panels used in manuscript
Figure 9: a discrete phi4 kink, a DNLS Gaussian wave packet, an Ablowitz-Ladik
Peregrine wave, and an Ablowitz-Ladik Kuznetsov-Ma breather.

As requested, the final composite image and Illustrator file are not duplicated
here.  `components/` contains the exact four panels used by the manuscript.

The four plotting scripts and their local Python dependencies are copied without
modification from the corresponding result directories.  Run them from their
own directories so all original relative imports and output names remain valid:

```bash
cd data/Figure_9_source_data/phi4
python postprocessing-kink.py

cd ../dnls
python postprocessing-GaussianI_IC.py

cd ../abl
python postprocessing_peregrine.py
python postprocessing_KM.py
```

Each script regenerates its analytic initial condition and both the ground-truth
and SREINet-discovered trajectories in Python.  The compact `*_plot_data.npz`
files separately archive the complete time coordinates and plotted state arrays;
they are provided as convenient source data and do not replace the original
plotting workflow.

To regenerate the compact arrays from the local result code, run:

```bash
python data/Figure_9_source_data/build_compact_data_from_results.py
```

The builder regenerates the same analytic initial conditions, integrates the
ground-truth systems through each result directory's `DataGenerator`, and
integrates the recovered equations with SciPy.  No MAT files are required or
copied.  The manuscript plotting scripts themselves are not reformatted or
adapted.

Dependencies: Python 3, NumPy, SciPy, Matplotlib, and cmocean.
