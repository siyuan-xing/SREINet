# Figure 12 source data and reproduction

This directory contains the timing data and current manuscript image for the
FPU data-scalability comparison among MANDy, SREINet, EQL, and SymbolNet.

The original aggregation-and-plotting script is copied without modification into
a replica of its expected directory layout.  Run:

```bash
cd "data/Figure_12_source_data/Comparison with Mandy"
python plot_fpu_time_scalability_all_methods.py
```

The unchanged script reads the copied EQL/SymbolNet source summary from the
adjacent `time scalability/` tree, combines it with the original MANDy/SREINet
timings, and writes its CSV, PNG, and EPS in `Comparison with Mandy/`.  Its
generated CSV contains the plotted 3000-point SREINet value (linearly
extrapolated from the 4000/5000-point runs) and the EQL/SymbolNet discovery-rate
fields.  The archived root CSV is the observed-timing table before that plotting
step; it contains 8 EQL, 8 MANDy, 7 observed SREINet, and 8 SymbolNet rows.

`MANDy_SREINet_legacy_source_data.xlsx` is retained as legacy source material.
`Figure_12.png` is the exact current manuscript image.

Dependencies: Python 3, NumPy, and Matplotlib.
