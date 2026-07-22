# Figure 13 source data and reproduction

This directory contains the memory-consumption values, plotting script, and
current panel for manuscript Figure 13. The comparison reports the nominal
memory required by SREINet and the conventional matrix formulation across
system dimensions and nonlinearity orders.

Run from this directory:

```bash
cd data/Figure_13_source_data
python memory_plot.py
```

The script writes `memory_consumption.png` and exports the plotted values to
`memory_consumption.xlsx`. The spreadsheet and current manuscript rendering are
also retained directly in this directory.

Dependencies: Python 3, NumPy, pandas, Matplotlib, seaborn, and openpyxl.
