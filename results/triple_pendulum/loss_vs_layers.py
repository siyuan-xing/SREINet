import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties

# Use non-LaTeX text so axis labels render with the standard font

params = {
    'image.origin': 'lower',
    'image.interpolation': 'nearest',
    'axes.grid': False,
    'savefig.dpi': 600,  # to adjust notebook inline plot size
    'axes.titlesize': 10,
    'text.usetex': True,
    #'figure.figsize': [7, 4],
    'font.family': 'Helvetica',
    #'text.latex.preamble': r'\usepackage{amsmath}',  # Optional, for advanced math symbols
    }

plt.rcParams.update(params)

# Match the color theme used in bar_chart_loss.py
bright_palette = sns.color_palette("Set2")
sns.set_theme(
    context="paper",
    style="white",
    palette=bright_palette,
    font="Arial",
    font_scale=0.8,
)

layer_list = [2, 3, 4]
mean_loss_vs_layer = [3.941721, 0.59171486, 0.3665725]

# Create bar chart
fig, ax = plt.subplots(figsize=(4, 3))

bars = ax.bar(
    layer_list,
    mean_loss_vs_layer,
    color=bright_palette[2],
    alpha=0.9,
    zorder=2,
)

# Set x, y labels with Helvetica font (no LaTeX)
helvetica_font = FontProperties(family='Helvetica', size=10)
ax.set_xlabel("Number of Layers", fontproperties=helvetica_font)
ax.set_ylabel("MSE Loss", fontproperties=helvetica_font)
# Disable LaTeX for x, y labels by accessing the Text objects
ax.xaxis.label.set_usetex(False)
ax.yaxis.label.set_usetex(False)
ax.set_xticks(layer_list)
ax.set_yticks([0, 1, 2, 3, 4])
# Tick labels will use LaTeX (from text.usetex = True)
ax.set_xticklabels(layer_list, fontsize=10)
ax.set_yticklabels([0, 1, 2, 3, 4], fontsize=10)

# For bar charts: x-axis (categorical) doesn't need tick marks, only y-axis (numerical) needs them
ax.tick_params(axis='x', which='major', length=0)  # Remove x-axis tick marks
ax.tick_params(axis='y', which='major', length=5, width=0.8, color='black', direction='out')  # Keep y-axis tick marks

# Light grid to mirror bar_chart_loss styling
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("loss_vs_layer_triple_pendulum.png", dpi=600, bbox_inches="tight")
plt.show()
