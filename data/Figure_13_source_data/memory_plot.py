import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

# Set Seaborn color palette
#sns.set_palette("muted")

#colors = sns.color_palette("muted", desat=0.8)

colors = ["#ccebc5", "#b3cde3", "#fbb4ae"]
#custom_colors = ['#B8DBB3', '#1E90FF']


import matplotlib

params = {
    'image.origin': 'lower',
    'image.interpolation': 'nearest',
    'image.cmap': 'gray',
    'axes.grid': False,
    'savefig.dpi': 600,  # to adjust notebook inline plot size
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'font.size': 10,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'font.family': 'Helvetica',
}

matplotlib.rcParams.update(params)


# Data from the first table (SINDy memory consumption) (already in GB)
n_values = [10, 100, 1000]
p_values = [2, 3, 5]

theta_memory = np.array([
    [4.92e-4, 3.83e-2, 3.74],   # p=2
    [2.13e-3, 1.32, 1249.23],    # p=3
    [0.022, 719.43, 6.30e+7]     # p=5
])

xi_memory = np.array([
    [2.46e-6, 1.9e-3, 1.87],   # p=2
    [1.065e-5, 6.59e-2, 624.61],  # p=3
    [1.12e-4, 35.97, 3.15e+7]     # p=5
])

SINDy_Memory = theta_memory + xi_memory

# Data from the second table (converted to GB)

new_structure_memory = [
    [5.26e-6, 8.61e-5, 0.0042],  # p=2
    [5.71e-6, 0.00012,  0.0079],  # p=3
    [6.61e-6, 0.0002, 0.015]  # p=5
]

combopnet_memory = [
   [5.26e-5, 8.61e-3, 4.2],  # p=2
    [5.71e-5, 0.012,  7.9],  # p=3
    [6.61e-5, 0.02, 15.4]  # 
]

# Bar chart settings
bar_width = 0.30
group_spacing = 0.75

n_groups = len(n_values)
index = np.arange(n_groups) * group_spacing

# Create positions for the two remaining bars in each group
srinet_pos = index - bar_width / 2
sindy_pos = index + bar_width / 2


# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.0), sharey=True, gridspec_kw={'wspace': 0.08})

for i, (ax, p) in enumerate(zip(axes, p_values)):
    # Plot for SREINet memory consumption (in GB)
    ax.bar(srinet_pos, new_structure_memory[i], bar_width, label=f'SREINet', color=colors[1], edgecolor="white", linewidth=0.5)
    
    # Plot for SINDy memory consumption (in GB)
    ax.bar(sindy_pos, SINDy_Memory[i], bar_width, label=f'Matrix formulation', color=colors[2], edgecolor="white", linewidth=0.5)
    

    ax.text(0.05, 0.95, f'$p={p}$', transform=ax.transAxes, fontsize=12,
            fontweight='bold',
            verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.75))
    

    # Configure each subplot
    ax.set_xlabel(r'Dimensions, $n$')
    #ax.set_title(f'Memory Consumption for p={p}')
    ax.set_xticks(index)
    ax.set_xticklabels([str(n) for n in n_values])
    ax.set_yscale('log')  # Apply log scale
    ax.set_ylim(1e-6, 1e9)  # Set the y-axis limits
    ax.set_yticks([1e-6, 1e-3, 1, 1e3, 1e6, 1e9])
    if i == 0:
        ax.legend(
            loc='upper right',
            fontsize=8,
            frameon=False,
        )

# Remove y-axis labels from all but the first plot
for ax in axes[1:]:
    ax.tick_params(left=False)

# General y-label for the figure (on the first subplot only)
axes[0].set_ylabel('Memory (GB)')

# Adjust layout so the plots are closely next to each other
fig.subplots_adjust(left=0.10, right=0.99, bottom=0.21, top=0.96, wspace=0.08)
plt.savefig('memory_consumption.png', dpi=600)

# --- 输出数据到Excel ---
rows = []
for i, p in enumerate(p_values):
    for j, n in enumerate(n_values):
        rows.append({
            'n': n,
            'p': p,
            'SINDy_Memory': SINDy_Memory[i][j],
            'SREINet': new_structure_memory[i][j],
            'CombOpNet': combopnet_memory[i][j]
        })

df = pd.DataFrame(rows)
df.to_excel('memory_consumption.xlsx', index=False)
