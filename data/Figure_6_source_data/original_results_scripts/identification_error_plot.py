import matplotlib.pyplot as plt
import numpy as np

# Parameters for styling
FS_SMALL = 10
FS_MEDIUM = 12
FS_LARGE = 14

params = {
    'image.origin': 'lower',
    'image.interpolation': 'nearest',
    'image.cmap': 'gray',
    'axes.grid': False,
    'savefig.dpi': 600,
    'axes.labelsize': FS_MEDIUM,
    'axes.titlesize': FS_LARGE,
    'font.size': FS_SMALL,
    'legend.fontsize': FS_SMALL,
    'xtick.labelsize': FS_SMALL,
    'ytick.labelsize': FS_SMALL,
    'font.family': 'Helvetica',
}
import matplotlib
matplotlib.rcParams.update(params)

import seaborn as sns
sns.set_palette('deep')

# Data for the models
#models = ['Lorenz-96', 'Kuramoto', r'$\phi^4$', 'DNLS', 'AL']
#L_ave = [8.4e-11, 5.4e-11, 1.1e-11, 4.0e-12, 1.7e-11]
#e_coef = [3.0e-6, 9.9e-7, 9.9e-8, 8.5e-8, 7.4e-7]

models = ['Lorenz-96', r'$\phi^4$', 'DNLS', 'AL']
L_ave = [8.4e-11, 1.1e-11, 4.0e-12, 1.7e-11]
e_coef = [3.0e-6, 9.9e-8, 8.5e-8, 7.4e-7]


# Setting up for grouped bar plot
x = np.arange(len(models))  # label locations
width = 0.35  # width of the bars

fig, ax = plt.subplots(figsize=(7.0, 3.1))

# Custom colors
custom_colors = ['#B8DBB3', '#719AAC']

# Plotting bars for L_ave and e_coef
bars1 = ax.bar(x - width/2, L_ave, width, label='Loss', color=custom_colors[0], alpha=0.75)
bars2 = ax.bar(x + width/2, e_coef, width, label='Coefficient error', color=custom_colors[1], alpha=0.75)

# Adding labels, title, and legend
ax.set_ylabel('Values')
ax.set_yscale('log')
ax.set_xticks(x)
ax.set_yticks([1e-11, 1e-9, 1e-7, 1e-5,1e-3])
ax.set_xticklabels(models)
ax.legend()

# Adjust layout and save the figure
fig.tight_layout()
plt.savefig('combined_identification_error_plot.png', dpi=600)
