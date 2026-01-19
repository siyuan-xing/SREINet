import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import patheffects

# Match the color theme used in bar_chart_loss.py
bright_palette = sns.color_palette("Set2")
sns.set_theme(
    context="paper",
    style="white",
    palette=bright_palette,
    font="Arial",
    font_scale=0.8,
)



params = {
    'image.origin': 'lower',
    'image.interpolation': 'nearest',
    'axes.grid': False,
    'savefig.dpi': 600,  # to adjust notebook inline plot size
    'axes.labelsize': 10, # fontsize for x and y labels (was 10)
    'axes.titlesize': 10,
    'font.size': 10, # was 10
    'legend.fontsize': 10, # was 10
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'text.usetex': True,
    #'figure.figsize': [7, 4],
    'font.family': 'Helvetica',
    #'text.latex.preamble': r'\usepackage{amsmath}',  # Optional, for advanced math symbols
    }

layer_2_loss_list = [8.403482e-11,
 7.41827e-11,
 8.682487e-11,
 8.818136e-11,
 8.340256e-11,
 6.153852e-11,
 7.343075e-11,
 5.533937e-11,
 9.619473e-11,
 9.233123e-11]


layer_3_loss_list = [7.337203e-12,
 1.6088978e-12,
 2.7480023e-12,
 1.4435961e-11,
 5.664637e-12,
 1.1955324e-11,
 1.2864791e-11,
 4.145328e-12,
 6.937347e-11,
 7.8794644e-12]


layer_4_loss_list = [3.427502e-11,
 4.9378375e-12,
 2.8485133e-11,
 8.271709e-12,
 2.7624684e-12,
 7.425539e-12,
 3.0728982e-11,
 2.6255114e-11,
 3.3066726e-11,
 9.39311e-11]

layer_5_loss_list = [2.794202e-12,
 1.9113407e-12,
 1.877923e-12,
 3.0246582e-12,
 2.2589003e-12,
 9.404456e-12,
 2.2328207e-12,
 3.5137778e-11,
 1.9164777e-11,
 4.0501205e-12]
 
layer_6_loss_list = [3.0745012e-12,
 2.9714644e-12,
 1.9159932e-12,
 1.2099135e-11,
 4.4889228e-11,
 2.9414398e-12,
 8.027341e-12,
 2.0181873e-12,
 1.40395915e-11,
 3.5345429e-12]

layer_7_loss_list = [1.7095563e-11,
 5.2883142e-11,
 5.209938e-11,
 4.5385612e-11,
 5.33164e-11,
 8.941601e-12,
 2.0179375e-07,
 3.7532932e-12,
 2.1310401e-11,
 2.975406e-11]

#calculate the average loss for each layer
layer_2_loss_list = np.array(layer_2_loss_list)
layer_3_loss_list = np.array(layer_3_loss_list)
layer_4_loss_list = np.array(layer_4_loss_list)
layer_5_loss_list = np.array(layer_5_loss_list)
layer_6_loss_list = np.array(layer_6_loss_list)
layer_7_loss_list = np.array(layer_7_loss_list)


layer_list = [2, 3, 4, 5, 6, 7]
mean_loss_vs_layer = [layer_2_loss_list.mean(), layer_3_loss_list.mean(), layer_4_loss_list.mean(), layer_5_loss_list.mean(), layer_6_loss_list.mean(), layer_7_loss_list.mean()]

# Apply matplotlib parameters
plt.rcParams.update(params)

# Create bar chart
fig, ax = plt.subplots(figsize=(4, 3))

# Plot all bars without outline
bars = ax.bar(layer_list, mean_loss_vs_layer, color=bright_palette[2], edgecolor='none', zorder=2)

# Add reference dashed line at 10^-10
ax.axhline(y=1e-10, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)

# Add text label on the dashed line
ax.text(2.0, 1e-10, 'early stopping', fontsize=12, color='gray', 
        verticalalignment='bottom', horizontalalignment='left',
        )

# Add line shadow effect to layers 5 and 6
#shadow_effect = patheffects.Shadow(offset=(2, -2), shadow_rgbFace='black', alpha=0.3)

#for i, layer in enumerate(layer_list):
#    if layer in [5, 6]:
#        # Apply shadow effect to the bar
#        bars[i].set_path_effects([shadow_effect])

ax.set_xlabel('Number of Layers')
ax.set_ylabel('Mean Loss')
ax.set_yscale('log')
#ax.set_yticks([0,5e-11,1e-10, 1.5e-10])
ax.set_xticks(layer_list)
plt.tight_layout()
plt.savefig('mean_loss_vs_layer_lorenz96.png')
plt.show()