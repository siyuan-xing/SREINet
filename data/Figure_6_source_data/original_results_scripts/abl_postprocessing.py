from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import scipy.io


SCRIPT_DIR = Path(__file__).resolve().parent
FS_SMALL = 10
FS_MEDIUM = 12
FS_LARGE = 14

# Specify the model type

train_data = scipy.io.loadmat(SCRIPT_DIR / 'train_data.mat')

#extract the data from the fourth sheet
train_data = train_data['data']

dim = 128


T = 500.0
dt = 0.01
N = int(T/dt)


#training data
dx_train = train_data[:N,1+dim:2*dim+1]

pred_dx = train_data[:N,1+2*dim:3*dim+1]

params = {
    'image.origin': 'lower',
    'image.interpolation': 'nearest',
    'axes.grid': False,
    'savefig.dpi': 600,  # to adjust notebook inline plot size
    'axes.labelsize': FS_MEDIUM,
    'axes.titlesize': FS_MEDIUM,
    'axes.titleweight': 'bold',
    'axes.labelweight': 'normal',
    'font.size': FS_SMALL,
    'legend.fontsize': FS_SMALL,
    'xtick.labelsize': FS_SMALL,
    'ytick.labelsize': FS_SMALL,
    'text.usetex': False,
    #'figure.figsize': [7, 4],
    'font.family': 'Helvetica',
}

import matplotlib

import cmocean
matplotlib.rcParams.update(params)

#plot heatmap and time trajectory of dim 1
node_num = 64
dx_dt_arr = dx_train
pred_dx_dt_arr = pred_dx

#heat_map_x = np.sqrt(x_arr[:,:node_num]**2 + x_arr[:,node_num:2*node_num]**2)
heat_map_dx = np.sqrt(dx_dt_arr[:,:node_num]**2 + dx_dt_arr[:,node_num:2*node_num]**2)


heat_map_pred_dx = np.sqrt(pred_dx_dt_arr[:,:node_num]**2 + pred_dx_dt_arr[:,node_num:2*node_num]**2)
#heat_map_pred_x = pred_x_arr[:,:node_num]

#heat_map_x = heat_map_x.T
heat_map_dx = heat_map_dx.T
heat_map_pred_dx = heat_map_pred_dx.T
#heat_map_pred_x = heat_map_pred_x.T

#print(heat_map_x.shape)
print(heat_map_dx.shape)
print(heat_map_pred_dx.shape)
#print(heat_map_pred_x.shape)

fig = plt.figure(figsize=(7.0 / 4.0, 4.08))
fig.patch.set_facecolor((0.95, 0.95, 0.95, 0.1))  # RGBA format where A=0.5 for transparency

gs = fig.add_gridspec(3, 2, hspace=0.28, width_ratios=[20, 1])

#ax2 line thickness set to 2
#plt.rcParams['text.usetex'] = True  # Uncomment if you have LaTeX installed
#plt.rcParams['font.family'] = 'Helvetica'
from matplotlib.ticker import FuncFormatter, MaxNLocator


ax1 = fig.add_subplot(gs[0, 0])
cax1 = ax1.imshow(heat_map_dx, aspect='auto', cmap='cmo.delta', extent=[0, 500, -32, 32])

ax1.set_xticks([])
ax1.set_yticks([-32, 32])
ax1.tick_params(axis='both', direction='in')

#ax1.set_title("Full Simulation")

title_ax1 = fig.add_subplot(gs[0, 1])

title_ax1.set_title(r'$|\dot{u}|_i$', loc="center", fontsize=FS_SMALL, y=0.4)
title_ax1.axis('off')  # Turn off the axis for the title


#### 3 and 4
ax2 = fig.add_subplot(gs[1, 0])
cax2 = ax2.imshow(heat_map_pred_dx, aspect='auto', extent=[0, 500, -32, 32], cmap='cmo.delta')
ax2.tick_params(axis='both', direction='in')

#ax1.set_xticks(ticks=np.arange(0,heat_map_x.shape[1],step), labels=heatmap_t_arr[::step])
ax2.set_xticks([])
ax2.set_yticks([-32, 32])



title_ax2 = fig.add_subplot(gs[1, 1])

title_ax2.set_title(r'$|\dot{\hat{u}}|_i$', loc="center", fontsize=FS_SMALL, y=0.4)
title_ax2.axis('off')  # Turn off the axis for the title

#ax2.set_ylabel(r'State Index, $i$')
#ax2.set_title("Identified System")

#error
ax3 = fig.add_subplot(gs[2, 0])
error_map = np.abs(heat_map_dx-heat_map_pred_dx)
cax3 = ax3.imshow(error_map, extent=[0, 500, -32, 32], aspect='auto', cmap='cmo.delta')


# Add colorbar in a new column (next to ax3)
cbar_ax = fig.add_subplot(gs[-1, 1])  # Span the entire height (all rows) for the colorbar
cbar_1 = fig.colorbar(cax3, cax=cbar_ax, orientation='vertical')
error_max = np.nanmax(error_map)
error_exponent = int(np.floor(np.log10(error_max))) if error_max > 0 else 0
error_scale = 10.0 ** error_exponent
cbar_1.locator = MaxNLocator(nbins=5, steps=[1, 2, 5, 10])
cbar_1.formatter = FuncFormatter(lambda value, _: f'{value / error_scale:g}')
cbar_1.update_ticks()
cbar_1.ax.set_title(rf'$\times 10^{{{error_exponent}}}$', fontsize=FS_SMALL, pad=2)

cbar_1.ax.tick_params(direction='in', labelleft=False, labelright=True)  # Labels on the right

ax3.tick_params(axis='both', direction='in')
ax3.set_yticks(ticks=[-32, 32])
ax3.set_xticks(ticks=[0, 250, 500])
ax3.set_xlabel(r'Time, $t$', labelpad=0)


plt.subplots_adjust(left=0.18, right=0.78)
#plt.show()
output_path = SCRIPT_DIR / "AL_64_node_results.png"
plt.savefig(output_path, dpi=600)
plt.close(fig)
print(f"Saved figure to {output_path}")
