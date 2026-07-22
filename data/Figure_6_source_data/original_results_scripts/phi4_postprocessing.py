import pandas as pd
import ast
import matplotlib.pyplot as plt
import numpy as np

FS_SMALL = 10
FS_MEDIUM = 12
FS_LARGE = 14

#read matrix from train_data.mat
import scipy.io
import numpy as np

from scipy.integrate import solve_ivp

from phi_ode import phi4_ode

import DataGenerator as DG

# Specify the model type
MY_MODEL = 'discrete_phi_quartic' # discrete sine Gordon model 

train_data = scipy.io.loadmat('train_data.mat')
val_data = scipy.io.loadmat('validation_data.mat')

#extract the data from the fourth sheet
train_data = train_data['data']
val_data = val_data['data']

dim = 100

T = 500.0
dt = 0.05
t_span = (0, T)
N = int(T/dt)

#one_traj_length = 5000 #in total 6 trajectories, each has 10000 points. We only plot the first one
t_arr =train_data[:N,0]
#access data using column number

#training data
x_train = train_data[:N,1:dim+1]
dx_train = train_data[:N,dim+1:2*dim+1]

pred_dx = train_data[:N,2*dim+1:3*dim+1]

#validation data
val_t_arr = val_data[1:,0]
val_x_train = val_data[1:,1:dim+1] #1
val_dx_train = val_data[1:,dim+1:2*dim+1]

val_pred_dx = val_data[1:,2*dim+1:3*dim+1]

# Define the initial state (data from nn_configurations.xlsx)
initial_state = x_train[0,:]
# Define the time span for the simulation

t_eval = np.arange(0, T, dt)

# Solve the ODE system
solution = solve_ivp(phi4_ode, t_span, initial_state, t_eval=t_eval)


params = {
    'image.origin': 'lower',
    'image.interpolation': 'nearest',
    'image.cmap': 'gray',
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
node_num = 50
x_arr = x_train

dx_dt_arr = dx_train
pred_x_arr = solution.y.T
pred_dx_dt_arr = pred_dx

sample_per_points = 1
heatmap_t_arr = t_arr

heat_map_x = x_arr[:,:node_num]
heat_map_dx = dx_dt_arr[:,:node_num]


heat_map_pred_dx = pred_dx_dt_arr[:,:node_num]
heat_map_pred_x = pred_x_arr[:,:node_num]

heat_map_x = heat_map_x.T
heat_map_dx = heat_map_dx.T
heat_map_pred_dx = heat_map_pred_dx.T
heat_map_pred_x = heat_map_pred_x.T

print(heat_map_x.shape)
print(heat_map_dx.shape)
print(heat_map_pred_dx.shape)
print(heat_map_pred_x.shape)

fig = plt.figure(figsize=(7.0 / 4.0, 4.08))
fig.patch.set_facecolor((0.95, 0.95, 0.95, 0.1))  # RGBA format where A=0.5 for transparency

gs = fig.add_gridspec(3, 2, hspace=0.28, width_ratios=[20, 1])

#ax2 line thickness set to 2
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter, MaxNLocator

ax1 = fig.add_subplot(gs[0, 0])
cax1 = ax1.imshow(heat_map_dx, aspect='auto', extent=[0, 500, -25, 25], cmap='cmo.delta')

step = 400
ax1.set_xticks([])
ax1.set_yticks([-25, 25])
ax1.tick_params(axis='both', direction='in')

#add a vertical dash line at the middle of x
x_mid = (plt.xlim()[0] + plt.xlim()[1]) / 2
#ax1.axvline(x=x_mid, color='k', linestyle='--')
#ax1.set_title("Full Simulation")

title_ax1 = fig.add_subplot(gs[0, 1])

title_ax1.set_title(r'$\dot{u}_i$', loc="center", fontsize=FS_SMALL, y=0.4)
title_ax1.axis('off')  # Turn off the axis for the title


#### 3 and 4
ax2 = fig.add_subplot(gs[1, 0])
cax2 = ax2.imshow(heat_map_pred_dx, aspect='auto', extent=[0, 500, -25, 25], cmap='cmo.delta')
ax2.tick_params(axis='both', direction='in')

#ax1.set_xticks(ticks=np.arange(0,heat_map_x.shape[1],step), labels=heatmap_t_arr[::step])
ax2.set_xticks([])
ax2.set_yticks([-25, 25])
#ax2.set_ylabel(r'Index, $i$', labelpad=-7)



title_ax2 = fig.add_subplot(gs[1, 1])

title_ax2.set_title(r'$\dot{\hat{u}}_i$', loc="center", fontsize=FS_SMALL, y=0.4)
title_ax2.axis('off')  # Turn off the axis for the title

#ax2.set_ylabel(r'State Index, $i$')
#add a vertical dash line at the middle of x
x_mid = (plt.xlim()[0] + plt.xlim()[1]) / 2
#ax2.axvline(x=x_mid, color='k', linestyle='--')
#ax2.set_title("Identified System")

#error
from mpl_toolkits.axes_grid1 import make_axes_locatable

ax3 = fig.add_subplot(gs[2, 0])
error_map = np.abs(heat_map_dx-heat_map_pred_dx)
cax3 = ax3.imshow(error_map, extent=[0, 500, -25, 25], aspect='auto', cmap='cmo.delta')


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
ax3.set_yticks(ticks=[-25, 25])
ax3.set_xticks(ticks=[0,250,500])
ax3.set_xlabel(r'Time, $t$', labelpad=0)


plt.subplots_adjust(left=0.18, right=0.78)
#plt.show()
plt.savefig("phi4_50_node_results.png", dpi=600)
