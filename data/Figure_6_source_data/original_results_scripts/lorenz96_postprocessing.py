import pandas as pd
import ast
import matplotlib.pyplot as plt
import numpy as np

FS_SMALL = 10
FS_MEDIUM = 12
FS_LARGE = 14

#read matrix from train_data.mat
import scipy.io
from lorenz96_ode import *

import numpy as np



from scipy.integrate import solve_ivp

train_data = scipy.io.loadmat('train_data.mat')
val_data = scipy.io.loadmat('validation_data.mat')

#extract the data from the fourth sheet
train_data = train_data['data']
val_data = val_data['data']

one_traj_length = 10000 #in total 6 trajectories, each has 10000 points. We only plot the first one
t_arr =train_data[:one_traj_length,0]
#access data using column number
dim = 100

#training data
x_train = train_data[:one_traj_length,1:dim+1]
dx_train = train_data[:one_traj_length,dim+1:2*dim+1]

pred_dx = train_data[:one_traj_length,2*dim+1:3*dim+1]

#validation data
val_t_arr = val_data[1:,0]
val_x_train = val_data[1:,1:dim+1] #1
val_dx_train = val_data[1:,dim+1:2*dim+1]

val_pred_dx = val_data[1:,2*dim+1:3*dim+1]


# Define the initial state (data from nn_configurations.xlsx)
initial_state = x_train[0,:]
# Define the time span for the simulation
t_span = (0, 20)
t_eval = np.arange(0, 20, 0.001)

# Solve the ODE system

solution = solve_ivp(lorenz96_ode, t_span, initial_state, t_eval=t_eval)


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
matplotlib.rcParams.update(params)

import cmocean

#plot heatmap and time trajectory of dim 1
x_arr = np.concatenate((x_train,val_x_train),axis=0)

dx_arr = np.concatenate((dx_train,val_dx_train),axis=0)
pred_dx_arr = np.concatenate((pred_dx,val_pred_dx),axis=0)
pred_x_arr = solution.y.T

t_arr = np.concatenate((t_arr,val_t_arr),axis=0)

sample_per_points = 1
heatmap_t_arr = t_arr[:-1:sample_per_points]
heat_map_dx = dx_arr[:t_arr.shape[0],:]
heat_map_dx=heat_map_dx[:-1:sample_per_points,:]

heat_map_x = x_arr[:t_arr.shape[0],:]
heat_map_x=heat_map_x[:-1:sample_per_points,:]

heat_map_pred_dx = pred_dx_arr[:-1:sample_per_points,:]
heat_map_pred_x = pred_x_arr[:-1:sample_per_points,:]

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
#plt.rcParams['text.usetex'] = True  # Uncomment if you have LaTeX installed
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter

ax1 = fig.add_subplot(gs[0, 0])
cax1 = ax1.imshow(heat_map_dx, aspect='auto', cmap='cmo.delta')

step = 400
ax1.set_xticks([])
ax1.set_yticks([0, 100])
ax1.set_ylabel(r'Index, $i$', labelpad=-10)
#ax1.set_ylabel(r'State Index, $i$')
ax1.tick_params(axis='both', direction='in')

#add a vertical dash line at the middle of x
x_mid = (plt.xlim()[0] + plt.xlim()[1]) / 2
#ax1.axvline(x=x_mid, color='k', linestyle='--')
#ax1.set_title("Full Simulation")

title_ax1 = fig.add_subplot(gs[0, 1])

title_ax1.set_title(r'$\dot{x}_i$', loc="center", fontsize=FS_SMALL, y=0.4)
title_ax1.axis('off')  # Turn off the axis for the title


#### 3 and 4
ax2 = fig.add_subplot(gs[1, 0])
cax2 = ax2.imshow(heat_map_pred_dx, aspect='auto', cmap='cmo.delta')
ax2.tick_params(axis='both', direction='in')

#ax1.set_xticks(ticks=np.arange(0,heat_map_x.shape[1],step), labels=heatmap_t_arr[::step])
ax2.set_xticks([])
ax2.set_yticks([0, 100])
ax2.set_ylabel(r'Index, $i$', labelpad=-10)



title_ax2 = fig.add_subplot(gs[1, 1])

title_ax2.set_title(r'$\dot{\hat{x}}_i$', loc="center", fontsize=FS_SMALL, y=0.4)
title_ax2.axis('off')  # Turn off the axis for the title

#ax2.set_ylabel(r'State Index, $i$')
#add a vertical dash line at the middle of x
x_mid = (plt.xlim()[0] + plt.xlim()[1]) / 2
#ax2.axvline(x=x_mid, color='k', linestyle='--')
#ax2.set_title("Identified System")

#error
from mpl_toolkits.axes_grid1 import make_axes_locatable

ax3 = fig.add_subplot(gs[2, 0])
cax3 = ax3.imshow(np.abs(heat_map_dx-heat_map_pred_dx), extent=[0, 20, 0, 100], aspect='auto', cmap='cmo.delta')


# Add colorbar in a new column (next to ax3)
cbar_ax = fig.add_subplot(gs[-1, 1])  # Span the entire height (all rows) for the colorbar
cbar_1 = fig.colorbar(cax3, cax=cbar_ax, orientation='vertical')
# Scale the tick labels explicitly and place the multiplier above the colorbar.
cbar_1.formatter = FuncFormatter(lambda value, _: f'{value / 1e-5:g}')
cbar_1.update_ticks()
cbar_1.ax.set_title(r'$\times 10^{-5}$', fontsize=FS_SMALL, pad=2)

cbar_1.ax.tick_params(direction='in', labelleft=False, labelright=True)  # Labels on the right

ax3.tick_params(axis='both', direction='in')
ax3.set_yticks(ticks=[0, 100])
ax3.set_xticks(ticks=[0,10,20])
#ax3.set_ylabel(r'State Index, $i$')
ax3.set_xlabel(r'Time, $t$', labelpad=0)
ax3.set_ylabel(r'Index, $i$', labelpad=-10)


#plt.tight_layout()
#plt.show()
plt.subplots_adjust(left=0.23, right=0.78)
plt.savefig("Lorenz_96_ground_truth_vs_identified", dpi=600)



###############################
################################
##############################
#time history of some dimensions
"""
fig3 = plt.figure(figsize=(8, 6))


gs = fig3.add_gridspec(4, 2)

half_point = 10000

ax11 = fig3.add_subplot(gs[0:2, 0])
ax11.plot(t_arr[:half_point], x_arr[:half_point,19], 'k', label='True')
ax11.plot(t_arr[:half_point], pred_x_arr[:half_point,19], 'r--', label='pred_dxicted')
ax11.set_ylabel(r'$x_{20}$')

ax12 = fig3.add_subplot(gs[0:2, 1])
ax12.plot(t_arr[:half_point], x_arr[:half_point,39], 'k', label='True')
ax12.plot(t_arr[:half_point], pred_x_arr[:half_point,39], 'r--', label='pred_dxicted')
ax12.set_ylabel(r'$x_{40}$')

ax13 = fig3.add_subplot(gs[2:4, 0])
ax13.plot(t_arr[:half_point], x_arr[:half_point,59], 'k', label='True')
ax13.plot(t_arr[:half_point], pred_x_arr[:half_point,59], 'r--', label='pred_dxicted')
ax13.set_xlabel(r'Time, $t$')
ax13.set_ylabel(r'$x_{60}$')

ax14 = fig3.add_subplot(gs[2:4, 1])
ax14.plot(t_arr[:half_point], x_arr[:half_point,79], 'k', label='True')
ax14.plot(t_arr[:half_point], pred_x_arr[:half_point,79], 'r--', label='pred_dxicted')
ax14.set_xlabel(r'Time, $t$')
ax14.set_ylabel(r'$x_{80}$')
ax14.set_yticks([-5,0, 5, 10])

plt.tight_layout()
plt.savefig("Lorenz_96_time_history", dpi=600)
"""
