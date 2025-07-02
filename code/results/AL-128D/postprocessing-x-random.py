import pandas as pd
import ast
import matplotlib.pyplot as plt
import numpy as np

#read matrix from train_data.mat
import scipy.io
import numpy as np

from scipy.integrate import solve_ivp

from AL_ode import AL_ode

import DataGenerator as DG

# Specify the model type

train_data = scipy.io.loadmat('train_data.mat')
val_data = scipy.io.loadmat('validation_data.mat')

#extract the data from the fourth sheet
train_data = train_data['data']
val_data = val_data['data']

dim = 128


T = 500.0
dt = 0.01
t_span = (0, T)
N = int(T/dt)


#one_traj_length = 5000 #in total 6 trajectories, each has 10000 points. We only plot the first one
t_arr = train_data[:N,0]
#access data using column number

#training data
x_train = train_data[:N,1:dim+1]
dx_train = train_data[:N,1+dim:2*dim+1]

pred_dx = train_data[:N,1+2*dim:3*dim+1]

#validation data
##val_t_arr = val_data[1:,0]
#val_x_train = val_data[1:,1:dim+1] #1
#val_dx_train = val_data[1:,dim+1:2*dim+1]

#val_pred_dx = val_data[1:,2*dim+1:3*dim+1]

# Define the initial state (data from nn_configurations.xlsx)
initial_state = x_train[0,:]
# Define the time span for the simulation

t_eval = np.arange(0, T, dt)

# Solve the ODE system
solution = solve_ivp(AL_ode, t_span, initial_state, t_eval=t_eval, method='BDF')

import matplotlib

params = {
    'image.origin': 'lower',
    'image.interpolation': 'nearest',
    'image.cmap': 'gray',
    'axes.grid': False,
    'savefig.dpi': 600,  # to adjust notebook inline plot size
    'axes.labelsize': 14, # fontsize for x and y labels (was 10)
    'axes.titlesize': 14,
    'font.size': 14, # was 10
    'legend.fontsize': 14, # was 10
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'text.usetex': False,
    #'figure.figsize': [7, 4],
    'font.family': 'Helvetica',
    #'text.latex.preamble': r'\usepackage{amsmath}',  # Optional, for advanced math symbols
    #'axes.labelweight': 'bold'
}

import cmocean
matplotlib.rcParams.update(params)

#plot heatmap and time trajectory of dim 1
node_num = 64
x_arr = x_train

dx_dt_arr = dx_train
pred_x_arr = solution.y.T
pred_dx_dt_arr = pred_dx

sample_per_points = 1
heatmap_t_arr = t_arr

heat_map_x = np.sqrt(x_arr[:,:node_num]**2 + x_arr[:,node_num:2*node_num]**2)
heat_map_dx = np.sqrt(dx_dt_arr[:,:node_num]**2 + dx_dt_arr[:,node_num:2*node_num]**2)


heat_map_pred_dx = np.sqrt(pred_dx_dt_arr[:,:node_num]**2 + pred_dx_dt_arr[:,node_num:2*node_num]**2)
heat_map_pred_x = np.sqrt(pred_x_arr[:,:node_num]**2 + pred_x_arr[:,node_num:2*node_num]**2)

heat_map_x = heat_map_x.T
heat_map_dx = heat_map_dx.T
heat_map_pred_dx = heat_map_pred_dx.T
heat_map_pred_x = heat_map_pred_x.T

print(heat_map_x.shape)
print(heat_map_dx.shape)
print(heat_map_pred_dx.shape)
print(heat_map_pred_x.shape)


#ax2 line thickness set to 2
plt.rcParams['text.usetex'] = False  # Uncomment if you have LaTeX installed
plt.rcParams['font.family'] = 'Helvetica'
#plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'  # Optional, for advanced math symbols
#plt.rcParams['axes.labelweight'] = 'bold'

import matplotlib.colors as mcolors

#error
from mpl_toolkits.axes_grid1 import make_axes_locatable


fig = plt.figure(figsize=(8, 4))

#ax3 = fig.add_subplot(gs[2, 0])
cax3 = plt.imshow(np.abs(heat_map_x-heat_map_pred_x), extent=[0, 500, -32, 32], aspect='auto', cmap='cmo.delta')


# Add colorbar in a new column (next to ax3)
#cbar_ax = fig.add_subplot(gs[-1, 1])  # Span the entire height (all rows) for the colorbar

#pos = cbar_ax.get_position()  # Get the original position
#pos = [pos.x0-0.03, pos.y0, pos.width, pos.height]  # Shift the colorbar to the left by 0.05
#cbar_ax.set_position(pos)  # Apply the new position


cbar_1 = fig.colorbar(cax3, orientation='vertical')
cbar_1.formatter.set_powerlimits((0, 0))  # Set threshold for scientific notation
#cbar_1.set_ticks([0, 2e-4, 4e-4, 6e-4, 8e-4, 1e-3])
cbar_1.update_ticks()  # Update the colorbar

cbar_1.ax.tick_params(direction='in', labelleft=False, labelright=True)  # Labels on the right

plt.tick_params(axis='y')
plt.tick_params(axis='both', direction='in')
plt.yticks(ticks=[-32, -16,0, 16, 32])
plt.xticks(ticks=[0, 250, 500])
#plt.xlim(0, 50)
#ax3.set_ylabel(r'State Index, $i$')
plt.xlabel(r'Time, $t$')
plt.ylabel(r'State Index, $j$', labelpad=-7)


plt.tight_layout()
#plt.show()
plt.savefig("AL_64_node_error_random.png", dpi=1200)



params = {
    'image.origin': 'lower',
    'image.interpolation': 'nearest',
    'image.cmap': 'gray',
    'axes.grid': False,
    'savefig.dpi': 600,  # to adjust notebook inline plot size
    'axes.labelsize': 14, # fontsize for x and y labels (was 10)
    'axes.titlesize': 14,
    'font.size': 14, # was 10
    'legend.fontsize': 14, # was 10
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'text.usetex': False,
    #'figure.figsize': [7, 4],
    'font.family': 'Arial',
    #'text.latex.preamble': r'\usepackage{amsmath}',  # Optional, for advanced math symbols
    #'axes.labelweight': 'bold'
}
matplotlib.rcParams.update(params)

fig3 = plt.figure(figsize=(8, 6))


gs = fig3.add_gridspec(4, 2)

points_to_plot = 20000

pred_amp = np.abs(pred_x_arr[:,:node_num] + 1j*pred_x_arr[:,node_num:])
sim_amp = np.abs(x_train[:,:node_num] + 1j*x_train[:,node_num:])
sim_amp = sim_amp.T
pred_amp = pred_amp.T

ax11 = fig3.add_subplot(gs[0:2, 0])
ax11.plot(t_arr[:points_to_plot], sim_amp[2,:points_to_plot], 'k', label='True')
ax11.plot(t_arr[:points_to_plot], pred_amp[2,:points_to_plot], 'r--', label='pred_dxicted')
ax11.set_ylabel(r'$| u_{-30} |$')
#ax11.set_xticks([0, 125, 250, 375, 500])
#reference_line = 60 
#ax11.axvline(x=reference_line, color='k', linestyle='--')


ax12 = fig3.add_subplot(gs[0:2, 1])
ax12.plot(t_arr[:points_to_plot], sim_amp[17,:points_to_plot], 'k', label='True')
ax12.plot(t_arr[:points_to_plot], pred_amp[17,:points_to_plot], 'r--', label='pred_dxicted')
ax12.set_ylabel(r'$| u_{-15} |$')
#ax12.set_xticks([0, 125, 250, 375, 500])
#ax12.axvline(x=reference_line, color='k', linestyle='--')


ax13 = fig3.add_subplot(gs[2:4, 0])
ax13.plot(t_arr[:points_to_plot], sim_amp[32,:points_to_plot], 'k', label='True')
ax13.plot(t_arr[:points_to_plot], pred_amp[32,:points_to_plot], 'r--', label='pred_dxicted')
ax13.set_xlabel(r'Time, $t$')
ax13.set_ylabel(r'$| u_{0} |$')
#ax13.set_xticks([0, 125, 250, 375, 500])
#ax13.axvline(x=reference_line, color='k', linestyle='--')


ax14 = fig3.add_subplot(gs[2:4, 1])
ax14.plot(t_arr[:points_to_plot], sim_amp[47,:points_to_plot], 'k', label='True')
ax14.plot(t_arr[:points_to_plot], pred_amp[47,:points_to_plot], 'r--', label='pred_dxicted')
ax14.set_xlabel(r'Time, $t$')
ax14.set_ylabel(r'$| u_{15} |$')
#ax14.set_yticks([-5,0, 5, 10])
#ax14.set_xticks([0, 125, 250, 375, 500])
#ax14.axvline(x=reference_line, color='k', linestyle='--')

plt.tight_layout()
plt.savefig("AL_64_node_results_x_random.png", dpi=1200)

