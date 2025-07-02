import pandas as pd
import ast
import matplotlib.pyplot as plt
import numpy as np
from kuramoto_ode import kuramoto_ode
from polar_plot import plot_kuramoto_phases
#read matrix from train_data.mat
import scipy.io


import numpy as np

from scipy.integrate import solve_ivp

train_data = scipy.io.loadmat('train_data.mat')
val_data = scipy.io.loadmat('validation_data.mat')

#extract the data from the fourth sheet
train_data = train_data['data']
val_data = val_data['data']

one_traj_length = 20000 #in total 6 trajectories, each has 10000 points. We only plot the first one
t_arr =train_data[:one_traj_length,0]
#access data using column number
dim = 60

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
t_span = (0, 200)
t_eval = np.arange(0, 200, 0.1)

sample_points=len(t_eval)

# Solve the ODE system
solution = solve_ivp(kuramoto_ode, t_span, initial_state, t_eval=t_eval,method='BDF')


params = {
    'image.origin': 'lower',
    'image.interpolation': 'nearest',
    'axes.grid': False,
    'savefig.dpi': 300,  # to adjust notebook inline plot size
    'axes.labelsize': 14, # fontsize for x and y labels (was 10)
    'axes.titlesize': 16,
    'font.size': 14, # was 10
    'legend.fontsize': 14, # was 10
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'text.usetex': True,
    #'figure.figsize': [7, 3],
    'font.family': 'Helvetica',
    'text.latex.preamble': r'\usepackage{amsmath}',  # Optional, for advanced math symbols
    'axes.labelweight': 'bold'}
import matplotlib
matplotlib.rcParams.update(params)

#plot heatmap and time trajectory of dim 1
#x_arr = np.concatenate((x_train,val_x_train),axis=0)
#dx_arr = np.concatenate((dx_train,val_dx_train),axis=0)
#pred_dx_arr = np.concatenate((pred_dx,val_pred_dx),axis=0)
#t_arr = np.concatenate((t_arr,val_t_arr),axis=0)

x_arr = x_train[:sample_points,:]
dx_arr = dx_train[:sample_points,:]
pred_dx_arr = pred_dx[:sample_points,:]
pred_x_arr = solution.y.T


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

fig = plt.figure(figsize=(12, 4))

gs = fig.add_gridspec(4, 12)

#ax2 line thickness set to 2
plt.rcParams['text.usetex'] = True  # Uncomment if you have LaTeX installed
plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'  # Optional, for advanced math symbols
plt.rcParams['axes.labelweight'] = 'bold'

import matplotlib.colors as mcolors

ax1 = fig.add_subplot(gs[:, 2:6])
cax1 = ax1.imshow(heat_map_dx, aspect='auto', extent=[0,200,0, 60], cmap='coolwarm')

step = 400
#ax1.set_xticks([])
ax1.set_yticks([0,20,40,60])

ax1.set_ylabel(r'State Index, $i$')
ax1.set_xlabel(r'Time, $t$')
#add a vertical dash line at the middle of x
x_mid = (plt.xlim()[0] + plt.xlim()[1]) / 2
ax1.set_title("Full Simulation")

#### 3 and 4
ax2 = fig.add_subplot(gs[:, 6:10])
cax2 = ax2.imshow(heat_map_pred_dx, aspect='auto',  extent=[0,200,0, 60], cmap='coolwarm')

#ax1.set_xticks(ticks=np.arange(0,heat_map_x.shape[1],step), labels=heatmap_t_arr[::step])
#ax2.set_yticks([0,20,40,60])
ax2.set_xlabel(r'Time, $t$')

#ax3.set_ylabel('State Index')
#add a vertical dash line at the middle of x
x_mid = (plt.xlim()[0] + plt.xlim()[1]) / 2
ax2.set_title("Identified System")

#add colorbar



ax3 = fig.add_subplot(gs[0:2, 0:2], projection='polar') 
plot_kuramoto_phases(pred_x_arr[0,0::10], 
                     pred_dx_arr[0,0::10], 
                     x_arr[0,0::10], 
                     dx_arr[0,0::10],
                     ax3,
                     "t=0")

ax4 = fig.add_subplot(gs[2:4, 0:2], projection='polar')

t1=200
plot_kuramoto_phases(pred_x_arr[t1,0::10], 
                     pred_dx_arr[t1,0::10], 
                     x_arr[t1,0::10], 
                     dx_arr[t1,0::10],
                     ax4,
                     "t=20")


ax5 = fig.add_subplot(gs[0:2, 10:12], projection='polar')
t2=400
plot_kuramoto_phases(pred_x_arr[t2,0::10],
                        pred_dx_arr[t2,0::10], 
                        x_arr[t2,0::10], 
                        dx_arr[t2,0::10],
                        ax5,
                        "t=40")

ax6 = fig.add_subplot(gs[2:4, 10:12], projection='polar')

t4=1000
plot_kuramoto_phases(pred_x_arr[t4,0::10], 
                     pred_dx_arr[t4,0::10], 
                     x_arr[t4,0::10], 
                     dx_arr[t4,0::10],
                     ax6,
                     "t=100")



plt.tight_layout()
plt.savefig("Kuramoto_60D_simulation_vs_infer", dpi=300)




