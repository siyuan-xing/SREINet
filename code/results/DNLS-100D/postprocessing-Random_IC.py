import pandas as pd
import ast
import matplotlib.pyplot as plt
import numpy as np

#read matrix from train_data.mat
import scipy.io
import numpy as np

from scipy.integrate import solve_ivp

from DNLS_ode import DNLS_ode

import DataGenerator as DG


MY_MODEL = 'dnls'  

# Generate data based on the specified model type
node_num = 50
C=2.0
train_data = scipy.io.loadmat('train_data.mat')
val_data = scipy.io.loadmat('validation_data.mat')

#extract the data from the fourth sheet
train_data = train_data['data']
val_data = val_data['data']

dim = 100

T = 100.0
dt = 0.01
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
#validation data are complex numbers

val_t_arr = val_data[1:,0]
val_x_train = val_data[1:,1:node_num+1] #1
val_x_train = np.concatenate((val_x_train.real, val_x_train.imag), axis=1)
val_dx_train = val_data[1:,node_num+1:2*node_num+1]
val_dx_train = np.concatenate((val_dx_train.real, val_dx_train.imag), axis=1)
#val_pred_dx = val_data[1:,2*node_num+1:3*node_num+1]
#val_pred_dx = np.concatenate((val_pred_dx.real, val_pred_dx.imag), axis=1)

# Define the initial state (data from nn_configurations.xlsx)
initial_state = x_train[0,:]
# Define the time span for the simulation
#x_data =

x_data = np.concatenate((x_train, val_x_train), axis=0)
dx_data = np.concatenate((dx_train,val_dx_train), axis=0)

sim_init = initial_state

# Define the time span for the simulation
t_span = (0, 2*T)
t_eval = np.arange(0, 2*T, dt)

# Solve the ODE system
solution = solve_ivp(DNLS_ode, t_span, sim_init, t_eval=t_eval)

sol = solution.y.T

pred_amp = np.abs(sol[:,:node_num] + 1j*sol[:,node_num:])
sim_amp = np.abs(x_data[:,:node_num] + 1j*x_data[:,node_num:])

t_arr = t_eval

params = {
    'image.origin': 'lower',
    'image.interpolation': 'nearest',
    'image.cmap': 'gray',
    'axes.grid': False,
    'savefig.dpi': 600,  # to adjust notebook inline plot size
    'axes.labelsize': 10, # fontsize for x and y labels (was 10)
    'axes.titlesize': 12,
    'font.size': 10, # was 10
    'legend.fontsize': 10, # was 10
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'text.usetex': False,
    #'figure.figsize': [7, 4],
    'font.family': 'Helvetica',
    #'text.latex.preamble': r'\usepackage{amsmath}',  # Optional, for advanced math symbols
    #'axes.labelweight': 'bold'
}
import matplotlib

matplotlib.rcParams.update(params)


pred_amp = pred_amp.T
sim_amp = sim_amp.T
print(pred_amp.shape)
print(sim_amp.shape)

my_size = 110/25.4 #110mm

fig = plt.figure(figsize=(my_size*2, my_size/2),facecolor='none')

gs = fig.add_gridspec(5, 3, width_ratios=[20, 20, 1], hspace=0.1, wspace=0.1)

import matplotlib.colors as mcolors
import cmocean

ax5 = fig.add_subplot(gs[0:-1, 0])
ax5.set_position([ax5.get_position().x0, ax5.get_position().y0 + 0.02, ax5.get_position().width, ax5.get_position().height])

cax5 = ax5.imshow(sim_amp, aspect='auto', cmap='cmo.tempo', extent=[0, 2*T, 0, node_num])
ax5.set_xlim(0,160)
ax5.set_xticks([])
ax5.set_ylim(0,node_num)
ax5.set_yticks([0,25, 50])
ax5.set_yticklabels([-25, 0, 25])

ax5.set_ylabel(r'State Index, $j$')
#x_mid = (plt.xlim()[0] + plt.xlim()[1]) / 2
#ax5.axvline(x=x_mid, color='k', linestyle='--')

ax6 = fig.add_subplot(gs[-1, 0])
ax6.set_position([ax6.get_position().x0, ax6.get_position().y0 + 0.02, ax6.get_position().width, ax6.get_position().height])
ax6.plot(t_arr, sim_amp[node_num//2-1,:],'k', linewidth=2.0)
ax6.set_xlim(0,160)
ax6.set_ylim(0,1.5)
ax6.set_xticks([0, 40, 80, 120, 160])
ax6.set_ylabel(r'$|u_{0}|$',rotation=0)
ax6.yaxis.set_label_coords(-0.15, 0.3)
##ax6.set_xlabel(r'Time, $t$')
#ax6.set_yticks([-5, 5])
#reference_line = 25 
#ax6.axvline(x=reference_line, color='k', linestyle='--')


#### 3 and 4
ax7 = fig.add_subplot(gs[0:-1, 1])
ax7.set_position([ax7.get_position().x0, ax7.get_position().y0 + 0.02, ax7.get_position().width, ax7.get_position().height])

cax7 = ax7.imshow(pred_amp, aspect='auto', extent=[0, 2*T, 0, node_num], cmap='cmo.tempo')
ax7.set_xlim(0,160)
ax7.set_xticks([])
ax7.set_ylim(0, node_num)
ax7.set_yticks([])

#ax3.set_ylabel('State Index')
#add a vertical dash line at the middle of x
#x_mid = (plt.xlim()[0] + plt.xlim()[1]) / 2
#ax7.axvline(x=x_mid, color='k', linestyle='--')

ax8 = fig.add_subplot(gs[-1, 1])

ax8.set_position([ax8.get_position().x0, ax8.get_position().y0 + 0.02, ax8.get_position().width, ax8.get_position().height])

ax8.plot(t_arr, pred_amp[node_num//2-1,:],'k', linewidth=2.0)
ax8.set_xlim(0,160)
ax8.set_ylim(0, 1.5)
#ax8.set_xlabel(r'Time, $t$')
#ax4.set_ylabel(r'Derivative, $\dot{x}_1$')
ax8.set_xticks([0, 40, 80, 120, 160])
ax8.set_yticks([])
#add a vertical dash line at the middle of x axis
#ax8.axvline(x=reference_line, color='k', linestyle='--')

#ax6.text(-0.05, -0.8, '(c)', transform=ax6.transAxes, size=14, weight='bold')
#ax8.text(-0.05, -0.8, '(d)', transform=ax8.transAxes, size=14, weight='bold')

cbar_ax = fig.add_subplot(gs[0:-1, 2])
cbar_ax.set_position([cbar_ax.get_position().x0, cbar_ax.get_position().y0 + 0.02, cbar_ax.get_position().width, cbar_ax.get_position().height])
cbar = fig.colorbar(cax5, ax=ax5, orientation='vertical', cax=cbar_ax)
cbar.set_label(r'$| u_i |$')


# Adjust the X label position for ax6
ax6.xaxis.set_label_coords(0.5, -0.7)  # Move the label slightly up; adjust y-value as needed

# Adjust the X label position for ax8
ax8.xaxis.set_label_coords(0.5, -0.7)  # Move the label slightly up; adjust y-value as needed

plt.savefig("DNLS_simulation_vs_infer_random_IC", dpi=600)


################################
#############
############# error plot
#############
#####################################

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
    #'text.usetex': True,
    #'figure.figsize': [7, 4],
    'font.family': 'Arial',
    #'text.latex.preamble': r'\usepackage{amsmath}',  # Optional, for advanced math symbols
    #'axes.labelweight': 'bold'
}
matplotlib.rcParams.update(params)

fig2 = plt.figure(figsize=(8, 4))

cax9 = plt.imshow(np.abs(pred_amp-sim_amp), extent=[0, 2*T, 0, node_num], aspect='auto', cmap='cmo.tempo')
cbar9=fig2.colorbar(cax9)  # Display a colorbar to interpret the color scale
plt.ylim(0,30)
plt.yticks([0, 12.5, 25, 37.5, 50], [-25, -12.5, 0, 12.5, 25])
#plt.set_yticklabels([-25, 0, 25])
plt.xticks([0, 40, 80, 120, 160])
plt.ylabel(r'State Index, $j$')
plt.xlabel(r'Time, $t$')
#plt.axvline(x=60, color='k', linestyle='--')

#set colobar label
cbar9.set_label(r'$| u_j - \hat{u}_j |$')

plt.tight_layout()
plt.savefig("DNLS_error_random_IC", dpi=1200)

###############################
################################
##############################
#time history of some dimensions
fig3 = plt.figure(figsize=(8, 6))


gs = fig3.add_gridspec(4, 2)

points_to_plot = -1

ax11 = fig3.add_subplot(gs[0:2, 0])
ax11.plot(t_arr[:points_to_plot], sim_amp[4,:points_to_plot], 'k', label='True')
ax11.plot(t_arr[:points_to_plot], pred_amp[4,:points_to_plot], 'r--', label='pred_dxicted')
ax11.set_ylabel(r'$| u_{-20} |$')
ax11.set_xticks([0,60,120, 180])
reference_line = 60 
#ax11.axvline(x=reference_line, color='k', linestyle='--')


ax12 = fig3.add_subplot(gs[0:2, 1])
ax12.plot(t_arr[:points_to_plot], sim_amp[14,:points_to_plot], 'k', label='True')
ax12.plot(t_arr[:points_to_plot], pred_amp[14,:points_to_plot], 'r--', label='pred_dxicted')
ax12.set_ylabel(r'$| u_{-10} |$')
ax12.set_xticks([0,60,120, 180])
#ax12.axvline(x=reference_line, color='k', linestyle='--')


ax13 = fig3.add_subplot(gs[2:4, 0])
ax13.plot(t_arr[:points_to_plot], sim_amp[24,:points_to_plot], 'k', label='True')
ax13.plot(t_arr[:points_to_plot], pred_amp[24,:points_to_plot], 'r--', label='pred_dxicted')
ax13.set_xlabel(r'Time, $t$')
ax13.set_ylabel(r'$| u_{0} |$')
ax13.set_xticks([0,60,120, 180])
#ax13.axvline(x=reference_line, color='k', linestyle='--')


ax14 = fig3.add_subplot(gs[2:4, 1])
ax14.plot(t_arr[:points_to_plot], sim_amp[34,:points_to_plot], 'k', label='True')
ax14.plot(t_arr[:points_to_plot], pred_amp[34,:points_to_plot], 'r--', label='pred_dxicted')
ax14.set_xlabel(r'Time, $t$')
ax14.set_ylabel(r'$| u_{10} |$')
#ax14.set_yticks([-5,0, 5, 10])
ax14.set_xticks([0,60,120, 180])
#ax14.axvline(x=reference_line, color='k', linestyle='--')

plt.tight_layout()
plt.savefig("DNLS_time_history_Random_IC", dpi=1200)
