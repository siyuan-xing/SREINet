import pandas as pd
import ast
import matplotlib.pyplot as plt
import numpy as np

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
    'axes.labelsize': 10, # fontsize for x and y labels (was 10)
    'axes.titlesize': 12,
    'font.size': 10, # was 10
    'legend.fontsize': 10, # was 10
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'text.usetex': True,
    #'figure.figsize': [7, 4],
    'font.family': 'serif',
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

my_size = 110/25.4 #110mm

fig = plt.figure(figsize=(my_size*2, my_size))

gs = fig.add_gridspec(10, 3, width_ratios=[20, 20, 1], hspace=0.1, wspace=0.2)

#ax2 line thickness set to 2
plt.rcParams['text.usetex'] = True  # Uncomment if you have LaTeX installed
plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'  # Optional, for advanced math symbols
plt.rcParams['axes.labelweight'] = 'bold'

import matplotlib.colors as mcolors

ax1 = fig.add_subplot(gs[:4, 0])
ax1.set_position([ax1.get_position().x0, ax1.get_position().y0 + 0.02, ax1.get_position().width, ax1.get_position().height])
cax1 = ax1.imshow(heat_map_dx, aspect='auto', cmap='cmo.tempo')

step = 400
ax1.set_xticks([])
ax1.set_yticks([0,25,50,75,100])

ax1.set_ylabel(r'State Index, $i$')
#add a vertical dash line at the middle of x
x_mid = (plt.xlim()[0] + plt.xlim()[1]) / 2
#ax1.axvline(x=x_mid, color='k', linestyle='--')
ax1.set_title("Full Simulation")

ax2 = fig.add_subplot(gs[4, 0])
ax2.set_position([ax2.get_position().x0, ax2.get_position().y0 + 0.02, ax2.get_position().width, ax2.get_position().height])
ax2.plot(heatmap_t_arr, heat_map_dx[0],'k', linewidth=2.0)
ax2.set_xlim(0,20)
ax2.set_ylim(-100,100)
#ax2.set_xlabel(r'Time, $t$')
ax2.set_ylabel(r'$\dot{x}_1$')
ax2.set_xticks([])
ax2.set_yticks([-50, 50])
#add a vertical dash line at the middle of x axis

reference_line = 10 
#ax2.axvline(x=reference_line, color='k', linestyle='--')


#### 3 and 4
ax3 = fig.add_subplot(gs[:4, 1])
ax3.set_position([ax3.get_position().x0, ax3.get_position().y0 + 0.02, ax3.get_position().width, ax3.get_position().height])
cax3 = ax3.imshow(heat_map_pred_dx, aspect='auto', cmap='cmo.tempo')

#ax1.set_xticks(ticks=np.arange(0,heat_map_x.shape[1],step), labels=heatmap_t_arr[::step])
ax3.set_xticks([])
ax3.set_yticks([0,25,50,75,100])

#ax3.set_ylabel('State Index')
#add a vertical dash line at the middle of x
x_mid = (plt.xlim()[0] + plt.xlim()[1]) / 2
#ax3.axvline(x=x_mid, color='k', linestyle='--')
ax3.set_title("Identified System")

ax4 = fig.add_subplot(gs[4, 1])
ax4.set_position([ax4.get_position().x0, ax4.get_position().y0 + 0.02, ax4.get_position().width, ax4.get_position().height])
ax4.plot(heatmap_t_arr, heat_map_pred_dx[0],'k', linewidth=2.0)
ax4.set_xlim(0,20)
ax4.set_ylim(-100,100)
#ax4.set_xlabel(r'Time, $t$')
#ax4.set_ylabel(r'Derivative, $\dot{x}_1$')
ax4.set_xticks([])
ax4.set_yticks([-50, 50])
#add a vertical dash line at the middle of x axis
#ax4.axvline(x=reference_line, color='k', linestyle='--')


#ax2.text(-0.05, -0.4, '(a)', transform=ax2.transAxes, size=14,  fontweight='bold')
#ax4.text(-0.05, -0.4, '(b)', transform=ax4.transAxes, size=14,  fontweight='bold')


cbar1_ax = fig.add_subplot(gs[:4, 2])
cbar1_ax.set_position([cbar1_ax.get_position().x0, cbar1_ax.get_position().y0 + 0.02, cbar1_ax.get_position().width, cbar1_ax.get_position().height])

cbar = fig.colorbar(cax1, ax=ax1, orientation='vertical', cax=cbar1_ax)
cbar.set_label(r'Derivative, $\dot{x}_i$')
#set cbar ticks
cbar.set_ticks([-100, -50, 0, 50])


ax5 = fig.add_subplot(gs[5:-1, 0])
ax5.set_position([ax5.get_position().x0, ax5.get_position().y0 - 0.02, ax5.get_position().width, ax5.get_position().height])

cax5 = ax5.imshow(heat_map_x, aspect='auto', cmap='cmo.tempo')

step = 400
ax5.set_xticks([])
ax5.set_yticks([0,25,50,75,100])

ax5.set_ylabel(r'State Index, $i$')
#add a vertical dash line at the middle of x
x_mid = (plt.xlim()[0] + plt.xlim()[1]) / 2
#ax5.axvline(x=x_mid, color='k', linestyle='--')

ax6 = fig.add_subplot(gs[-1, 0])
ax6.set_position([ax6.get_position().x0, ax6.get_position().y0 - 0.02, ax6.get_position().width, ax6.get_position().height])
ax6.plot(heatmap_t_arr, heat_map_x[0],'k', linewidth=2.0)
ax6.set_xlim(0,20)
ax6.set_ylim(-15,15)
ax6.set_xlabel(r'Time, $t$',labelpad=-0.5)
ax6.set_ylabel(r'$x_1$')
ax6.set_xticks([0,5.0,10.0, 15, 20])
ax6.set_yticks([-5,5])
#add a vertical dash line at the middle of x axis

reference_line = 10 
#ax6.axvline(x=reference_line, color='k', linestyle='--')


#### 3 and 4
ax7 = fig.add_subplot(gs[5:-1, 1])
ax7.set_position([ax7.get_position().x0, ax7.get_position().y0 - 0.02, ax7.get_position().width, ax7.get_position().height])

cax7 = ax7.imshow(heat_map_pred_x, aspect='auto', cmap='cmo.tempo')

#ax1.set_xticks(ticks=np.arange(0,heat_map_x.shape[1],step), labels=heatmap_t_arr[::step])
ax7.set_xticks([])
ax7.set_yticks([0,25,50,75,100])

#ax3.set_ylabel('State Index')
#add a vertical dash line at the middle of x
x_mid = (plt.xlim()[0] + plt.xlim()[1]) / 2
#ax7.axvline(x=x_mid, color='k', linestyle='--')

ax8 = fig.add_subplot(gs[-1, 1])
ax8.set_position([ax8.get_position().x0, ax8.get_position().y0 - 0.02, ax8.get_position().width, ax8.get_position().height])

ax8.plot(heatmap_t_arr, heat_map_pred_x[0],'k', linewidth=2.0)
ax8.set_xlim(0,20)
ax8.set_ylim(-15,15)
ax8.set_xlabel(r'Time, $t$', labelpad=-0.5)
#ax4.set_ylabel(r'Derivative, $\dot{x}_1$')
ax8.set_xticks([0,5.0,10.0, 15, 20])
ax8.set_yticks([-5,5])
#add a vertical dash line at the middle of x axis
#ax8.axvline(x=reference_line, color='k', linestyle='--')


#ax6.text(-0.05, -0.8, '(c)', transform=ax6.transAxes, size=14, weight='bold')
#ax8.text(-0.05, -0.8, '(d)', transform=ax8.transAxes, size=14, weight='bold')

cbar2_ax = fig.add_subplot(gs[5:-1, 2])
cbar2_ax.set_position([cbar2_ax.get_position().x0, cbar2_ax.get_position().y0 - 0.02, cbar2_ax.get_position().width, cbar2_ax.get_position().height])

cbar = fig.colorbar(cax5, ax=ax5, orientation='vertical', cax=cbar2_ax)
cbar.set_ticks([-10, -5, 0, 5, 10])
cbar.set_label(r'State, $x_i$')





plt.tight_layout()
plt.savefig("Lorenz_96_simulation_vs_infer", dpi=600)




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
    'text.usetex': True,
    #'figure.figsize': [7, 4],    
    'font.family': 'Helvetica',
    'text.latex.preamble': r'\usepackage{amsmath}',  # Optional, for advanced math symbols
    'axes.labelweight':'bold'
    
}
matplotlib.rcParams.update(params)



################################
#############
############# error plot
#############
#####################################
fig2 = plt.figure(figsize=(10, 8))

gs = fig2.add_gridspec(2, 1)

ax9 = fig2.add_subplot(gs[0, 0])
cax9 = ax9.imshow(np.abs(heat_map_dx-heat_map_pred_dx), extent=[0, 20, 0, 100], aspect='auto', cmap='cmo.tempo')
cbar_1=fig2.colorbar(cax9)  # Display a colorbar to interpret the color scale
cbar_1.set_label(r'Derivative Error, $|\dot{x}_i - \hat{\dot{x}}_i|$')
cbar_1.formatter.set_powerlimits((0, 0))  # This sets the threshold for using scientific notation.
cbar_1.update_ticks()  # This updates the colorbar with the new formatter settings.
ax9.set_yticks(ticks=[0,25,50,75,100])
ax9.set_xticks(ticks=[0,5,10,15,20])
ax9.set_ylabel(r'State Index, $i$')
#ax9.axvline(x=reference_line, color='k', linestyle='--')

ax10 = fig2.add_subplot(gs[1, 0])
cax10 = ax10.imshow(np.abs(heat_map_x-heat_map_pred_x), extent=[0, 20, 0, 100], aspect='auto', cmap='cmo.tempo')
cbar_2 = fig2.colorbar(cax10)  # Display a colorbar to interpret the color scale
cbar_2.set_label(r'Error, $|x_i - \hat{x}_i|$')
ax10.set_yticks(ticks=[0,25,50,75,100])
ax10.set_xticks(ticks=[0,5,10,15,20])
ax10.set_ylabel(r'State Index, $i$')
ax10.set_xlabel(r'Time, $t$')
#ax10.axvline(x=reference_line, color='k', linestyle='--')

ax9.text(-0.05, -0.1, '(a)', transform=ax9.transAxes, size=14, weight='bold', fontfamily='Arial')
ax10.text(-0.05, -0.1, '(b)', transform=ax10.transAxes, size=14, weight='bold', fontfamily='Arial')


plt.tight_layout()
plt.savefig("Lorenz_96_error", dpi=600)

###############################
################################
##############################
#time history of some dimensions
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
