import pandas as pd
import ast
import matplotlib.pyplot as plt
import numpy as np
from kuramoto_ode import kuramoto_ode
#read matrix from train_data.mat
import scipy.io

import DataGenerator as DG

import numpy as np

from scipy.integrate import solve_ivp

# Generate data based on the specified model type

"""
DIM = 60  # You can change this value to your desired number of elements
num_traj = 1

data_initial_conditions = DG.DataGenerator.generate_initial_conditions(DIM, 1) #np.random.normal(1, 2, LORENZ96_DIM

wn = np.linspace(-5, 5, DIM)

data_T = 200.0       # Length of the data (T)
data_dt = 0.1     # Resolution of the data (dt)
myDG = DG.DataGenerator(data_initial_conditions,T=data_T, dt=data_dt, derivative_mode="exact")
t_arr, x_train, dx_train, guess_highest_order_polynomial = myDG.generate_dataset_by_model_name('kuramoto' , wn, method='BDF')
"""


train_data = scipy.io.loadmat('train_data.mat')
val_data = scipy.io.loadmat('validation_data.mat')

#extract the data from the fourth sheet
train_data = train_data['data']
val_data = val_data['data']

dim = 60

T = 500
dt = 0.1
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
solution = solve_ivp(kuramoto_ode, t_span, initial_state, t_eval=t_eval, method='BDF')

params = {
    'image.origin': 'lower',
    'image.interpolation': 'nearest',
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
    'font.family': 'Helvetica',
    'text.latex.preamble': r'\usepackage{amsmath}',  # Optional, for advanced math symbols
    'axes.labelweight': 'bold'}
import matplotlib
import cmocean
matplotlib.rcParams.update(params)

#plot heatmap and time trajectory of dim 1
#x_arr = np.concatenate((x_train,val_x_train),axis=0)
#dx_arr = np.concatenate((dx_train,val_dx_train),axis=0)
#pred_dx_arr = np.concatenate((pred_dx,val_pred_dx),axis=0)
#t_arr = np.concatenate((t_arr,val_t_arr),axis=0)

x_arr = x_train
dx_arr = dx_train
pred_dx_arr = pred_dx
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

mysize = 110/25.4
fig = plt.figure(figsize=(mysize*2, mysize))

gs = fig.add_gridspec(10, 3, width_ratios=[20, 20, 1], hspace=0.1, wspace=0.2)


import matplotlib.colors as mcolors

ax1 = fig.add_subplot(gs[:4, 0])
ax1.set_position([ax1.get_position().x0, ax1.get_position().y0 + 0.02, ax1.get_position().width, ax1.get_position().height])
cax1 = ax1.imshow(heat_map_dx, aspect='auto', extent=[0, T, 0, dim], cmap='cmo.tempo')

step = 400
ax1.set_xticks([])
ax1.set_yticks([0,20,40,60])

ax1.set_ylabel(r'State Index, $i$')
#add a vertical dash line at the middle of x
x_mid = (plt.xlim()[0] + plt.xlim()[1]) / 2
ax1.axvline(x=x_mid, color='k', linestyle='--')
ax1.set_title("Full Simulation")

ax2 = fig.add_subplot(gs[4, 0])
ax2.set_position([ax2.get_position().x0, ax2.get_position().y0 + 0.02, ax2.get_position().width, ax2.get_position().height])
ax2.plot(heatmap_t_arr, heat_map_dx[0],'k', linewidth=2.0)
ax2.set_xlim(0,T)
ax2.set_ylim(-7,-3)
#ax2.set_xlabel(r'Time, $t$')
ax2.set_ylabel(r'$\dot{\theta}_1$')
ax2.set_xticks([])
#ax2.set_yticks([-50, 50])
#add a vertical dash line at the middle of x axis

reference_line = 250 
ax2.axvline(x=reference_line, color='k', linestyle='--')


#### 3 and 4
ax3 = fig.add_subplot(gs[:4, 1])
ax3.set_position([ax3.get_position().x0, ax3.get_position().y0 + 0.02, ax3.get_position().width, ax3.get_position().height])
cax3 = ax3.imshow(heat_map_pred_dx, aspect='auto', extent=[0, T, 0, dim], cmap='cmo.tempo')

#ax1.set_xticks(ticks=np.arange(0,heat_map_x.shape[1],step), labels=heatmap_t_arr[::step])
ax3.set_xticks([])
ax3.set_yticks([0,20,40,60])

#ax3.set_ylabel('State Index')
#add a vertical dash line at the middle of x
x_mid = (plt.xlim()[0] + plt.xlim()[1]) / 2
ax3.axvline(x=x_mid, color='k', linestyle='--')
ax3.set_title("Identified System")

ax4 = fig.add_subplot(gs[4, 1])
ax4.set_position([ax4.get_position().x0, ax4.get_position().y0 + 0.02, ax4.get_position().width, ax4.get_position().height])
ax4.plot(heatmap_t_arr, heat_map_pred_dx[0],'k', linewidth=2.0)
ax4.set_xlim(0,T)
ax4.set_ylim(-7,-3)
#ax4.set_xlabel(r'Time, $t$')
#ax4.set_ylabel(r'Derivative, $\dot{x}_1$')
ax4.set_xticks([])
#ax4.set_yticks([-50, 50])
#add a vertical dash line at the middle of x axis
ax4.axvline(x=reference_line, color='k', linestyle='--')


#ax2.text(-0.05, -0.4, '(a)', transform=ax2.transAxes, size=14,  fontweight='bold')
#ax4.text(-0.05, -0.4, '(b)', transform=ax4.transAxes, size=14,  fontweight='bold')

cbar_ax_1 = fig.add_subplot(gs[:4, 2])
cbar_ax_1.set_position([cbar_ax_1.get_position().x0, cbar_ax_1.get_position().y0 + 0.02, cbar_ax_1.get_position().width, cbar_ax_1.get_position().height])

cbar = fig.colorbar(cax1, ax=ax1, orientation='vertical', cax=cbar_ax_1)
cbar.set_label(r'Frequencies, $\dot{\theta}_i$')



ax5 = fig.add_subplot(gs[5:-1, 0])
ax5.set_position([ax5.get_position().x0, ax5.get_position().y0 - 0.02, ax5.get_position().width, ax5.get_position().height])

cax5 = ax5.imshow(heat_map_x % (2*np.pi), aspect='auto', extent=[0, T, 0, dim], cmap='cmo.tempo')

step = 400
ax5.set_xticks([])
ax5.set_yticks([0,20,40,60])

ax5.set_ylabel(r'State Index, $i$')
#add a vertical dash line at the middle of x
#x_mid = (plt.xlim()[0] + plt.xlim()[1]) / 2
#ax5.axvline(x=x_mid, color='k', linestyle='--')

ax6 = fig.add_subplot(gs[-1, 0])
ax6.set_position([ax6.get_position().x0, ax6.get_position().y0 - 0.02, ax6.get_position().width, ax6.get_position().height])
ax6.plot(heatmap_t_arr, heat_map_x[29] % (2*np.pi),'k', linewidth=2.0)
ax6.set_xlim(0,T)
#ax6.set_ylim(0,2*np.pi)
ax6.set_xlabel(r'Time, $t$', labelpad=-0.5)
ax6.set_ylabel(r'$\theta_{30}$')
ax6.set_yticks([0,np.pi])
ax6.set_yticklabels([r'$0$', r'$\pi$'])
#ax6.set_xticks([0,5.0,10.0, 15, 20])
#ax6.set_yticks([-5,5])
#add a vertical dash line at the middle of x axis

#reference_line = 100
#ax6.axvline(x=reference_line, color='k', linestyle='--')

#### 3 and 4
ax7 = fig.add_subplot(gs[5:-1, 1])
ax7.set_position([ax7.get_position().x0, ax7.get_position().y0 - 0.02, ax7.get_position().width, ax7.get_position().height])

cax7 = ax7.imshow(heat_map_pred_x % (2*np.pi), aspect='auto', extent=[0, T, 0, dim], cmap='cmo.tempo')

#ax1.set_xticks(ticks=np.arange(0,heat_map_x.shape[1],step), labels=heatmap_t_arr[::step])
ax7.set_xticks([])
ax7.set_yticks([0,20,40,60])

#ax3.set_ylabel('State Index')
#add a vertical dash line at the middle of x
#x_mid = (plt.xlim()[0] + plt.xlim()[1]) / 2
#ax7.axvline(x=x_mid, color='k', linestyle='--')

ax8 = fig.add_subplot(gs[-1, 1])
ax8.set_position([ax8.get_position().x0, ax8.get_position().y0 - 0.02, ax8.get_position().width, ax8.get_position().height])

ax8.plot(heatmap_t_arr, heat_map_pred_x[29,:] % (2*np.pi),'k', linewidth=2.0)
#ax8.set_xlim(0,20)
#ax8.set_ylim(0,2*np.pi)
ax8.set_xlabel(r'Time, $t$',labelpad=-0.5)
#ax4.set_ylabel(r'Derivative, $\dot{x}_1$')
#ax8.set_xticks([0,5.0,10.0, 15, 20])
ax8.set_xlim(0,T)
ax8.set_yticks([0, np.pi,])
ax8.set_yticklabels([r'$0$', r'$\pi$'])
#add a vertical dash line at the middle of x axis
#ax8.axvline(x=reference_line, color='k', linestyle='--')


#ax6.text(-0.05, -0.8, '(c)', transform=ax6.transAxes, size=14, weight='bold')
#ax8.text(-0.05, -0.8, '(d)', transform=ax8.transAxes, size=14, weight='bold')


cbar_ax = fig.add_subplot(gs[5:-1, 2])
cbar_ax.set_position([cbar_ax.get_position().x0, cbar_ax.get_position().y0 - 0.02, cbar_ax.get_position().width, cbar_ax.get_position().height])
cbar_2 = fig.colorbar(cax5, ax=ax5, orientation='vertical', cax=cbar_ax)

cbar_2.set_label(r'Phase, $\theta_i$')
cbar_2.set_ticks([0, np.pi, 2*np.pi])
cbar_2.set_ticklabels([r'$0$', r'$\pi$', r'$2\pi$'])

#change color bard tick labels

plt.tight_layout()
plt.savefig("Kuramoto_60D_simulation_vs_infer", dpi=600)



################################
#############
############# error plot
#############
#####################################



params = {
    'image.origin': 'lower',
    'image.interpolation': 'nearest',
    'axes.grid': False,
    'savefig.dpi': 600,  # to adjust notebook inline plot size
    'axes.labelsize': 14, # fontsize for x and y labels (was 10)
    'axes.titlesize': 16,
    'font.size': 14, # was 10
    'legend.fontsize': 14, # was 10
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'text.usetex': True,
    #'figure.figsize': [7, 4],
    'font.family': 'Helvetica',
    'text.latex.preamble': r'\usepackage{amsmath}',  # Optional, for advanced math symbols
    'axes.labelweight': 'bold'
}
import matplotlib
matplotlib.rcParams.update(params)


fig2 = plt.figure(figsize=(8, 4))


"""
gs = fig2.add_gridspec(2, 1)

ax9 = fig2.add_subplot(gs[0, 0])
cax9 = ax9.imshow(np.abs(heat_map_dx-heat_map_pred_dx), extent=[0, T, 0, 60], aspect='auto', cmap='cmo.tempo')
cbar = fig2.colorbar(cax9)  # Display a colorbar to interpret the color scale
cbar.set_label(r'$|\hat{\dot{\theta}}_i - \dot{\theta}_i^|$')
ax9.set_yticks(ticks=[0,20,40,60])
#ax9.set_xticks(ticks=[0,5,10,15,20])
ax9.set_ylabel(r'State Index, $i$')
#ax9.axvline(x=reference_line, color='k', linestyle='--')

ax10 = fig2.add_subplot(gs[1, 0])
"""
cax10 = plt.imshow(np.abs(heat_map_x-heat_map_pred_x) % (2*np.pi), extent=[0, T, 0, 60], aspect='auto', cmap='cmo.tempo')
cbar2 = fig2.colorbar(cax10)  # Display a colorbar to interpret the color scale
#colorbar range -10 to 10
plt.xlim(0, 100)
plt.yticks(ticks=[0,20,40,60])
plt.xticks(ticks=[0, 25, 50, 75, 100])
#plt.set_xticks(ticks=[0,5,10,15,20])
plt.ylabel(r'State Index, $i$')
plt.xlabel(r'Time, $t$')
cbar2.set_label(r'$|\hat{\theta}_i - \theta_i^|$')
#plt.axvline(x=reference_line, color='k', linestyle='--')

#ax9.text(-0.05, -0.1, '(a)', transform=ax9.transAxes, size=14, weight='bold', fontfamily='Arial')
#ax10.text(-0.05, -0.1, '(b)', transform=ax10.transAxes, size=14, weight='bold', fontfamily='Arial')


plt.tight_layout()
plt.savefig("Kuramoto_60D_error", dpi=600)

###############################
################################
##############################
#time history of some dimensions


fig3 = plt.figure(figsize=(8, 6))


from polar_plot import plot_kuramoto_phases

gs = fig3.add_gridspec(4, 2)

half_point = 10000

ax11 = fig3.add_subplot(gs[0:2, 0], projection='polar')
plot_kuramoto_phases(pred_x_arr[0,0::10], 
                     pred_dx_arr[0,0::10], 
                     x_arr[0,0::10], 
                     dx_arr[0,0::10],
                     ax11,
                     r"$t=0$")

t=100
ax12 = fig3.add_subplot(gs[0:2, 1], projection='polar')
plot_kuramoto_phases(pred_x_arr[t,0::10], 
                     pred_dx_arr[t,0::10], 
                     x_arr[t,0::10], 
                     dx_arr[t,0::10],
                     ax12,
                     r"$t=10$")

t=200
ax13 = fig3.add_subplot(gs[2:4, 0], projection='polar')
plot_kuramoto_phases(pred_x_arr[t,0::10], 
                     pred_dx_arr[t,0::10], 
                     x_arr[t,0::10], 
                     dx_arr[t,0::10],
                     ax13,
                     r"$t=20$")

t=300
ax14 = fig3.add_subplot(gs[2:4, 1], projection='polar')
plot_kuramoto_phases(pred_x_arr[t,0::10], 
                     pred_dx_arr[t,0::10], 
                     x_arr[t,0::10], 
                     dx_arr[t,0::10],
                     ax14,
                     r"$t=30$")

# Adjust subplot spacing to give more room for polar plots
# Increase margins to prevent circle clipping
fig3.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08, wspace=0.3, hspace=0.4)

#plt.tight_layout()
plt.savefig("Kuramoto_time_history", dpi=600)
