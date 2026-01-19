import pandas as pd
import ast
import matplotlib.pyplot as plt
import numpy as np

#read matrix from train_data.mat
import scipy.io
import numpy as np

from scipy.integrate import solve_ivp

from phi_ode import phi4_ode

import DataGenerator as DG

# Specify the model type
MY_MODEL = 'discrete_phi_quartic' # discrete sine Gordon model 

# Generate data based on the specified model type
node_num = 50
C = 2.0 #coupling strength



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
#dx_train = train_data[:one_traj_length,dim+1:2*dim+1]

#pred_dx = train_data[:one_traj_length,2*dim+1:3*dim+1]

#validation data
#val_t_arr = val_data[1:,0:1]
#val_x_train = val_data[1:,1:dim+1] #1
#val_dx_train = val_data[1:,dim+1:2*dim+1]

#val_pred_dx = val_data[1:,2*dim+1:3*dim+1]


# Define the initial state (data from nn_configurations.xlsx)
data_initial_conditions = x_train[0,:]

#initial conditions for Simulation
"""
xn = np.linspace(-node_num/2/np.sqrt(C), node_num/2/np.sqrt(C), node_num)

v=0.0
x0 =  np.tanh(xn/np.sqrt(1-v**2))
d_x0 = 1/np.sqrt(1-v**2) * (1 - np.tanh(xn/np.sqrt(1-v**2))**2)
data_initial_conditions = np.concatenate((x0, d_x0), axis=0) # add velocity
"""
data_T =  500.0       # Length of the data (T)
data_dt = 0.05       # Resolution of the data (dt)
myDG = DG.DataGenerator(data_initial_conditions,T=data_T, dt=data_dt)

t_arr, x_train, dx_train, guess_highest_order_polynomial = myDG.generate_dataset_by_model_name(MY_MODEL, C, node_num)

print("Guess highest order polynomial:", guess_highest_order_polynomial)

#plot x_train and dx_train
import matplotlib.pyplot as plt
print(x_train.shape)
#plt.scatter(x_train, dx_train)
#plt.show()

# Define the initial state (data from nn_configurations.xlsx)
initial_state = x_train[0,:]
# Define the time span for the simulation
T = data_T
dt = 0.05
t_span = (0, T)
t_eval = np.arange(0, T, dt)

# Solve the ODE system
solution = solve_ivp(phi4_ode, t_span, initial_state, t_eval=t_eval)


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
    'font.family': 'serif',
}
import matplotlib

import cmocean
matplotlib.rcParams.update(params)


#plot heatmap and time trajectory of dim 1
x_arr = x_train

dx_dt_arr = dx_train
pred_x_arr = solution.y[0:node_num,:].T
pred_dx_dt_arr = solution.y[node_num:,:].T

sample_per_points = 1
heatmap_t_arr = t_arr

heat_map_x = x_arr[:,:node_num]
heat_map_dx = dx_dt_arr[:,:node_num]


heat_map_pred_dx = pred_dx_dt_arr
heat_map_pred_x = pred_x_arr

heat_map_x = heat_map_x.T
heat_map_dx = heat_map_dx.T
heat_map_pred_dx = heat_map_pred_dx.T
heat_map_pred_x = heat_map_pred_x.T

print(heat_map_x.shape)
print(heat_map_dx.shape)
print(heat_map_pred_dx.shape)
print(heat_map_pred_x.shape)

my_size = 110/25.4 #110mm

fig = plt.figure(figsize=(my_size*2, my_size/2), facecolor='none')

gs = fig.add_gridspec(5, 3, width_ratios=[20, 20, 1], hspace=0.1, wspace=0.1)

#ax2 line thickness set to 2
plt.rcParams['text.usetex'] = False  # Uncomment if you have LaTeX installed
plt.rcParams['font.family'] = 'Helvetica'
#plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'  # Optional, for advanced math symbols
#plt.rcParams['axes.labelweight'] = 'bold'

import matplotlib.colors as mcolors


ax5 = fig.add_subplot(gs[0:-1, 0])
ax5.set_position([ax5.get_position().x0, ax5.get_position().y0 + 0.02, ax5.get_position().width, ax5.get_position().height])

plot_until_points = 1600
cax5 = ax5.imshow(heat_map_x[:,:plot_until_points], aspect='auto', extent=[0,80, 0, 50], cmap='cmo.tempo')

ax5.set_xlim(0,45)
ax5.set_xticks([])
ax5.set_ylim(0,50)
ax5.set_yticks([0, 25, 50],[-25,0,25])
ax5.set_title("Full Simulation")

ax5.set_ylabel(r'State Index, $i$')
#add a vertical dash line at the middle of x
#x_mid = (plt.xlim()[0] + plt.xlim()[1]) / 2
#ax5.axvline(x=x_mid, color='k', linestyle='--')

ax6 = fig.add_subplot(gs[-1, 0])
ax6.set_position([ax6.get_position().x0, ax6.get_position().y0 + 0.02, ax6.get_position().width, ax6.get_position().height])
ax6.plot(heatmap_t_arr[:plot_until_points], heat_map_x[16][:plot_until_points],'k', linewidth=2.0)
ax6.set_xlim(0,45)
ax6.set_ylim(-5,5)
#ax6.set_xlabel(r'Time, $t$', labelpad=-1)
ax6.set_ylabel(r'$u_{-9}$', rotation=0)
ax6.yaxis.set_label_coords(-0.12,0.22)
#rotate y axis label by 90 degree


ax6.set_xticks([0, 15, 30, 45])
ax6.set_yticks([0])
#add a vertical dash line at the middle of x axis

#reference_line = 25 
#ax6.axvline(x=reference_line, color='k', linestyle='--')

#### 3 and 4
ax7 = fig.add_subplot(gs[0:-1, 1])
ax7.set_position([ax7.get_position().x0, ax7.get_position().y0 + 0.02, ax7.get_position().width, ax7.get_position().height])

cax7 = ax7.imshow(heat_map_pred_x[:,:plot_until_points], aspect='auto', extent=[0,80, 0, 50], cmap='cmo.tempo')

ax7.set_xlim(0,45)
ax7.set_xticks([])
ax7.set_ylim(0,50)
ax7.set_yticks([])
#ax7.set_yticks([0, 25, 50])
ax7.set_title("Identified System")

#ax3.set_ylabel('State Index')
#add a vertical dash line at the middle of x
#x_mid = (plt.xlim()[0] + plt.xlim()[1]) / 2
#ax7.axvline(x=x_mid, color='k', linestyle='--')

ax8 = fig.add_subplot(gs[-1, 1])

ax8.set_position([ax8.get_position().x0, ax8.get_position().y0 + 0.02, ax8.get_position().width, ax8.get_position().height])

ax8.plot(heatmap_t_arr[:plot_until_points], heat_map_pred_x[16][:plot_until_points],'k', linewidth=2.0)
ax8.set_xlim(0,45)
ax8.set_ylim(-5,5)
#ax8.set_xlabel(r'Time, $t$', labelpad=-1)
#ax4.set_ylabel(r'Derivative, $\dot{x}_1$')
ax8.set_xticks([0, 15, 30, 45])
ax8.set_yticks([])
#ax8.set_yticks([-2, 2])
#add a vertical dash line at the middle of x axis
#ax8.axvline(x=reference_line, color='k', linestyle='--')

#ax6.text(-0.05, -0.8, '(c)', transform=ax6.transAxes, size=14, weight='bold')
#ax8.text(-0.05, -0.8, '(d)', transform=ax8.transAxes, size=14, weight='bold')

cbar_ax = fig.add_subplot(gs[0:-1, 2])
cbar_ax.set_position([cbar_ax.get_position().x0, cbar_ax.get_position().y0 + 0.02, cbar_ax.get_position().width, cbar_ax.get_position().height])
cbar = fig.colorbar(cax5, ax=ax5, orientation='vertical', cax=cbar_ax)
cbar.set_label(r'$u_i$')

#plt.tight_layout()
plt.savefig("phi_4_simulation_vs_infer_random", dpi=600)


###############################
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
    'text.usetex': True,
    #'figure.figsize': [7, 4],
    'font.family': 'Helvetica',
    'text.latex.preamble': r'\usepackage{amsmath}\usepackage{helvet}\renewcommand{\familydefault}{\sfdefault}',  # Optional, for advanced math symbols
    'axes.labelweight': 'bold'
}
matplotlib.rcParams.update(params)

fig2 = plt.figure(figsize=(8, 4))

#gs = fig2.add_gridspec(2, 1)

#ax9 = fig2.add_subplot(gs[1, 0])
#cax9 = ax9.imshow(np.abs(heat_map_dx-heat_map_pred_dx), extent=[0, T, 0, 100], aspect='auto', cmap='cmo.tempo')
#cbar9=fig2.colorbar(cax9)  # Display a colorbar to interpret the color scale
#ax9.set_ylim(50,100)
#ax9.set_xlim(0,200)
##ax9.set_yticks(ticks=[50,62.5,75, 87.5, 100])
#ax9.set_yticklabels([0,12.5,25,37.5,50])
#ax9.set_xticks(ticks=[0,50, 100, 150, 200])
#ax9.set_ylabel(r'State Index, $i$')
#ax9.axvline(x=reference_line, color='k', linestyle='--')
#set colobar label
#cbar9.set_label(r'$| \dot{u}_i - \dot{\hat{u}}_i |$')

#ax10 = fig2.add_subplot(gs[0, 0])
cax10 = plt.imshow(np.abs(heat_map_x-heat_map_pred_x), extent=[0, T, 0, 100], aspect='auto', cmap='cmo.tempo')
cbar10 = fig2.colorbar(cax10)  # Display a colorbar to interpret the color scale
plt.ylim(0,50)
plt.xlim(0,200)
plt.yticks([0, 12.5, 25, 37.5, 50],[-25, -12.5, 0, 12.5, 25])
plt.xticks(ticks=[0, 50, 100, 150, 200])
cbar10.set_label(r'$| u_i - \hat{u}_i |$')

plt.ylabel(r'State Index, $i$')
plt.xlabel(r'Time, $t$')
#ax10.axvline(x=reference_line, color='k', linestyle='--')

#ax9.text(-0.05, -0.1, '(a)', transform=ax9.transAxes, size=14, weight='bold', fontfamily='Arial')
#ax10.text(-0.05, -0.1, '(b)', transform=ax10.transAxes, size=14, weight='bold', fontfamily='Arial')




plt.tight_layout()
plt.savefig("phi_4_error_random", dpi=600)

###############################
################################
##############################
#time history of some dimensions
fig3 = plt.figure(figsize=(8, 6))


gs = fig3.add_gridspec(4, 2)

points_to_plot = 2000

ax11 = fig3.add_subplot(gs[0:2, 0])
ax11.plot(t_arr[:points_to_plot], x_arr[:points_to_plot,4], 'k', label='True')
ax11.plot(t_arr[:points_to_plot], pred_x_arr[:points_to_plot,4], 'r--', label='pred_dxicted')
ax11.set_ylabel(r'$u_{-20}$')

ax12 = fig3.add_subplot(gs[0:2, 1])
ax12.plot(t_arr[:points_to_plot], x_arr[:points_to_plot,14], 'k', label='True')
ax12.plot(t_arr[:points_to_plot], pred_x_arr[:points_to_plot,14], 'r--', label='pred_dxicted')
ax12.set_ylabel(r'$u_{-10}$')

ax13 = fig3.add_subplot(gs[2:4, 0])
ax13.plot(t_arr[:points_to_plot], x_arr[:points_to_plot,24], 'k', label='True')
ax13.plot(t_arr[:points_to_plot], pred_x_arr[:points_to_plot,24], 'r--', label='pred_dxicted')
ax13.set_xlabel(r'Time, $t$')
ax13.set_ylabel(r'$u_{0}$')

ax14 = fig3.add_subplot(gs[2:4, 1])
ax14.plot(t_arr[:points_to_plot], x_arr[:points_to_plot,34], 'k', label='True')
ax14.plot(t_arr[:points_to_plot], pred_x_arr[:points_to_plot,34], 'r--', label='pred_dxicted')
ax14.set_xlabel(r'Time, $t$')
ax14.set_ylabel(r'$u_{10}$')
#ax14.set_yticks([-5,0, 5, 10])

plt.tight_layout()
plt.savefig("phi_4_time_history_random", dpi=600)
