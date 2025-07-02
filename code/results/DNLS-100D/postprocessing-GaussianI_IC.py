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
C = 2.0 #coupling strength
xn = np.linspace(-node_num/2/np.sqrt(C), node_num/2/np.sqrt(C), node_num)

#x0 = np.exp(-xn**2) + np.zeros(node_num)*1j # zero imaginary part

x0 = np.exp(-xn**2/2.0/2.0**2) + np.zeros(node_num)*1j # zero imaginary part


data_T =  60.0 * 3       # Length of the data (T)
data_dt = 0.01       # Resolution of the data (dt)
myDG = DG.DataGenerator(x0,T=data_T, dt=data_dt)
t_arr, x_train, dx_train, guess_highest_order_polynomial = myDG.generate_dataset_by_model_name(MY_MODEL, C, node_num)


#train_data = scipy.io.loadmat('train_data.mat')
#val_data = scipy.io.loadmat('validation_data.mat')

#extract the data from the fourth sheet
#train_data = train_data['data']
#val_data = val_data['data']

# data (validation)
x_data = np.concatenate((np.real(x_train), np.imag(x_train)), axis=1)
dx_data = np.concatenate((np.real(dx_train), np.imag(dx_train)), axis=1)

sim_init = np.concatenate((np.real(x0), np.imag(x0)), axis=0)

# Define the time span for the simulation
t_span = (0, data_T)
t_eval = np.arange(0, data_T, data_dt)

# Solve the ODE system
solution = solve_ivp(DNLS_ode, t_span, sim_init, t_eval=t_eval)

sol = solution.y.T

pred_amp = np.abs(sol[:,:node_num] + 1j*sol[:,node_num:])
sim_amp = np.abs(x_train)



params = {
    'image.origin': 'lower',
    'image.interpolation': 'nearest',
    'image.cmap': 'gray',
    'axes.grid': False,
    'savefig.dpi': 600,  # to adjust notebook inline plot size
    'axes.labelsize': 14, # fontsize for x and y labels (was 10)
    'axes.titlesize': 14,
    'font.size': 12, # was 10
    'legend.fontsize': 12, # was 10
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
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


t_f= 30
cax5 = ax5.imshow(sim_amp, aspect='auto', cmap='magma', extent=[0, data_T, 0, node_num])
ax5.set_xlim(0,t_f)
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
ax6.set_xlim(0,t_f)
ax6.set_ylim(0,1.5)
ax6.set_xticks([0, 10, 20, 30])
ax6.set_ylabel(r'$|u_{0}|$',rotation=0)
ax6.yaxis.set_label_coords(-0.15, 0.3)
##ax6.set_xlabel(r'Time, $t$')
#ax6.set_yticks([-5, 5])
#reference_line = 25 
#ax6.axvline(x=reference_line, color='k', linestyle='--')


#### 3 and 4
ax7 = fig.add_subplot(gs[0:-1, 1])
ax7.set_position([ax7.get_position().x0, ax7.get_position().y0 + 0.02, ax7.get_position().width, ax7.get_position().height])

cax7 = ax7.imshow(pred_amp, aspect='auto', extent=[0, data_T, 0, node_num], cmap='magma')
ax7.set_xlim(0,t_f)
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
ax8.set_xlim(0,t_f)
ax8.set_ylim(0, 1.5)
#ax8.set_xlabel(r'Time, $t$')
#ax4.set_ylabel(r'Derivative, $\dot{x}_1$')
ax8.set_xticks([0, 10, 20, 30])
ax8.set_yticks([])
#add a vertical dash line at the middle of x axis
#ax8.axvline(x=reference_line, color='k', linestyle='--')

#ax6.text(-0.05, -0.8, '(c)', transform=ax6.transAxes, size=14, weight='bold')
#ax8.text(-0.05, -0.8, '(d)', transform=ax8.transAxes, size=14, weight='bold')

cbar_ax = fig.add_subplot(gs[0:-1, 2])
cbar_ax.set_position([cbar_ax.get_position().x0, cbar_ax.get_position().y0 + 0.02, cbar_ax.get_position().width, cbar_ax.get_position().height])
cbar = fig.colorbar(cax5, ax=ax5, orientation='vertical', cax=cbar_ax)
cbar.set_label(r'$| u_j |$')


# Adjust the X label position for ax6
ax6.xaxis.set_label_coords(0.5, -0.7)  # Move the label slightly up; adjust y-value as needed

# Adjust the X label position for ax8
ax8.xaxis.set_label_coords(0.5, -0.7)  # Move the label slightly up; adjust y-value as needed

plt.savefig("DNLS_simulation_vs_infer_Gaussian", dpi=600)


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
    'text.usetex': True,
    #'figure.figsize': [7, 4],
    'font.family': 'Helvetica',
    #'text.latex.preamble': r'\usepackage{amsmath}',  # Optional, for advanced math symbols
    #'axes.labelweight': 'bold'
}
matplotlib.rcParams.update(params)

fig2 = plt.figure(figsize=(8, 4))

cax9 = plt.imshow(np.abs(pred_amp-sim_amp), extent=[0, data_T, 0, node_num], aspect='auto', cmap='magma')
cbar9=fig2.colorbar(cax9)  # Display a colorbar to interpret the color scale
plt.ylim(0,30)
plt.yticks([0, 12.5, 25, 37.5, 50], [-25, -12.5, 0, 12.5, 25])
#plt.set_yticklabels([-25, 0, 25])
plt.xticks([0, 60, 120, 180])
plt.ylabel(r'State Index, $i$')
plt.xlabel(r'Time, $t$')
#plt.axvline(x=60, color='k', linestyle='--')

#set colobar label
cbar9.set_label(r'$| u_i - \hat{u}_i |$')

plt.tight_layout()
plt.savefig("DNLS_error_gaussian", dpi=1200)

###############################
################################
##############################
#time history of some dimensions
fig3 = plt.figure(figsize=(8, 6))


gs = fig3.add_gridspec(4, 2)

points_to_plot = -1

ax11 = fig3.add_subplot(gs[0:2, 0])
ax11.plot(t_arr[:points_to_plot], np.abs(sim_amp[4,:points_to_plot]), 'k', label='True')
ax11.plot(t_arr[:points_to_plot], np.abs(pred_amp[4,:points_to_plot]), 'r--', label='pred_dxicted')
ax11.set_ylabel(r'$| u_{-20} |$')
ax11.set_xticks([0,60,120, 180])
reference_line = 60 
#ax11.axvline(x=reference_line, color='k', linestyle='--')


ax12 = fig3.add_subplot(gs[0:2, 1])
ax12.plot(t_arr[:points_to_plot], np.abs(sim_amp[14,:points_to_plot]), 'k', label='True')
ax12.plot(t_arr[:points_to_plot], np.abs(pred_amp[14,:points_to_plot]), 'r--', label='pred_dxicted')
ax12.set_ylabel(r'$| u_{-10} |$')
ax12.set_xticks([0,60,120, 180])
#ax12.axvline(x=reference_line, color='k', linestyle='--')


ax13 = fig3.add_subplot(gs[2:4, 0])
ax13.plot(t_arr[:points_to_plot], np.abs(sim_amp[24,:points_to_plot]), 'k', label='True')
ax13.plot(t_arr[:points_to_plot], np.abs(pred_amp[24,:points_to_plot]), 'r--', label='pred_dxicted')
ax13.set_xlabel(r'Time, $t$')
ax13.set_ylabel(r'$| u_{0} |$')
ax13.set_xticks([0,10,20, 30])
#ax13.axvline(x=reference_line, color='k', linestyle='--')


ax14 = fig3.add_subplot(gs[2:4, 1])
ax14.plot(t_arr[:points_to_plot], np.abs(sim_amp[34,:points_to_plot]), 'k', label='True')
ax14.plot(t_arr[:points_to_plot], np.abs(pred_amp[34,:points_to_plot]), 'r--', label='pred_dxicted')
ax14.set_xlabel(r'Time, $t$')
ax14.set_ylabel(r'$| u_{10} |$')
#ax14.set_yticks([-5,0, 5, 10])
ax14.set_xticks([0,10,120, 180])
#ax14.axvline(x=reference_line, color='k', linestyle='--')

plt.tight_layout()
plt.savefig("DNLS_time_history_gaussian", dpi=1200)

#save data to mat file
import scipy.io
scipy.io.savemat('DNLS_data_gaussian.mat', {'t_arr': t_arr, 'sim_amp': sim_amp, 'pred_amp': pred_amp})

