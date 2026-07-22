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

FS_SMALL = 10
FS_MEDIUM = 12
FS_LARGE = 14

MY_MODEL = 'abolowitz_ladik'  

# Generate data based on the specified model type
node_num = 64
n= np.arange(-node_num/2, node_num/2)

t0 = -4.0
x0 = 1/np.sqrt(2) * (1-6*(1+1j*t0)/(1+2*n**2+6/4*t0**2))

#plot the magnitude of x0

plt.plot(np.abs(x0))
plt.show()

data_T = 8.0       # Length of the data (T)
data_dt = 0.01       # Resolution of the data (dt)

tf = data_T + t0
myDG = DG.DataGenerator(x0,T=data_T, dt=data_dt)
t_arr, x_train, dx_train, guess_highest_order_polynomial = myDG.generate_dataset_by_model_name(MY_MODEL, node_num, method='BDF')

t_arr = t_arr + t0 #shift the time axis to the original time

x_data = np.concatenate((np.real(x_train), np.imag(x_train)), axis=1)
dx_data = np.concatenate((np.real(dx_train), np.imag(dx_train)), axis=1)



sim_init = np.concatenate((np.real(x0), np.imag(x0)))

# Define the time span for the simulation
t_span = (t0, tf)
t_eval = np.arange(t0, tf, data_dt)

# Solve the ODE system
solution = solve_ivp(AL_ode, t_span, sim_init, t_eval=t_eval,method='BDF')

sol = solution.y.T

pred_amp = np.abs(sol[:,:node_num] + 1j*sol[:,node_num:])
sim_amp = np.abs(x_train)

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
    # Compensate for the downscaling of each 7 x 2 in panel in the 7 x 8 in composite.
    'xtick.labelsize': FS_MEDIUM,
    'ytick.labelsize': FS_MEDIUM,
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

fig = plt.figure(figsize=(7.0, 8.0 / 4.0), facecolor='none', constrained_layout=True)

gs = fig.add_gridspec(5, 5, width_ratios=[20, 3, 20, 3, 1], hspace=0.1, wspace=0.15)

import matplotlib.colors as mcolors
import cmocean

ax5 = fig.add_subplot(gs[:4, 0])
cax5 = ax5.imshow(sim_amp, aspect='auto', cmap='cmo.thermal', extent=[t0, tf, -node_num//2, node_num//2])
ax5.set_xlim(t0, tf)
ax5.set_xticks([])
#ax5.set_ylim(0,node_num)
ax5.set_yticks([-32,0, 32])

ax5.set_ylabel(r'State Index, $j$')
ax5.axvline(x=0, color='white', linestyle='--', linewidth=1.0, alpha=0.5)
#ax5.set_title("Full Simulation")
#x_mid = (plt.xlim()[0] + plt.xlim()[1]) / 2
#ax5.axvline(x=x_mid, color='k', linestyle='--')

ax6 = fig.add_subplot(gs[-1, 0])
ax6.plot(t_arr, sim_amp[node_num//2,:],'k', linewidth=2.0)
ax6.set_xlim(t0,tf)
ax6.set_ylim(0,5)
ax6.set_yticks([0,3])
ax6.set_xticks([-4, 0, 4])
ax6.set_xlabel(r'Time, $t$')
ax6.set_ylabel(r'$|u_{1}|$', rotation=0, labelpad=16)
#ax6.set_yticks([-5, 5])
#reference_line = 25 
#ax6.axvline(x=reference_line, color='k', linestyle='--')


t_sample = int(data_T // data_dt//2)
ax1 = fig.add_subplot(gs[:4, 1])
ax1.plot(sim_amp[:,t_sample] , n ,'k', linewidth=2.0)
ax1.set_xlim()
ax1.set_ylim(-node_num//2,node_num//2)
ax1.set_yticks([])
ax1.set_xticks([])
ax1.xaxis.set_label_position('top')  # Move the label to the top
ax1.set_xlabel(r'$|u_j(0)|$', fontsize=FS_SMALL)


#### 3 and 4
ax7 = fig.add_subplot(gs[0:4, 2])
cax7 = ax7.imshow(pred_amp, aspect='auto', extent=[t0, tf, -node_num//2, node_num//2], cmap='cmo.thermal')
ax7.set_xlim(t0, tf)
ax7.set_xticks([])
#add a vertical dash line at the middle of x, transparaent 0.5
ax7.axvline(x=0, color='white', linestyle='--', linewidth=1.0, alpha=0.5)

#ax7.set_ylim(0, node_num)
ax7.set_yticks([])
#ax7.set_title("Identified System")

#ax3.set_ylabel('State Index')
#add a vertical dash line at the middle of x
#x_mid = (plt.xlim()[0] + plt.xlim()[1]) / 2
#ax7.axvline(x=x_mid, color='k', linestyle='--')

t_sample = int(data_T // data_dt//2)
ax2 = fig.add_subplot(gs[:4, 3])
ax2.plot(pred_amp[:,t_sample] , n ,'k', linewidth=2.0)
ax2.set_xlim()
ax2.set_ylim(-node_num//2,node_num//2)
ax2.set_yticks([])
ax2.set_xticks([])
ax2.xaxis.set_label_position('top')  # Move the label to the top
ax2.set_xlabel(r'$|u_j(0)|$', fontsize=FS_SMALL)

ax8 = fig.add_subplot(gs[-1, 2])

ax8.plot(t_arr, pred_amp[node_num//2,:],'k', linewidth=2.0)
ax8.set_xlim(t0, tf)
ax8.set_ylim(0, 5)
ax8.set_xlabel(r'Time, $t$')
#ax4.set_ylabel(r'Derivative, $\dot{x}_1$')
ax8.set_xticks([-4, 0, 4])
ax8.set_yticks([])
#add a vertical dash line at the middle of x axis
#ax8.axvline(x=reference_line, color='k', linestyle='--')

#ax6.text(-0.05, -0.8, '(c)', transform=ax6.transAxes, size=14, weight='bold')
#ax8.text(-0.05, -0.8, '(d)', transform=ax8.transAxes, size=14, weight='bold')

cbar_ax = fig.add_subplot(gs[:4, 4])
cbar = fig.colorbar(cax5, ax=ax5, orientation='vertical', cax=cbar_ax)
cbar.set_label(r'$| u_j |$')


# Adjust the X label position for ax6
#ax6.xaxis.set_label_coords(0.5, -0.7)  # Move the label slightly up; adjust y-value as needed

# Adjust the X label position for ax8
#ax8.xaxis.set_label_coords(0.5, -0.7)  # Move the label slightly up; adjust y-value as needed
#plt.tight_layout()
plt.savefig("AL_peregrine_simulation_vs_infer", dpi=600)


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
    'text.latex.preamble': r'\usepackage{amsmath}\usepackage{helvet}\renewcommand{\familydefault}{\sfdefault}',  # Load Helvetica for LaTeX
    'axes.labelweight': 'bold',
    'font.family': 'Arial',

}
matplotlib.rcParams.update(params)

fig2 = plt.figure(figsize=(8, 4))

cax9 = plt.imshow(np.abs(pred_amp-sim_amp), extent=[t0, tf, 0, node_num], aspect='auto', cmap='cmo.tempo')
cbar9=fig2.colorbar(cax9)  # Display a colorbar to interpret the color scale
plt.ylim(0,64)
plt.yticks([0,16,32,48, 64],[-32,-16,0,16,32])
plt.xticks([-4,-2, 0, 2, 4])
plt.ylabel(r'State Index, $j$')
plt.xlabel(r'Time, $t$')
#plt.axvline(x=60, color='k', linestyle='--')

#set colobar label
cbar9.set_label(r'$| u_j - \hat{u}_j |$')

plt.tight_layout()
plt.savefig("AL_peregrine_error", dpi=1200)

###############################
################################
##############################
#time history of some dimensions
fig3 = plt.figure(figsize=(8, 6))


gs = fig3.add_gridspec(4, 2)

points_to_plot = -1

ax11 = fig3.add_subplot(gs[0:2, 0])
ax11.plot(t_arr[:points_to_plot], sim_amp[2,:points_to_plot], 'k', label='True')
ax11.plot(t_arr[:points_to_plot], pred_amp[2,:points_to_plot], 'r--', label='pred_dxicted')
ax11.set_ylabel(r'$| u_{-30} |$')
ax11.set_xticks([-4, -2, 0, 2, 4])
ax11.set_ylim(0.6, 0.8)
#reference_line = 60 
#ax11.axvline(x=reference_line, color='k', linestyle='--')


ax12 = fig3.add_subplot(gs[0:2, 1])
ax12.plot(t_arr[:points_to_plot], sim_amp[17,:points_to_plot], 'k', label='True')
ax12.plot(t_arr[:points_to_plot], pred_amp[17,:points_to_plot], 'r--', label='pred_dxicted')
ax12.set_ylabel(r'$| u_{-15} |$')
ax12.set_xticks([-4, -2, 0, 2, 4])
ax12.set_ylim(0.6, 0.8)
#ax12.axvline(x=reference_line, color='k', linestyle='--')


ax13 = fig3.add_subplot(gs[2:4, 0])
ax13.plot(t_arr[:points_to_plot], sim_amp[32,:points_to_plot], 'k', label='True')
ax13.plot(t_arr[:points_to_plot], pred_amp[32,:points_to_plot], 'r--', label='pred_dxicted')
ax13.set_xlabel(r'Time, $t$')
ax13.set_ylabel(r'$| u_{0} |$')
ax13.set_xticks([-4, -2, 0, 2, 4])
#ax13.axvline(x=reference_line, color='k', linestyle='--')


ax14 = fig3.add_subplot(gs[2:4, 1])
ax14.plot(t_arr[:points_to_plot], sim_amp[47,:points_to_plot], 'k', label='True')
ax14.plot(t_arr[:points_to_plot], pred_amp[47,:points_to_plot], 'r--', label='pred_dxicted')
ax14.set_xlabel(r'Time, $t$')
ax14.set_ylabel(r'$| u_{15} |$')
#ax14.set_yticks([-5,0, 5, 10])
ax14.set_xticks([-4, -2, 0, 2, 4])
ax14.set_ylim(0.6, 0.8)
#ax14.axvline(x=reference_line, color='k', linestyle='--')

plt.tight_layout()
plt.savefig("AL_peregrine_time_history", dpi=1200)
