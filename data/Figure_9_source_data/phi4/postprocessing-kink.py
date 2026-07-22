import pandas as pd
import ast
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tempfile import gettempdir
from fontTools.ttLib import TTFont
from matplotlib.font_manager import FontProperties

#read matrix from train_data.mat
import scipy.io
import numpy as np

from scipy.integrate import solve_ivp

from phi_ode import phi4_ode

import DataGenerator as DG

FS_SMALL = 10
FS_MEDIUM = 12
FS_LARGE = 14
# Specify the model type
MY_MODEL = 'discrete_phi_quartic' # discrete sine Gordon model 

# Generate data based on the specified model type
node_num = 50
C = 2.0 #coupling strength

#initial conditions for Simulation
xn = np.linspace(-node_num/2/np.sqrt(C), node_num/2/np.sqrt(C), node_num)

v=0.0
x0 =  np.tanh(xn/np.sqrt(1-v**2))
d_x0 = 1/np.sqrt(1-v**2) * (1 - np.tanh(xn/np.sqrt(1-v**2))**2)
data_initial_conditions = np.concatenate((x0, d_x0), axis=0) # add velocity

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
    'axes.labelsize': FS_MEDIUM,
    'axes.titlesize': FS_MEDIUM,
    'axes.titleweight': 'bold',
    'axes.labelweight': 'normal',
    'font.size': FS_SMALL,
    'legend.fontsize': FS_SMALL,
    # Compensate for the downscaling of each 7 x 2 in panel in the 7 x 8 in composite.
    'xtick.labelsize': FS_MEDIUM,
    'ytick.labelsize': FS_MEDIUM,
    #'figure.figsize': [7, 4],
    'font.family': 'Helvetica',
    'text.latex.preamble': r'\usepackage{amsmath}',  # Optional, for advanced math symbols
    }

import matplotlib

import cmocean
matplotlib.rcParams.update(params)

# Matplotlib registers only the regular face from macOS's Helvetica TTC.
# Extract face 1 (Helvetica Bold) to a temporary standalone font so titles
# render with the actual bold face rather than synthetic or fallback bold.
helvetica_collection = Path('/System/Library/Fonts/Helvetica.ttc')
helvetica_bold_path = Path(gettempdir()) / 'sreinet_helvetica_bold.ttf'
if helvetica_collection.exists():
    if not helvetica_bold_path.exists():
        helvetica_bold_font = TTFont(str(helvetica_collection), fontNumber=1)
        helvetica_bold_font.save(str(helvetica_bold_path))
        helvetica_bold_font.close()
    title_font = FontProperties(fname=str(helvetica_bold_path), size=params['axes.titlesize'])
else:
    title_font = FontProperties(family='Helvetica', weight='bold', size=params['axes.titlesize'])


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

fig = plt.figure(figsize=(7.0, 8.0 / 4.0), facecolor='none', constrained_layout=True)

gs = fig.add_gridspec(5, 3, width_ratios=[20, 20, 1], hspace=0.1, wspace=0.2)

#ax2 line thickness set to 2
#plt.rcParams['text.usetex'] = True  # Uncomment if you have LaTeX installed
#plt.rcParams['font.family'] = 'Helvetica'
#plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'  # Optional, for advanced math symbols
#plt.rcParams['axes.labelweight'] = 'bold'

import matplotlib.colors as mcolors


ax5 = fig.add_subplot(gs[0:-1, 0])
plot_until_points = 1600
cax5 = ax5.imshow(heat_map_x[:,:plot_until_points], aspect='auto', extent=[0,80, 0, 50], cmap='cmo.deep')

ax5.set_xlim(0,45)
ax5.set_xticks([])
ax5.set_ylim(0,50)
ax5.set_yticks([0, 25, 50],[-25,0,25])
ax5.set_title("Ground Truth", fontproperties=title_font)

ax5.set_ylabel(r'State Index, $i$')
#add a vertical dash line at the middle of x
#x_mid = (plt.xlim()[0] + plt.xlim()[1]) / 2
#ax5.axvline(x=x_mid, color='k', linestyle='--')

ax6 = fig.add_subplot(gs[-1, 0])
ax6.plot(heatmap_t_arr[:plot_until_points], heat_map_x[16][:plot_until_points],'k', linewidth=2.0)
ax6.set_xlim(0,45)
ax6.set_ylim(-5,5)
#ax6.set_xlabel(r'Time, $t$', labelpad=-1)
ax6.set_ylabel(r'$u_{-9}$', rotation=0, labelpad=16)
#rotate y axis label by 90 degree


ax6.set_xticks([0, 22.5, 45])
ax6.set_yticks([0])
#add a vertical dash line at the middle of x axis

#reference_line = 25 
#ax6.axvline(x=reference_line, color='k', linestyle='--')

#### 3 and 4
ax7 = fig.add_subplot(gs[0:-1, 1])
cax7 = ax7.imshow(heat_map_pred_x[:,:plot_until_points], aspect='auto', extent=[0,80, 0, 50], cmap='cmo.deep')

ax7.set_xlim(0,45)
ax7.set_xticks([])
ax7.set_ylim(0,50)
ax7.set_yticks([])
#ax7.set_yticks([0, 25, 50])
ax7.set_title("SREINet-Discovered\nSystem", fontproperties=title_font)

#ax3.set_ylabel('State Index')
#add a vertical dash line at the middle of x
#x_mid = (plt.xlim()[0] + plt.xlim()[1]) / 2
#ax7.axvline(x=x_mid, color='k', linestyle='--')

ax8 = fig.add_subplot(gs[-1, 1])

ax8.plot(heatmap_t_arr[:plot_until_points], heat_map_pred_x[16][:plot_until_points],'k', linewidth=2.0)
ax8.set_xlim(0,45)
ax8.set_ylim(-5,5)
#ax8.set_xlabel(r'Time, $t$', labelpad=-1)
#ax4.set_ylabel(r'Derivative, $\dot{x}_1$')
ax8.set_xticks([0, 22.5, 45])
ax8.set_yticks([])
#ax8.set_yticks([-2, 2])
#add a vertical dash line at the middle of x axis
#ax8.axvline(x=reference_line, color='k', linestyle='--')

#ax6.text(-0.05, -0.8, '(c)', transform=ax6.transAxes, size=14, weight='bold')
#ax8.text(-0.05, -0.8, '(d)', transform=ax8.transAxes, size=14, weight='bold')

cbar_ax = fig.add_subplot(gs[0:-1, 2])
cbar = fig.colorbar(cax5, ax=ax5, orientation='vertical', cax=cbar_ax)
cbar.set_label(r'$u_i$')

#plt.tight_layout()
plt.savefig("phi_4_simulation_vs_infer", dpi=600)


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
    'font.family': 'Arial',
    'text.latex.preamble': r'\usepackage{amsmath}',  # Optional, for advanced math symbols
    'axes.labelweight': 'bold'
}
matplotlib.rcParams.update(params)

fig2 = plt.figure(figsize=(8, 4))


"""
gs = fig2.add_gridspec(2, 1)

ax9 = fig2.add_subplot(gs[1, 0])
cax9 = ax9.imshow(np.abs(heat_map_dx-heat_map_pred_dx), extent=[0, T, 0, 100], aspect='auto', cmap='cmo.tempo')
cbar9=fig2.colorbar(cax9)  # Display a colorbar to interpret the color scale
ax9.set_ylim(50,100)
ax9.set_xlim(0,200)
ax9.set_yticks(ticks=[50,62.5,75, 87.5, 100])
ax9.set_yticklabels([0,12.5,25,37.5,50])
ax9.set_xticks(ticks=[0,50, 100, 150, 200])
ax9.set_ylabel(r'State Index, $i$')
#ax9.axvline(x=reference_line, color='k', linestyle='--')
#set colobar label
cbar9.set_label(r'$| \dot{u}_i - \dot{\hat{u}}_i |$')

ax10 = fig2.add_subplot(gs[0, 0])
"""
cax10 = plt.imshow(np.abs(heat_map_x-heat_map_pred_x), extent=[0, T, 0, 100], aspect='auto', cmap='cmo.tempo')
cbar10 = fig2.colorbar(cax10)  # Display a colorbar to interpret the color scale
plt.ylim(0,50)
plt.xlim(0,200)
plt.yticks([0, 12.5, 25, 37.5, 50],[-25,-12.5,0, 12.5, 25])
plt.xticks(ticks=[0, 50, 100, 150, 200])
cbar10.set_label(r'$| u_i - \hat{u}_i |$')

plt.ylabel(r'State Index, $i$')
plt.xlabel(r'Time, $t$')
#ax10.axvline(x=reference_line, color='k', linestyle='--')

#ax9.text(-0.05, -0.1, '(a)', transform=ax9.transAxes, size=14, weight='bold', fontfamily='Arial')
#ax10.text(-0.05, -0.1, '(b)', transform=ax10.transAxes, size=14, weight='bold', fontfamily='Arial')




plt.tight_layout()
plt.savefig("phi_4_error_kink", dpi=600)

###############################
################################
##############################
#time history of some dimensions
fig3 = plt.figure(figsize=(8, 6))


gs = fig3.add_gridspec(4, 2)

points_to_plot = 4000

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
plt.savefig("phi_4_time_history_kink", dpi=600)
