import sys
import os
import numpy as np

sys.path.append(os.path.abspath('../../utilities/'))

import Model_zoo

import DataGenerator as DG
import re
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Step 1: Read the ODEs from the text file and parse them
def parse_odes_from_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    ode_list = []
    for line in lines:
        # Remove (xN)'= from the beginning and '^' to '**' for exponentiation
        line = re.sub(r"\(\w+\)'=", "", line).strip().replace("^", "**")
        ode_list.append(line)

    return ode_list

# Step 2: Generate the ODE function from the parsed equations
def generate_ode_function(ode_list):
    def ode_system(t, x):
        # Create a dictionary to map variable names x1, x2, ..., x20 to x[0], x[1], ..., x[19]
        var_map = {f'x{i+1}': x[i] for i in range(20)}
        
        dxdt = [0] * 20
        
        # Evaluate each ODE using the var_map
        for i, ode in enumerate(ode_list):
            # Evaluate the right-hand side of each ODE expression
            dxdt[i] = eval(ode, {}, var_map)  # Pass var_map as the local variable dictionary
        
        return dxdt

    return ode_system

# Step 3: Simulate the ODE system using solve_ivp
def simulate_ode(ode_function, initial_conditions, t_span, t_eval):
    # Solve the system using solve_ivp
    sol = solve_ivp(ode_function, t_span, initial_conditions, t_eval=t_eval, method='DOP853')
    return sol


MY_MODEL = 'discrete_phi_quartic' # discrete sine Gordon model 

# Generate data based on the specified model type
node_num = 10
C = 2.0 #coupling strength

#initial conditions for Simulation
xn = np.linspace(-node_num/2/np.sqrt(C), node_num/2/np.sqrt(C), node_num)

#x0 = np.tanh(xn/np.sqrt(1-v**2))  
num_traj = 1
v = np.linspace(0.0, 0.99, num_traj)
x0 =  np.random.normal(0, 0.1, size=xn.shape) 
d_x0 = np.random.normal(0, 0.1, size=xn.shape) 
#d_x0 = 1/np.sqrt(1-v[i]**2) * (1 - np.tanh(xn/np.sqrt(1-v[i]**2))**2)
new_IC = np.concatenate((x0, d_x0), axis=0) # add velocity
 #data_initial_conditions.append(new_IC)

new_IC = [-0.06828551, -0.18244346,  0.09761284, -0.1241317,  -0.01717222,  0.02260519,
  0.09351957, -0.05316819, -0.11128258, -0.03881156, -0.06591778, -0.07887937,
  0.05126925, -0.06826662,  0.04954326, -0.05551705,  0.23267888, -0.09298385,
 -0.24814171, -0.06788285]
new_IC = np.array(new_IC)
print(new_IC)

data_T =  23.0       # Length of the data (T)
data_dt = 0.01       # Resolution of the data (dt)


file_list=['./Continuous Gaussian Noise/noise_level_1_percent/output_1.txt',
            './Continuous Gaussian Noise/noise_level_5_percent/output_2.txt',
            './Continuous Gaussian Noise/noise_level_10_percent/output_1.txt',
            './Continuous Gaussian Noise/noise_level_15_percent/output_1.txt']

fig, axs = plt.subplots(1, 4, figsize=(7.0, 2.1), facecolor=(1, 1, 1, 0))

for i, ode_file in enumerate(file_list):
    #axs[i].set_facecolor('white')

    #extract noise level
    noise_match = re.search(r'level_(\d+)_', ode_file)
    noise_match = float(noise_match.group(1))
    noise_level = noise_match/100

    myDG_no_noise = DG.DataGenerator([new_IC],T=data_T, dt=data_dt)
    myDG_noise = DG.DataGenerator([new_IC],T=data_T, dt=data_dt, noise_level=noise_level)

    t_arr, x_train, dx_train, guess_highest_order_polynomial = myDG_no_noise.generate_dataset_by_model_name(MY_MODEL, C, node_num,method='DOP853')

    t_arr, x_train_noise, dx_train_noise, guess_highest_order_polynomial = myDG_noise.generate_dataset_by_model_name(MY_MODEL, C, node_num,method='DOP853')

    node_to_plot = 5


    import matplotlib.pyplot as plt

    params = {
        'image.origin': 'lower',
        'image.interpolation': 'nearest',
        'image.cmap': 'gray',
        'axes.grid': False,
        'axes.labelsize': 12, # fontsize for x and y labels (was 10)
        'axes.titlesize': 12,
        'axes.titleweight': 'normal',
        'axes.labelweight': 'normal',
        'font.size': 10,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        #'text.usetex': True,
        'font.family': 'Helvetica',
    }
    import matplotlib
    import cmocean
    matplotlib.rcParams.update(params)



    ode_list = parse_odes_from_file(ode_file)

    # Step 2: Generate the ODE function from the parsed ODEs
    ode_function = generate_ode_function(ode_list)

    # Step 3: Set initial conditions and time span for the simulation
    x0 = new_IC  # Initial conditions (all zeros for this example)
    t_span = (0, data_T)  # Time range to simulate from 0 to 50
    t_eval = np.linspace(t_span[0], t_span[1], int(data_T//data_dt))  # Time points for the simulation

    # Step 4: Simulate the ODE system
    solution = simulate_ode(ode_function, x0, t_span, t_eval)

    # Step 5: Plot the results
    #axs[i].plot(t_arr, x_train[:,node_to_plot], color='black', linewidth=2.0, label='Clean data')
    #axs[i].plot(t_arr, x_train_noise[:,node_to_plot], alpha=0.3, linewidth=1.5, label='Noisy data')

    #axs[i].plot(t_eval, solution.y[node_to_plot,:], color='red', linestyle="--", linewidth=1.2, label='Identified')
    ave_x_train = np.mean(x_train[:,:node_num], axis=1)
    ave_dx_train = np.mean(x_train[:,node_num:], axis=1)
    ave_x_train_noise = np.mean(x_train_noise[:,:node_num], axis=1)
    ave_dx_train_noise = np.mean(x_train_noise[:,node_num:], axis=1)

    axs[i].plot(ave_x_train, ave_dx_train, color='black', linewidth=2.0, label='Clean data')
    axs[i].plot(ave_x_train_noise, ave_dx_train_noise, alpha=0.3, linewidth=2.5, label='Noisy data')
    
    ave_y = np.mean(solution.y[:node_num,:],axis=0)
    ave_dy = np.mean(solution.y[node_num:,:],axis=0)
    axs[i].plot(ave_y, ave_dy, color='red', linestyle="--", linewidth=1.0, label='Identified')
    axs[i].set_xlim(-1.2, 1.2)
    axs[i].set_ylim(-1.2, 1.2)
    
    #plt.legend()
    # This four-panel strip spans the full two-column width in Fig. 10(c).
    # Use the large label size so its final rendered x labels match panels (a,b).
    axs[i].set_xlabel(r'State, $\bar{x}$', fontsize=12)
    #axs[i].set_yticks([-1.5, -0.75, 0, 0.75, 1.5])
    axs[i].set_yticklabels([])
    #axs[i].set_xticks([0,15,30])
    axs[i].tick_params(direction='in')
    axs[i].set_title(f'{int(noise_match)}% Noise')

    if i==0:
        axs[i].set_yticks([-0.75, 0, 0.75])
        axs[i].set_ylabel(r'$\dot{\bar{x}}$', rotation=0)

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
plt.tight_layout(rect=(0, 0, 1, 0.82))
plt.savefig('continuous_gaussian_varying_nl_state_space.png', format='png', dpi=600)
