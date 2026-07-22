import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.io import loadmat
from scipy.signal import savgol_filter

FS_SMALL = 10
FS_MEDIUM = 12
FS_LARGE = 14

def load_triple_pendulum_data(mat_path, downsample=1):
    """
    Loads triple pendulum data from a MATLAB .mat file and returns the relevant arrays, with optional downsampling.
    Args:
        mat_path (str): Path to the .mat file.
        downsample (int): Take every nth sample (default 1, i.e., no downsampling)
    Returns:
        my_data (np.ndarray): Concatenated array of [theta1, theta2, theta3, d_theta1, d_theta2, d_theta3], shape (N, 6)
        d_theta (np.ndarray): Concatenated array of [d_theta1, d_theta2, d_theta3], shape (N, 3)
    """
    data = loadmat(mat_path)
    theta1 = data['Theta1'][::downsample]
    theta2 = data['Theta2'][::downsample]
    theta3 = data['Theta3'][::downsample]
    d_theta1 = data['dTheta1'][::downsample]
    d_theta2 = data['dTheta2'][::downsample]
    d_theta3 = data['dTheta3'][::downsample]

    my_data = np.concatenate([theta1, theta2, theta3, d_theta1, d_theta2, d_theta3], axis=1)
    d_theta = np.concatenate([d_theta1, d_theta2, d_theta3], axis=1)

    return my_data, d_theta

bright_palette = sns.color_palette("Set2")  # Can also use any of the custom color palettes above

# === Basic Settings ===
sns.set_theme(context='paper',  # 'talk' for slides, 'paper' for publication
              style='white',    # White background
              palette=bright_palette,  # Colorblind-friendly color scheme
              font='Helvetica'
              )   # Adjust overall font size

plt.rcParams.update({
    'font.family': 'Helvetica',
    'font.size': FS_SMALL,
    'axes.labelsize': FS_SMALL,
    'xtick.labelsize': FS_SMALL,
    'ytick.labelsize': FS_SMALL,
})

palette = sns.color_palette("Set2")
my_colors = [palette[i] for i in [4, 2, 0, 6]]  # e.g., cyan-blue-purple-apple green-gray-brown
sns.set_palette(my_colors)

# === Load Data ===
data_1, d_theta_1 = load_triple_pendulum_data('TriplePendulum_Data/TripleDataFreeSwing_1_Dt_0_0001.mat', downsample=100)

# Set time parameters
dt = 0.0001 * 100  # Original dt multiplied by downsample factor
time_steps = np.arange(len(data_1)) * dt

# Extract theta_3, theta_3_dot
theta_3 = data_1[:, 2]  # Column 3 is theta3
theta_3_dot = data_1[:, 5]  # Column 6 is d_theta3

# Calculate theta_3_ddot using Savitzky-Golay filter
window_length = 7
polyorder = 2
theta_3_ddot = savgol_filter(theta_3_dot, window_length, polyorder, delta=dt, deriv=1, axis=0)

# === Plotting ===
fig, axes = plt.subplots(3, 1, figsize=(7.0 / 3.0, 3.5))

# Plot theta_3
axes[0].plot(time_steps, theta_3, linewidth=1, color=my_colors[0])
axes[0].set_ylabel(r'$\theta_3$ (rad)', fontsize=FS_SMALL)
axes[0].set_yticks([2, 3.2, 4.4])  # Set specific y-axis ticks
axes[0].tick_params(axis='both', which='major', labelsize=FS_SMALL)  # Add ticks
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# Plot theta_3_dot
axes[1].plot(time_steps, theta_3_dot, linewidth=1, color=my_colors[1])
axes[1].set_ylabel(r'$\dot{\theta}_3$ (rad/s)', fontsize=FS_SMALL)
axes[1].tick_params(axis='both', which='major', labelsize=FS_SMALL)  # Add ticks
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

# Plot theta_3_ddot
axes[2].plot(time_steps, theta_3_ddot, linewidth=1, color=my_colors[2])
axes[2].set_xlabel('Time (s)', fontsize=FS_SMALL)
axes[2].set_ylabel(r'$\ddot{\theta}_3$ (rad/s$^2$)', fontsize=FS_SMALL)
axes[2].tick_params(axis='both', which='major', labelsize=FS_SMALL)  # Add ticks
axes[2].spines['top'].set_visible(False)
axes[2].spines['right'].set_visible(False)

# === Ensure Publication Quality ===
plt.tight_layout()
plt.savefig("triple_pendulum_theta3_plots.png", dpi=600)  # Recommended to export as vector PDF
plt.close(fig)
