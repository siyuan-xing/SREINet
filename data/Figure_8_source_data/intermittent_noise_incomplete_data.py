import os
import numpy as np
import re
import math


def get_file_paths(main_folder, target_missing=0.00):
    """
    Traverse subfolders of the main folder and process three files to be processed in each subfolder.
    """
    results = []
    for root, dirs, files in os.walk(main_folder):
        for subdir in dirs:
            noise, level, missing = extract_noise_level_missing(subdir)

            if np.abs(missing- target_missing)<1e-10:
                subdir_path = os.path.join(root, subdir)
            
            
                coefficient_ratio = []
                for i in range(1, 4):
                    filename = os.path.join(subdir_path, f'output_{i}.txt')
                    ave_ratio = process_files(filename)
                    coefficient_ratio.append(ave_ratio)
                results.append((noise, level, np.mean(coefficient_ratio)))
    return results


def extract_noise_level_missing(folder_name):
    """
    Extract noise, level and missing values from folder name.
    """
    noise_match = re.search(r"noise_(\d\.\d)", folder_name)
    level_match = re.search(r"level_(\d+\.\d+)", folder_name)
    missing_match = re.search(r"missing_(\d+\.\d+)", folder_name)
    
    if noise_match and level_match and missing_match:
        noise = float(noise_match.group(1))
        level = float(level_match.group(1))
        missing = float(missing_match.group(1))
        return noise, level, missing
    return None, None, None



def process_files(filename):
    real_coeff_first_10_dim = 1.0
    real_coeff_last_10_dim = 2.0  # absolute values, you've checked the sign by hand

    """
    Process coefficients from three files and return the average value.
    Specific processing logic can be added here, currently empty.
    """
    # Add file processing logic here
      # Step 5: Read the file content
    with open(filename, 'r') as file:
        data_string = file.read()

        # Step 6: Extract coefficients from each line using regular expressions
    coefficients = []
    lines = data_string.strip().split('\n')

        # this method will only capture the absolute value of the coefficients, but it is fine.
    for line in lines:
        # Extract the constant term (first number after the equal sign)
        #constant = re.search(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", line).group()
        # Find all floating-point numbers before the 'x' character (coefficients)
        coeffs = re.findall(r"([-+]?\d*\.\d+(?:[eE][-+]?\d+)?)(?=\s*[\*\s]*x)", line)
            
        # Combine the constant term with the rest of the coefficients
        all_coeffs =  [np.abs(float(c)) for c in coeffs]
        coefficients.append(all_coeffs)

        # Step 8: Convert the list of coefficients into a numpy array (matrix)
    coeff_matrix_fist_10_dim = np.concatenate(coefficients[:10])
    coeff_matrix_last_10_dim = np.concatenate(coefficients[10:])
    #pick the four biggest elements from coefficients[10:]
    

        # Step 9: Calculate the accuracy of the coefficients
        # Calculate the relative error for each coefficient and average them
    accuracy_first_10_dim = np.abs(coeff_matrix_fist_10_dim) / real_coeff_first_10_dim
    accuracy_last_10_dim = np.abs(coeff_matrix_last_10_dim) / real_coeff_last_10_dim

    return  np.mean(list(accuracy_first_10_dim) + list(accuracy_last_10_dim))


main_folder = "Incomplete Noise"

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib as mpl

# Set the font to Helvetica
mpl.rcParams['font.family'] = 'Helvetica'

params = {
    'image.origin': 'lower',
    'image.interpolation': 'nearest',
    'image.cmap': 'gray',
    'axes.grid': False,
    'savefig.dpi': 600,  # to adjust notebook inline plot size
    'axes.labelsize': 12, # fontsize for x and y labels (was 10)
    'axes.titlesize': 14,
    'font.size': 12, # was 10
    'legend.fontsize': 12, # was 10
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'font.family': 'Helvetica',
}
import matplotlib
matplotlib.rcParams.update(params)


plt.figure()

fig, axs = plt.subplots(2, 3, gridspec_kw={'height_ratios': [0.12,1]}, figsize=(8, 3.2), facecolor=(1, 1, 1, 0))

set_with_wrong_odes ={0.00:[(2,2),(2,3),(3,2),(4,3)],
                      0.20:[(0,2),(2,2),(3,2),(3,3),(4,2),(4,0)],
                      0.50:[(0,3),(1,0),(1,1),(1,3),(2,1),(2,3),(3,3),(4,2)]
                      }

for i, target_missing in enumerate([0.00, 0.20, 0.50]):

    results = get_file_paths(main_folder, target_missing)
    #print(results)

    data = np.array(results)


    # Extract unique X and Y values to construct 2D matrix
    x_values = np.unique(data[:, 0])
    y_values = np.unique(data[:, 1])
    heatmap_data = np.zeros((len(x_values), len(y_values)))

    # Fill data into heatmap matrix
    for entry in data:
        x_idx = np.where(x_values == entry[0])[0][0]
        y_idx = np.where(y_values == entry[1])[0][0]
        heatmap_data[x_idx, y_idx] = entry[2]

    heatmap_data = heatmap_data[:-1,:]

    
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set_palette('Blues')
    # Plot heatmap without interpolation, flip y-axis, show values as percentages (2 decimal places), white font
    img = axs[1,i].imshow(heatmap_data, cmap='crest', interpolation='none', aspect='auto')
    img.set_clim(0.90, 1.0)  # Adjust color limits after plotting

    #cbar = plt.colorbar(label="Values (%)")
    #cbar.ax.set_yticklabels([f"{int(float(label.get_text()) * 100)}" for label in cbar.ax.get_yticklabels()])

    # Set axis labels as percentages for X and Y
    x_values = x_values[:-1]
    axs[1, i].set_xticks(ticks=np.arange(len(y_values)), labels=[f"{y*100:.0f}%" for y in y_values])
    axs[1, i].set_yticks(ticks=np.arange(len(x_values)), labels=[f"{x*100:.0f}%" for x in x_values])
    #axs[1, i].invert_yaxis()  # Flip y-axis
    axs[1, i].set_title(f"{target_missing * 100:.0f}% Data Missing", fontweight='bold')
    from matplotlib.patches import Rectangle

    # Display each cell value as a percentage with 1 decimal places in white font
    for j in range(heatmap_data.shape[0]):
        for k in range(heatmap_data.shape[1]):
            if  j + k == heatmap_data.shape[1] - 1:
                axs[1, i].text(k, j, f"{(heatmap_data[j, k] * 100) :.1f}", ha='center', va='center', color='white')
            
    for (j,k) in set_with_wrong_odes[target_missing]:
        rect = Rectangle((k - 0.5, j - 0.5), 1, 1, fill=False, hatch='//', edgecolor='black')
        axs[1, i].add_patch(rect)


    # Set border grid lines only, without crossing values
    axs[1, i].set_xticks(np.arange(-0.5, len(y_values)-0.5, 1), minor=True)
    axs[1, i].set_yticks(np.arange(-0.5, len(x_values)-0.5, 1), minor=True)
    axs[1, i].grid(which='minor', color='gray', linestyle='--', linewidth=0.5)
    axs[1, i].tick_params(axis='both', length=0)
    axs[1, i].set_xlabel("Noise Level")
    if i != 0:
        axs[1, i].set_yticklabels([])

axs[1, 0].set_ylabel("Data Pollution Ratio")


cax = fig.add_axes([0.2, 0.90, 0.6, 0.03])  # x-position, y-position, width, height for the color bar

# Remove empty axes in the top row
for ax in axs[0]:
    ax.axis('off')


cbar = plt.colorbar(img, cax=cax, orientation='horizontal')
cbar.set_label("Coefficient Ratio (%)", fontsize=12)
#cbar.ax.set_xticklabels([f"{int(float(label.get_text()) * 100)}" for label in cbar.ax.get_xticklabels()])
cbar.ax.tick_params(labelsize=10)
cbar.set_ticks([0.9, 0.95, 1.0])
cbar.set_ticklabels([90, 95, 100])
cbar.ax.tick_params(direction='in')
cbar.ax.xaxis.set_label_position('top')
#plt.show()
#fig.subplots_adjust(hspace=0.05)  # Change 0.4 to the desired width space

plt.tight_layout()
plt.savefig('intermittent_noise_incomplete_data.png', format='png', dpi=1200)

