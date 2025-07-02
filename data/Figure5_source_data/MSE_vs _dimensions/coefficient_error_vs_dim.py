import os
import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# Step 1: Define the directory paths and corresponding noise levels
node_number = ['10 nodes', '20 nodes', '30 nodes', '40 nodes', '50 nodes']
base_directory = './'  # Base directory where the noise folders are located
file_count = 1  # Number of files to process per folder (e.g., 10 files with incremental index)

# Step 2: Initialize an empty list to store MSEs for each noise level
all_mse_errors = []

# Define the real coefficients (absolute values)
real_coeff_first_half_dim = 1.0
real_coeff_second_half_dim = 2.0  # absolute values

# Step 3: Loop through each noise level directory
for node_num in node_number:
    mse_errors = []
    directory = os.path.join(base_directory, node_num)
    filename = f'{directory}/output_{1}.txt'
    
    with open(filename, 'r') as file:
        data_string = file.read()

    # Step 6: Extract coefficients from each line using regular expressions
    coefficients = []
    lines = data_string.strip().split('\n')
    
    for line in lines:
        coeffs = re.findall(r"([-+]?\d*\.\d+(?:[eE][-+]?\d+)?)(?=\s*[\*\s]*x)", line)
        all_coeffs = [float(c) for c in coeffs]
        coefficients.append(all_coeffs)

    num = int(re.search(r'\d+', node_num).group())

    coeff_matrix_first_half_dim = np.array(coefficients[:num])
    max_len = max(len(lst) for lst in coefficients[num:])
    coeff_matrix_second_half_dim = np.array([lst + [np.nan] * (max_len - len(lst)) for lst in coefficients[num:]])

    mse_first_half_dim = np.sqrt((np.abs(coeff_matrix_first_half_dim) - real_coeff_first_half_dim) ** 2).flatten()
    mse_second_half_dim = np.sqrt((np.abs(coeff_matrix_second_half_dim) - real_coeff_second_half_dim) ** 2).flatten()
    mse_second_half_dim = mse_second_half_dim[~np.isnan(mse_second_half_dim)]

    mse_errors += list(mse_first_half_dim) + list(mse_second_half_dim)
    all_mse_errors.append(mse_errors)

node_number = ['20', '40', '60', '80', '100']

# Update plotting parameters

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

sns.set_palette('coolwarm')

# Flatten data and labels for plotting
flattened_mse_errors = [error for sublist in all_mse_errors for error in sublist]
node_labels = [level for level, errors in zip(node_number, all_mse_errors) for _ in errors]

data = pd.DataFrame({
    'Node Number': node_labels,
    'Coefficient MSE': flattened_mse_errors
})

summary_stats = data.groupby('Node Number')['Coefficient MSE'].describe(percentiles=[0.25, 0.5, 0.75])

print(summary_stats)

for node_num in summary_stats.index:
    median = summary_stats.loc[node_num, '50%']
    q1 = summary_stats.loc[node_num, '25%']
    q3 = summary_stats.loc[node_num, '75%']
    print(f"Node Number {node_num}: Q1 = {q1}, Median = {median}, Q3 = {q3}")

# Plot the violin plot of MSEs
plt.figure(figsize=(4, 3))
ax = sns.boxplot(x='Node Number', y='Coefficient MSE', data=data, width=0.5, 
                 boxprops=dict(facecolor='none', edgecolor='black'),  # Remove background color
                 medianprops=dict(color='red', linewidth=2),
                 whiskerprops=dict(color='black', linewidth=1),
                 capprops=dict(color='black', linewidth=1),         
                 showfliers=True)          # Make median line red

ax.set_yscale('log')
#ax=sns.violinplot(x='Node Number', y='Coefficient MSE', data=data, inner='quartile', palette="coolwarm", width=0.4)
ax.text(0.10, 0.92, '5% NL', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=10, fontweight='bold')
#ax.set_ylim([0,0.02])
#ax.set_ylim([0.90,1.05])
plt.xlabel('Number of Dimensions')
plt.ylabel('MSE of Coefficients')

plt.tight_layout()
plt.savefig('Dimensions_versus_mse.png', format='png', dpi=1200)
