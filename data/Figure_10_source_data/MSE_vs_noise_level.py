import os
import numpy as np
import re

# Step 1: Define the directory paths and corresponding noise levels
noise_levels = ['noise_level_1_percent', 'noise_level_5_percent', 'noise_level_10_percent', 'noise_level_15_percent']
base_directory = './Continuous Gaussian Noise/'  # Base directory where the noise folders are located
file_count = 5  # Number of files to process per folder (e.g., 10 files with incremental index)

# Step 2: Initialize an empty list to store accuracies for each noise level
all_accuracies = []

# Define the real coefficients (absolute values)
real_coeff_first_10_dim = 1.0
real_coeff_last_10_dim = 2.0  # absolute values, you've checked the sign by hand

# Step 3: Loop through each noise level directory
for noise_level in noise_levels:
    # Initialize a list to store accuracies for the current noise level
    mse_errors = []
    
    # Step 4: Loop through each file in the current directory
    directory = os.path.join(base_directory, noise_level)
    for i in range(1, file_count + 1):
        filename = f'{directory}/output_{i}.txt'
        
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

        # Step 7: Swap the last two numbers of the first two rows and the last row -> formatting issues
        # No need

        # Step 8: Convert the list of coefficients into a numpy array (matrix)
        coeff_matrix_fist_10_dim = np.concatenate(coefficients[:10])
        coeff_matrix_last_10_dim = np.concatenate(coefficients[10:])

        # Step 9: Calculate the accuracy of the coefficients
        # Calculate the relative error for each coefficient and average them
        #accuracy_first_10_dim = np.abs(coeff_matrix_fist_10_dim) / real_coeff_first_10_dim
        #accuracy_last_10_dim = np.abs(coeff_matrix_last_10_dim) / real_coeff_last_10_dim


        mse_first_half_dim = np.sqrt((np.abs(coeff_matrix_fist_10_dim) - real_coeff_first_10_dim) ** 2).flatten()
        mse_second_half_dim = np.sqrt((np.abs(coeff_matrix_last_10_dim) - real_coeff_last_10_dim) ** 2).flatten()
        mse_second_half_dim = mse_second_half_dim[~np.isnan(mse_second_half_dim)]

        mse_errors += list(mse_first_half_dim) + list(mse_second_half_dim)
    
    # Step 11: Append the accuracies for the current noise level to the main list
    all_accuracies.append(mse_errors)


noise_levels = ['1%', '5%', '10%', '15%']



params = {
    'image.origin': 'lower',
    'image.interpolation': 'nearest',
    'image.cmap': 'gray',
    'axes.grid': False,
    'savefig.dpi': 600,  # to adjust notebook inline plot size
    'axes.labelsize': 12, # fontsize for x and y labels (was 10)
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelweight': 'normal',
    'font.size': 10,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'font.family': 'Helvetica',
}
import matplotlib
matplotlib.rcParams.update(params)

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_palette('Blues')





# Create a DataFrame from the accuracies to use with Seaborn
import pandas as pd

# Flatten the data and create labels for each noise level
flattened_accuracies = [accuracy for sublist in all_accuracies for accuracy in sublist]
noise_labels = [level for level, accuracies in zip(noise_levels, all_accuracies) for _ in accuracies]

# Create a DataFrame to hold the data for Seaborn
data = pd.DataFrame({
    'Noise Level': noise_labels,
    'Coefficient MSE': flattened_accuracies
})


summary_stats = data.groupby('Noise Level')['Coefficient MSE'].describe(percentiles=[0.25, 0.5, 0.75])

# This will print the summary statistics for each noise level including percentiles
print(summary_stats)

# You can access specific percentiles like this, for example:
for noise_level in summary_stats.index:
    median = summary_stats.loc[noise_level, '50%']
    q1 = summary_stats.loc[noise_level, '25%']
    q3 = summary_stats.loc[noise_level, '75%']
    print(f"Noise Level {noise_level}: Q1 = {q1}, Median = {median}, Q3 = {q3}")

from matplotlib.ticker import FuncFormatter

def to_percent(y, position):
    return f'{100 * y:.0f}%'

# Plot the boxplot using Seaborn
plt.figure(figsize=(7.0 / 2.0, 2.625))


#ax = sns.violinplot(x='Noise Level', y='Performance Metric', data=data, showfliers=False, width=0.5, color='skyblue')
# Violin plot with wider violins and inner quartiles
#ax = sns.violinplot(x='Noise Level', y='Performance Metric', data=data, inner='quartile', palette="coolwarm", width=0.4)

ax = sns.boxplot(x='Noise Level', y='Coefficient MSE', data=data, width=0.5, 
                 boxprops=dict(facecolor='none', edgecolor='black'),  # Remove background color
                 medianprops=dict(color='blue', linewidth=2),
                 whiskerprops=dict(color='black', linewidth=1),
                 capprops=dict(color='black', linewidth=1),         
                 showfliers=True)          # Make median line red

ax.set_yscale('log')
# Set y-axis limits to reduce height
#ax.set_yscale('log')
#ax.set_ylim(0.6, 1.2)  # Adjust these limits based on the data range

#sns.swarmplot(x="Noise Level", y="Performance Metric", data=data, color=".25", size=4)

#ax.yaxis.set_major_formatter(FuncFormatter(to_percent))

#ax.set_xticklabels(ax.get_xticklabels(), fontname='Helvetica')
#ax.set_yticklabels(ax.get_yticklabels(), fontname='Helvetica')

#sns.violinplot(x="Noise Level", y="Performance Metric", data=data, inner=None, palette="coolwarm")

# Set title and labels
#plt.title('Accuracy at Different Noise Levels')
plt.xlabel('Noise Level')


#plt.yticks([1.2, 1.0, 0.8, 0.6])
plt.ylabel('MSE of Coefficients')

plt.text(0.01, 0.89, 'n=20', fontsize=10, color='black', ha='left', weight='bold', va='bottom',
         transform=plt.gca().transAxes)



plt.tight_layout()
plt.savefig('noise_plot.png', format='png', dpi=600)
