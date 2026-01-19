import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.colors as mcolors

# === Basic Settings ===
bright_palette = sns.color_palette("Set2")  # Use the same color scheme as draw_dataset.py

sns.set_theme(context='paper',  # 'paper' for publication
              style='white',    # White background
              palette=bright_palette,  # Colorblind-friendly color scheme
              font='Arial',     # Nature typically recommends Arial or Helvetica
              font_scale=0.8)   # Adjust overall font size

palette = sns.color_palette("Set2")
# Group colors by variable type, using similar but slightly different colors within each group
# First derivative group: use 3 shades of blue
# Second derivative group: use 3 shades of orange

# Create blue gradient (first derivatives)
blue_base = palette[2]  # First color of Set2 (blue-green)
blue_colors = [
    mcolors.to_rgba(blue_base, alpha=0.7),  # Lighter
    blue_base,                              # Original color
    tuple(min(1.0, c*0.8) if i < 3 else c for i, c in enumerate(blue_base))  # Darker
]

# Create orange gradient (second derivatives)
orange_base = palette[0]  # Second color of Set2 (orange)
orange_colors = [
    mcolors.to_rgba(orange_base, alpha=0.7),  # Lighter
    orange_base,                              # Original color
    tuple(min(1.0, c*0.8) if i < 3 else c for i, c in enumerate(orange_base))  # Darker
]

my_colors = blue_colors + orange_colors

def plot_loss_bar_chart(loss_values, save_path="loss_bar_chart.png", use_log_scale=True):
    """
    Plot bar chart for loss values of 6 variables
    Args:
        loss_values (list): List of 6 loss values corresponding to [theta_1_dot, theta_2_dot, theta_3_dot, theta_1_ddot, theta_2_ddot, theta_3_ddot]
        save_path (str): Path to save the image
        use_log_scale (bool): Whether to use logarithmic scale
    """
    
    # Define labels
    labels = [r'$\dot{\theta}_1$', r'$\dot{\theta}_2$', r'$\dot{\theta}_3$', 
              r'$\ddot{\theta}_1$', r'$\ddot{\theta}_2$', r'$\ddot{\theta}_3$']
    
    # Create x-axis positions
    x_pos = np.arange(len(labels))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(2, 3))
    
    # Plot bar chart
    bars = ax.bar(x_pos, loss_values, color=my_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Set labels and title
    ax.set_xlabel('Vector Field', fontsize=10)
    ax.set_ylabel('Loss', fontsize=12)
    #ax.set_title('Loss Comparison for Different Variables', fontsize=14, fontweight='bold')
    
    # Set x-axis labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=10)
    
    # If using logarithmic scale
    if use_log_scale:
        ax.set_yscale('log')
        ax.set_ylabel('Loss (log scale)', fontsize=10)
        
        # Set specific y-axis ticks
        yticks = [1e0, 1e-5, 1e-10, 1e-15]
        ax.set_yticks(yticks)
        ax.set_yticklabels([r'$10^0$', r'$10^{-5}$', r'$10^{-10}$', r'$10^{-15}$'])
    else:
        # Remove number labels for linear scale
        pass
    
    # Beautify the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.show()
    
    print(f"Bar chart saved to: {save_path}")

# User-provided actual loss values
if __name__ == "__main__":
    # User-provided loss values
    actual_loss_values = [4.1400e-15, 1.7127e-8, 2.4584e-14, 5.4800e-01, 1.4875e+00, 3.5452e+00]
    
    print("Plotting bar chart using user-provided loss values...")
    print("Loss values:")
    labels = ['theta_1_dot', 'theta_2_dot', 'theta_3_dot', 'theta_1_ddot', 'theta_2_ddot', 'theta_3_ddot']
    for label, value in zip(labels, actual_loss_values):
        print(f"{label}: {value:.4e}")
    print()
    
    # Plot bar chart (using logarithmic scale)
    plot_loss_bar_chart(actual_loss_values, "loss_bar_chart_log_scale.png", use_log_scale=True) 