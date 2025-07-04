import numpy as np
import matplotlib.pyplot as plt

def plot_kuramoto_phases(pred_phase, pred_velc, true_phase, true_velc, ax, title):
    """
    Plots the phases of Kuramoto oscillators for two datasets on a polar plot.

    """

    # Extract data from the datasets and apply modulo 2π to phases
    primary_phases = np.mod(pred_phase, 2*np.pi)
    primary_velocities = pred_velc
    validation_phases = np.mod(true_phase, 2*np.pi)
    validation_velocities = true_velc

    # Normalize velocities to range [0, 1] for color mapping
    normalized_primary_velocities = (primary_velocities - primary_velocities.min()) / (primary_velocities.max() - primary_velocities.min())
    normalized_validation_velocities = (validation_velocities - validation_velocities.min()) / (validation_velocities.max() - validation_velocities.min())

    # Use a color map (e.g., "viridis") to assign colors based on velocities
    primary_colors = plt.cm.viridis(normalized_primary_velocities)
    validation_colors = plt.cm.viridis(normalized_validation_velocities)

    # Setup the polar plot
    ax.set_theta_direction(-1)  # Clockwise
    ax.set_theta_zero_location('N')  # Set 0 at the top (North)

    ax.grid(False)  # This line removes the grid from the polar plot
    ax.set_yticklabels([])  # This line removes the radial labels
    ax.spines['polar'].set_visible(False)  # This hides the axis circle

    # Set the angles to be shown (0°, 90°, 180°, 270°)
    ax.set_xticklabels([])
    ax.set_title(title)
     # Optionally, draw a unit circle
    circle = plt.Circle((0.0, 0.0), 1.0, transform=ax.transData._b, color="black", fill=False, lw=1)  # Thinner black line for the circle
    ax.add_artist(circle)


    # Plot validation data (larger circles with colored edge and white center)
    validation_radii = np.ones(len(validation_phases))  # All validation oscillators on the same radius
    for i in range(len(validation_phases)):
        ax.scatter(validation_phases[i], validation_radii[i], color='white', s=150, edgecolor=validation_colors[i], linewidth=2)  # Larger circle with white center and colored edge

    # Plot each oscillator phase with a color based on its velocity (smaller circles)
    primary_radii = np.ones(len(primary_phases))  # All oscillators on the same radius
    for i in range(len(primary_phases)):
        ax.scatter(primary_phases[i], primary_radii[i], color='white', s=120, edgecolor='black', linewidth=0.5)  # White background circle with a thin black edge
        ax.scatter(primary_phases[i], primary_radii[i], color=primary_colors[i], s=100)  # Colored circle on top

   
    # Add a color bar to indicate the mapping from velocity to color
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=primary_velocities.min(), vmax=primary_velocities.max()))
    sm.set_array([])

    # plt.colorbar(sm, ax=ax, orientation='vertical', label='Oscillator Velocity')

    

