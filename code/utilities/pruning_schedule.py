import matplotlib.pyplot as plt
import numpy as np

def pruning_schedule(initial_threshold, final_threshold, initial_step, total_steps, pruning_frequency=1, show_plot=False):
    """
    Generates a function that calculates the pruning threshold for a given step, 
    with options for pruning frequency.
    
    Args:
    - initial_threshold: The initial threshold value up to initial_step.
    - final_threshold: The final threshold value at total_steps.
    - initial_step: The step until which the threshold remains at initial_threshold.
    - total_steps: The total number of steps in the schedule.
    - pruning_frequency: The number of steps to hold the current threshold before incrementing.
    
    Returns:
    - A function that takes a step as input and returns the corresponding threshold value.
    """
    def threshold_function(step):
        if step < initial_step:
            return initial_threshold
        elif step >= total_steps:
            return final_threshold
        else:
            remaining_steps = total_steps - initial_step
            increments = remaining_steps // pruning_frequency
            step_increase = (final_threshold - initial_threshold) / increments
            steps_since_initial = step - initial_step
            current_increment = steps_since_initial // pruning_frequency
            current_threshold = initial_threshold + (current_increment * step_increase)
            return min(max(current_threshold, initial_threshold), final_threshold)

    # Plot if required
    if show_plot:
        steps = list(range(total_steps))
        thresholds = [threshold_function(step) for step in steps]
        plt.figure(figsize=(10, 6))
        plt.plot(steps, thresholds, linestyle='-', color='b')
        plt.title("Pruning Threshold Schedule")
        plt.xlabel("Epochs")
        plt.ylabel("Threshold Value")
        plt.grid(True)
        plt.show()

    return threshold_function 

def pruning_schedule_ramp(maximum_threshold, 
                          steps_per_period, num_periods, 
                          zero_percentage=0.20,                           
                          ramp_percentage=0.75,          
                          ramp_exponent = 5,                 
                          show_plot=False, 
                          ramp_type='sine',
                          ZERO_THRESHOLD=0.001):
    # Calculate the period (steps per cycle)
    total_steps = steps_per_period * num_periods
    period = steps_per_period
    
    # Calculate the number of steps for each phase within one period
    zero_steps = int(period * zero_percentage)
    ramp_steps = int(period * ramp_percentage)
    hold_steps = period - zero_steps - ramp_steps
    
    # Generate one cycle of the ramp curve
    if ramp_type == 'linear':
        ramp_curve = np.concatenate((np.zeros(zero_steps),
                                     np.linspace(0, maximum_threshold, ramp_steps),
                                     np.full(hold_steps, maximum_threshold)))
    elif ramp_type == 'sine':  # Sine wave ramp-up curve
        sine_curve = np.sin(np.linspace(-np.pi/2, np.pi/2, ramp_steps)) * (maximum_threshold / 2) + (maximum_threshold / 2)
        ramp_curve = np.concatenate((np.zeros(zero_steps),
                                     sine_curve,
                                     np.full(hold_steps, maximum_threshold)))
    elif ramp_type == 'exp':  # Exponential growth ramp-up curve
        exp_curve = np.exp(np.linspace(0, (maximum_threshold + 1), ramp_steps) * ramp_exponent) - 1
        exp_max = max(exp_curve)
        exp_curve = (exp_curve / exp_max) * maximum_threshold
        ramp_curve = np.concatenate((np.zeros(zero_steps),
                                     exp_curve,
                                     np.full(hold_steps, maximum_threshold)))
        ramp_curve[ramp_curve < ZERO_THRESHOLD] = 0 # Trimming extremely small values
    else:
        raise ValueError("Invalid ramp_type. Supported types are 'linear' and 'exponential'.")


    # Repeat the ramp curve for n_cycles
    combined_array = np.concatenate([ramp_curve for _ in range(num_periods)])

    # Trim or pad the combined array to match total_steps
    #combined_array = combined_array[:total_steps]

    if show_plot:

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
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            #'font.family': 'Helevetica',  # Specify font family
        }

        import matplotlib
        matplotlib.rcParams.update(params)
        import seaborn as sns
        sns.set_palette("muted")

        #two subplots 
        fig, axs = plt.subplots(1,2, figsize=(8, 3))
        fig.subplots_adjust(wspace=0.1)
        #fig.suptitle('Pruning Schedule')
        axs[0].plot(combined_array, linewidth=2)
        axs[0].set(ylabel='Pruning Threshold')
        axs[0].set(xlabel='Epoch')
        #axs[0].set_yticks([0, 0.05, 0.10])
        axs[1].plot(combined_array[:steps_per_period], linewidth=2)
        axs[1].set(xlabel='Epoch')
        #axs[1].set_yticks([0, 0.05, 0.10])
        axs[1].set_yticklabels([])

        #plt.grid(True)
        #space between subplots
        plt.tight_layout()
        #plt.savefig('pruning_schedule.png', dpi=1200)
        plt.show()


    return combined_array

if __name__ == "__main__":

    # Define the pruning schedule
    pruning_schedule = pruning_schedule_ramp(maximum_threshold=0.10, 
                                                        steps_per_period=50, 
                                                        num_periods=5, 
                                                        zero_percentage=0.20,
                                                        show_plot=True)
