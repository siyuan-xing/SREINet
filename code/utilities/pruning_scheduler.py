import matplotlib.pyplot as plt
import numpy as np


"""PruningScheduler module.

This module allows the user to create a PruningScheduler.
    
    Author: Siyuan (Simon) Xing and Qingyu Han
    Email: sixing@calpoly.edu
    Licence: MIT Licence
    Copyright (c) 2025
    Version: 1.0.1 
"""

class PruningScheduler:
    """Piecewise Continuous Pruning (PPCP) schedule."""
    
    def __init__(self, steps_per_period, num_periods, 
                 maximum_threshold=0.10, zero_percentage=0.20, ramp_percentage=0.75,
                 ramp_exponent=5, ramp_type='sine', ZERO_THRESHOLD=0.001, show_plot=False):
        """Initialize PPCP pruning schedule.
        
        Args:
            steps_per_period: Number of epochs per pruning period
            num_periods: Number of pruning periods
            maximum_threshold: Maximum pruning threshold (default: 0.10)
            zero_percentage: Percentage of period with zero threshold (default: 0.20)
            ramp_percentage: Percentage of period for ramp-up (default: 0.75)
            ramp_exponent: Exponent for exponential ramp (default: 5)
            ramp_type: Type of ramp ('linear', 'sine', 'exp') (default: 'sine')
            ZERO_THRESHOLD: Minimum threshold value (default: 0.001)
            show_plot: Whether to display the schedule plot (default: False)
        """
        self.steps_per_period = steps_per_period
        self.num_periods = num_periods
        self.total_epochs = steps_per_period * num_periods
        
        # Generate the schedule
        self.schedule = self._create_schedule(
            maximum_threshold, zero_percentage, ramp_percentage,
            ramp_exponent, ramp_type, ZERO_THRESHOLD, show_plot
        )

    def get_threshold(self, epoch):
        """Get the pruning threshold for a given epoch."""
        return self.schedule[epoch]

    def get_period(self, epoch):
        """Get the period number for a given epoch."""
        return epoch // self.steps_per_period
    
    
    def _create_schedule(self, maximum_threshold, zero_percentage, ramp_percentage,
                        ramp_exponent, ramp_type, ZERO_THRESHOLD, show_plot):
        """Create the PPCP schedule."""
        period_steps = self.steps_per_period
        num_periods = self.num_periods
        
        # Calculate the number of steps for each phase within one period
        zero_steps = int(period_steps * zero_percentage)
        ramp_steps = int(period_steps * ramp_percentage)
        hold_steps = period_steps - zero_steps - ramp_steps
        
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
            raise ValueError("Invalid ramp_type. Supported types are 'linear', 'sine', and 'exp'.")

        # Repeat the ramp curve for n_cycles
        combined_array = np.concatenate([ramp_curve for _ in range(num_periods)])

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
            }

            import matplotlib
            matplotlib.rcParams.update(params)
            import seaborn as sns
            sns.set_palette("muted")

            #two subplots 
            fig, axs = plt.subplots(1,2, figsize=(8, 3))
            fig.subplots_adjust(wspace=0.1)
            axs[0].plot(combined_array, linewidth=2)
            axs[0].set(ylabel='Pruning Threshold')
            axs[0].set(xlabel='Epoch')
            axs[1].plot(combined_array[:period_steps], linewidth=2)
            axs[1].set(xlabel='Epoch')
            axs[1].set_yticklabels([])

            plt.tight_layout()
            plt.show()

        return combined_array


#unit test
if __name__ == "__main__":
    # Simple usage with defaults
     
    # All custom parameters
    schedule3 = PruningSchedule(40, 3, maximum_threshold=0.20, zero_percentage=0.30, 
                               ramp_percentage=0.60, ramp_exponent=3, ramp_type='sine', show_plot=True)
    
