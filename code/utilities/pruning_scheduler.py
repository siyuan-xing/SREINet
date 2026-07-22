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
                 maximum_threshold=0.10, zero_percentage=None,
                ramp_percentage=None,
                ramp_type='sine', sigmoid_steepness=15.0, sigmoid_steepness_jitter=0.0,
                sigmoid_jitter_seed=42, show_plot=False):
        """Initialize PPCP pruning schedule.
        
        Args:
            steps_per_period: Number of epochs per pruning period
            num_periods: Number of pruning periods
            maximum_threshold: Maximum pruning threshold (default: 0.10)
            zero_percentage: Percentage of period with zero threshold (default: 0.20)
            ramp_percentage: Percentage of period for ramp-up (default: 0.75)
            ramp_type: Type of ramp ('linear', 'sine') (default: 'sine')
            sigmoid_steepness_jitter: Standard deviation for normal integer jitter when ramp_type
                                      is 'sigmoid' (default: 1.0, set 0 for deterministic behaviour)
            sigmoid_jitter_seed: Seed for jitter RNG to make sigmoid ramp reproducible (default: 42)
            show_plot: Whether to display the schedule plot (default: False)
        """
        self.steps_per_period = steps_per_period
        self.num_periods = num_periods
        self.total_epochs = steps_per_period * num_periods
        self.sigmoid_steepness = sigmoid_steepness
        self.sigmoid_steepness_jitter = sigmoid_steepness_jitter
        self.sigmoid_jitter_seed = sigmoid_jitter_seed

        defaults = {
        'sine': (0.20, 0.75),
        'sigmoid': (0.0, 1.0)
        }
        default_zero, default_ramp = defaults.get(ramp_type, (0.0, 1.0))
        zero_percentage = zero_percentage if zero_percentage is not None else default_zero
        ramp_percentage = ramp_percentage if ramp_percentage is not None else default_ramp
        
        
        # Generate the schedule
        self.schedule = self._create_schedule(
            maximum_threshold, zero_percentage, ramp_percentage,
             ramp_type, sigmoid_steepness, sigmoid_steepness_jitter,
             sigmoid_jitter_seed, show_plot
        )

    def get_threshold(self, epoch):
        """Get the pruning threshold for a given epoch."""
        return self.schedule[epoch]

    def get_period(self, epoch):
        """Get the period number for a given epoch."""
        return epoch // self.steps_per_period
    
    
    def _create_schedule(self, maximum_threshold, zero_percentage, ramp_percentage,
                         ramp_type, sigmoid_steepness, sigmoid_steepness_jitter,
                         sigmoid_jitter_seed, show_plot=False):
        """Create the PPCP schedule."""
        period_steps = self.steps_per_period
        num_periods = self.num_periods
        
        # Calculate the number of steps for each phase within one period
        zero_steps = int(period_steps * zero_percentage)
        ramp_steps = int(period_steps * ramp_percentage)
        hold_steps = period_steps - zero_steps - ramp_steps

        #std = max(0.0, float(sigmoid_steepness_jitter))
        #rng = np.random.default_rng(sigmoid_jitter_seed) if std > 0 else None

        # For sigmoid type, each period needs different jitter, so generate separately
        if ramp_type == 'sigmoid':
            sigmoid_midpoint = 0.5
            ramp_curves = []
            for period_idx in range(num_periods):
                # Generate a unique jitter for each period
                #
                #jitter_delta = int(np.round(np.abs(rng.normal(loc=0.0, scale=std)))) if rng is not None else 0
                #effective_sigmoid_steepness = max(1, int(round(sigmoid_steepness + jitter_delta)))
                progress = np.linspace(0.0, 1.0, ramp_steps)
                sigmoid_curve = 1 / (1.0 + np.exp(-sigmoid_steepness * (progress - sigmoid_midpoint)))
                sigmoid_curve = maximum_threshold*(sigmoid_curve - sigmoid_curve.min()) / max(sigmoid_curve.max() - sigmoid_curve.min(), 1e-8)
                ramp_curve = np.concatenate((np.zeros(zero_steps),
                                             sigmoid_curve,
                                             np.full(hold_steps, maximum_threshold)))
                ramp_curves.append(ramp_curve)
            combined_array = np.concatenate(ramp_curves)
        else:
            # For linear and sine types, generate once and repeat
            if ramp_type == 'linear':
                ramp_curve = np.concatenate((np.zeros(zero_steps),
                                             np.linspace(0, maximum_threshold, ramp_steps),
                                             np.full(hold_steps, maximum_threshold)))
            elif ramp_type == 'sine':  # Sine wave ramp-up curve
                sine_curve = np.sin(np.linspace(-np.pi/2, np.pi/2, ramp_steps)) * (maximum_threshold / 2) + (maximum_threshold / 2)
                ramp_curve = np.concatenate((np.zeros(zero_steps),
                                             sine_curve,
                                             np.full(hold_steps, maximum_threshold)))
            else:
                raise ValueError("Invalid ramp_type. Supported types are 'linear', 'sine', and 'sigmoid'.")
            
            # Repeat the ramp curve for n_cycles
            combined_array = np.concatenate([ramp_curve for _ in range(num_periods)])

        if show_plot:
            params = {
                'image.origin': 'lower',
                'image.interpolation': 'nearest',
                'image.cmap': 'gray',
                'axes.grid': False,
                'savefig.dpi': 600,  # to adjust notebook inline plot size
                'axes.labelsize': 12,
                'axes.titlesize': 12,
                'font.size': 10,
                'legend.fontsize': 10,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
                'font.family': 'Helvetica',
            }

            import matplotlib
            matplotlib.rcParams.update(params)
            import seaborn as sns
            sns.set_palette("muted")

            #two subplots 
            fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.7))
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
    schedule3 = PruningScheduler(40, 3, maximum_threshold=0.10, zero_percentage=0.00, 
                               ramp_percentage=1.0, sigmoid_steepness=10.0, ramp_type='sigmoid', show_plot=True)
    
