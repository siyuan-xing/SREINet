import Model_zoo as zoo
import numpy as np

class DataGenerator:
    # model_mappings is a dictionary that maps model names to a tuple of the form (function, num_parameters, additional_parameters)
    model_mappings = {
        'linear 3d ode':    (zoo.linear_3D, 2, {}),
        'lorenz':           (zoo.lorenz,        2, {'method': 'RK45'}),
        'duffing':          (zoo.duffing,       5, {}),
        'meanfield':        (zoo.meanfield,     2, {}),
        'mhd':              (zoo.mhd,           2, {}),
        'lorenz_4d':        (zoo.lorenz_4d,     2, {}),
        'shimizu morioka':  (zoo.shimizu_morioka, 2, {}),
        'lorenz_96':        (zoo.lorenz96,      2, {}),  # Adding arguments specific to Lorenz 96
        'kuramoto':         (zoo.kuramoto,      2, {}), 
        "discrete_sine_gordon": (zoo.DSG, 2, {}),
        "discrete_phi_quartic": (zoo.discrete_phi_quartic, 3, {}),
        "dnls":            (zoo.DNLS, 3, {}),
        "dnls_split":       (zoo.DNLS_split, 3, {}),
        "fermi_pasta_ulam": (zoo.fermi_pasta_ulam, 3, {}),
        }
    
    @staticmethod
    def generate_initial_conditions(n, num_conditions=1, seed=None):
        
        """
        Generates random initial conditions for the Lorenz '96 model, ensuring
        that all initial conditions are greater than 1.
        """
        if seed is not None:
            np.random.seed(seed)  # Ensures reproducibility if a seed is provided
        
        initial_conditions = []
        for _ in range(num_conditions):
            # Generate values in the range [1, 11) to ensure all numbers are > 1
            initial_conditions.append(np.random.rand(n))
        return initial_conditions
    
    @staticmethod
    def generate_initial_conditions_by_adding_dx(n, x0, num_conditions=1, seed=None):
        
        """
        Generates random initial conditions for the Lorenz '96 model, ensuring
        that all initial conditions are greater than 1.
        """
        if seed is not None:
            np.random.seed(seed)  # Ensures reproducibility if a seed is provided
        
        initial_conditions = []
        for _ in range(num_conditions):
            # Generate values in the range [1, 11) to ensure all numbers are > 1
            new_IC = np.concatenate((x0,  np.random.rand(n)), axis=0) # add velocity
            initial_conditions.append(new_IC)
        return initial_conditions

    def __init__(self, initial_conditions:np.array, T:float=10.0, dt:float=0.1, t0:float=0.0, noise_level:float=0.0, derivative_mode="exact"):
        """
        Initializes a DataGenerator object.

        Parameters:
        - initial_conditions: A list or a single value representing the initial conditions of ODEs.
        """
        # if initial_condition is not a list ,make it a list
        if not isinstance(initial_conditions, list):
            initial_conditions = [initial_conditions]
        
        initial_conditions = [initial_condition.reshape(1, -1) if len(initial_condition)== 1 
                              else initial_condition 
                              for initial_condition in initial_conditions]

        self.ICs = initial_conditions
        self.T = T
        self.t0 = t0
        self.dt = dt
        self.noise_level = noise_level
        self.derivative_mode = derivative_mode

    def generate_dataset_by_model_name(self, model_name, *args, **kwargs):
        """
        Generate a dataset based on the provided model name. 

        Args:
            model_name (str): The name of the model to use for generating the dataset. The model is defined in model zoo.
            integrator_keywords (dict): Additional keyword arguments for the integrator function (default: {}).
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            tuple: A tuple containing the generated input data (x_train) and the corresponding output data (dx_train),
                   as well as the highest order polynomial to guess.

        Raises:
            ValueError: If an invalid or ambiguous model type is provided.
        """
        # Find a model using model_name
        model_name_lower = model_name.lower()  # Convert to all lower case
        matched_model = None
        for key in self.model_mappings.keys():
            if key.startswith(model_name_lower):
                matched_model = self.model_mappings[key]
                break

        if matched_model is None:
            raise ValueError("Invalid or ambiguous model type provided.")
        
        gen_function, guess_highest_order_polynomial, default_args = matched_model
        
        #combine default arguments with user provided arguments
        default_args.update(kwargs)
        
        # Merge default arguments with those provided; user-provided override defaults
        
        # Create an empty list to store the data
        x_train  = []
        dx_train = []
        for x0 in self.ICs:
            # Generate data for each initial condition, append to the data array
            temp_x, temp_dx = self.generate_ode_traj_with_ic(gen_function, x0, self.derivative_mode, *args, **default_args)
            x_train.append(temp_x)
            dx_train.append(temp_dx)
        
        x_train = np.vstack(x_train)
        dx_train = np.vstack(dx_train)
        t_arr = (np.arange(self.t0, self.T, self.dt)).reshape(-1, 1)

        return t_arr, x_train, dx_train, guess_highest_order_polynomial

    def generate_dataset_by_custom_ODEs(self, ODEfuns, *args, **kwargs):
        """
        Generates a dataset by solving custom ODEs using the provided ODE functions.

        Args:
            ODEfuns (list): A list of ODE functions to solve.
            integrator_keywords (dict, optional): Additional keyword arguments for the integrator. Defaults to {}.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            tuple: A tuple containing the generated input data (x_train) and the corresponding output data (dx_train).
        """
        
        # Create an empty list to store the data
        x_train  = []
        dx_train = []
        for x0 in self.ICs:
            # Generate data for each initial condition, append to the data array
            temp_x, temp_dx = self.generate_ode_traj_with_ic(ODEfuns, x0, self.derivative_mode, *args, **kwargs)
            x_train.append(temp_x)
            dx_train.append(temp_dx)
        
        x_train = np.vstack(x_train)
        dx_train = np.vstack(dx_train)
        t_arr = np.arange(self.t0, self.T, self.dt)

        return t_arr, x_train, dx_train

    def shuffleData(self, x_train, dx_train):
        """
        Shuffles the input and output data.

        Args:
            x_train (array-like): The input data.
            dx_train (array-like): The output data.

        Returns:
            tuple: A tuple containing the shuffled input data (x_train) and the corresponding output data (dx_train).
        """
        indices = np.arange(x_train.shape[0])
        np.random.shuffle(indices)
        x_train = x_train[indices]
        dx_train = dx_train[indices]
        return x_train, dx_train

    # def generate_ode_traj_with_ic(self, model_ode, x0, integrator_keywords, derivative_mode="exact"): 
    def generate_ode_traj_with_ic(self, model_ode, x0, derivative_mode="exact", *args, **kwargs):
        """
        Generates trajectory data for an ODE model with initial conditions.

        Args:
            model_ode (callable): The ODE model function.
            x0 (array-like): The initial conditions for the ODE.
            integrator_keywords (dict): Additional keyword arguments to pass to the ODE solver.
            derivative_mode (str, optional): The mode for calculating the derivative data. 
                Can be "exact" or "finite_difference". Defaults to "exact".

        Returns:
            tuple: A tuple containing the trajectory data (x_train) and the derivative data (dx_train).

        Raises:
            None
        """

        t_span = (self.t0, self.T)
        t_train = np.arange(self.t0, self.T, self.dt)
        from scipy.integrate import solve_ivp
        x_train = solve_ivp(
            lambda t, x: model_ode(t, x, *args), 
            t_span, 
            x0, 
            t_eval=t_train,
            **kwargs
        ).y.T
        
        if derivative_mode == "exact":
            # Exact velocity data
            dx_train = np.array([model_ode(t, x, *args) for t, x in zip(t_train, x_train)])
        elif derivative_mode == "finite_difference":
            # Velocity data calculated by finite difference
            dx_train = np.diff(x_train, axis=0) / self.dt
            dx_train = np.vstack([dx_train, dx_train[-1]])  # Pad to maintain the same length

        x_train = x_train + self.noise_level * np.random.randn(*x_train.shape)
        dx_train = dx_train + self.noise_level * np.random.randn(*dx_train.shape)

        return x_train, dx_train

