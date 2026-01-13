"""
Symbolic Model Module for SREINet ODE Recovery

This module provides a RecoveredModel class for parsing, simulating, and evaluating
ODEs identified by SREINet. It integrates with the coefficient reconstructor to
directly process SREINet neural network weights.
"""

import re
import sys
import numpy as np
from scipy.integrate import solve_ivp


class RecoveredModel:
    """
    A class for handling recovered ODEs from SREINet output.

    Provides functionality to:
    - Parse ODE equations from text output
    - Simulate the recovered system
    - Evaluate identification accuracy
    """

    def __init__(self, variable_patterns=None, transcendental_functions=None):
        self.equations = []
        self.variables = set()
        self.coefficients = {}
        self.variable_mapping = {}

        if variable_patterns is None:
            self.variable_patterns = [r'[a-zA-Z]\d*']
        else:
            self.variable_patterns = variable_patterns

        if transcendental_functions is None:
            self.transcendental_functions = {
                'sin': np.sin,
                'cos': np.cos,
                'tan': np.tan,
                'exp': np.exp,
                'log': np.log,
                'ln': np.log,
                'sqrt': np.sqrt,
                'abs': np.abs
            }
        else:
            self.transcendental_functions = transcendental_functions

    def parse_ode_equations(self, ode_text, minimum_coefficient_threshold=1e-6):
        if isinstance(ode_text, str):
            equations = ode_text.strip().split('\n')
        else:
            equations = ode_text

        self.equations = []
        self.variables = set()
        self.coefficients = {}

        for eq_str in equations:
            if not eq_str.strip():
                continue

            # Parse equation of form: (var)' = expression
            match = re.match(r"\(([^)]+)\)'\s*=\s*(.+)", eq_str.strip())
            if not match:
                continue

            lhs_var = match.group(1)
            rhs_expr = match.group(2)

            # Extract variables from the expression using user-defined patterns
            found_vars = set()
            found_vars.add(lhs_var)

            for pattern in self.variable_patterns:
                potential_vars = re.findall(pattern, rhs_expr)
                # Filter out function names from potential variables
                for var in potential_vars:
                    # Check if this variable is part of a function name
                    is_part_of_function = False
                    for func_name in self.transcendental_functions.keys():
                        if var in func_name:
                            # Check if this character appears in the context of the function
                            func_pattern = rf'\b{func_name}\('
                            if re.search(func_pattern, rhs_expr):
                                is_part_of_function = True
                                break
                    
                    if not is_part_of_function:
                        found_vars.add(var)

            self.variables.update(found_vars)

            # Parse terms and coefficients
            terms = self._parse_expression_terms(rhs_expr, minimum_coefficient_threshold)

            equation_data = {
                'lhs': lhs_var,
                'rhs': rhs_expr,
                'terms': terms
            }

            self.equations.append(equation_data)
            self.coefficients[lhs_var] = terms

        # Create variable mapping for numerical simulation
        self._create_variable_mapping()

        result = {
            'equations': self.equations,
            'variables': sorted(list(self.variables)),
            'coefficients': self.coefficients
        }
        return result

    def _parse_expression_terms(self, expression, threshold):
        terms = []

        # Split by + and - while keeping the signs
        parts = re.split(r'(\s*[+-]\s*)', expression)
        current_term = ""

        for i, part in enumerate(parts):
            if re.match(r'\s*[+-]\s*', part) and i > 0:
                if current_term.strip():
                    term_data = self._parse_single_term(current_term.strip(), threshold)
                    if term_data:
                        terms.append(term_data)
                current_term = part
            else:
                current_term += part

        # Handle the last term
        if current_term.strip():
            term_data = self._parse_single_term(current_term.strip(), threshold)
            if term_data:
                terms.append(term_data)

        return terms

    def _parse_single_term(self, term, threshold):
        term = term.strip()

        # Handle sign
        sign = 1
        if term.startswith('-'):
            sign = -1
            term = term[1:].strip()
        elif term.startswith('+'):
            term = term[1:].strip()

        # Check for transcendental functions
        func_type = None
        func_arg = None

        for func in self.transcendental_functions.keys():
            pattern = rf'{func}\(([^)]+)\)'
            match = re.search(pattern, term)
            if match:
                func_type = func
                func_arg = match.group(1)
                # Remove function part and continue parsing coefficient
                term = re.sub(pattern, '', term)
                if term.startswith('*'):
                    term = term[1:]
                elif term == '':
                    term = '1'
                break

        # Extract coefficient and variables
        if '*' in term:
            parts = term.split('*')
            try:
                coeff = float(parts[0]) * sign
            except ValueError:
                coeff = sign
                parts = [term]
        else:
            # Check if it's just a number (constant term)
            try:
                coeff = float(term) * sign
                parts = []
            except ValueError:
                coeff = sign
                parts = [term]

        # Filter by threshold
        if abs(coeff) < threshold:
            return None

        # Extract variable factors
        variables = []
        for part in parts[1:] if '*' in term else parts:
            # Handle powers like a1^2
            if '^' in part:
                # Use any of the variable patterns to match
                for pattern in self.variable_patterns:
                    var_match = re.match(rf'({pattern})\^(\d+)', part)
                    if var_match:
                        var_name = var_match.group(1)
                        power = int(var_match.group(2))
                        variables.extend([var_name] * power)
                        break
            else:
                # Check if part matches any variable pattern and is not a function name
                for pattern in self.variable_patterns:
                    if re.match(f'^{pattern}$', part) and part not in self.transcendental_functions:
                        variables.append(part)
                        break

        # Add function information
        result = {
            'coefficient': coeff,
            'variables': variables,
            'original_term': ('+' if sign > 0 and (not parts or term != parts[0]) else '') + term
        }

        if func_type:
            result['function'] = func_type
            result['function_arg'] = func_arg

        return result

    def _create_variable_mapping(self):
        """Create mapping from variable names to indices for numerical simulation."""
        # Sort variables properly: x1, x2, ..., x9, x10, x11, etc.
        def sort_key(var):
            # Extract the number part for proper numerical sorting
            import re
            match = re.match(r'([a-zA-Z]+)(\d+)', var)
            if match:
                prefix, number = match.groups()
                return (prefix, int(number))
            else:
                return (var, 0)
        
        sorted_vars = sorted(list(self.variables), key=sort_key)
        self.variable_mapping = {var: i for i, var in enumerate(sorted_vars)}

    def _compile_ode_system(self):
        """Pre-compile the ODE system function for efficient evaluation."""
        if not self.equations:
            raise ValueError("No equations to compile. Call parse_ode_equations first.")
        
        # Pre-compute equation structure for efficient evaluation
        self._compiled_equations = []
        
        for eq in self.equations:
            lhs_var = eq['lhs']
            lhs_idx = self.variable_mapping[lhs_var]
            
            # Pre-compile terms for this equation
            compiled_terms = []
            for term in eq['terms']:
                coeff = term['coefficient']
                var_indices = [self.variable_mapping[var] for var in term['variables']]
                
                compiled_term = {
                    'coefficient': coeff,
                    'var_indices': var_indices,
                    'has_function': 'function' in term
                }
                
                if 'function' in term:
                    func_type = term['function']
                    func_arg = term['function_arg']
                    func_arg_idx = self.variable_mapping[func_arg]
                    
                    compiled_term.update({
                        'function': self.transcendental_functions[func_type],
                        'func_type': func_type,
                        'func_arg_idx': func_arg_idx
                    })
                
                compiled_terms.append(compiled_term)
            
            self._compiled_equations.append({
                'lhs_idx': lhs_idx,
                'terms': compiled_terms
            })
    
    def _create_ode_function_from_strings(self):
        """Create ODE function directly from exported equation strings."""
        if not self.equations:
            raise ValueError("No equations to compile. Call parse_ode_equations first.")
        
        # Get equation strings in Python format
        python_eqs = self.export_equations(format_type='python')
        
        # Create variable mapping for the function with proper sorting
        def sort_key(var):
            # Extract the number part for proper numerical sorting
            import re
            match = re.match(r'([a-zA-Z]+)(\d+)', var)
            if match:
                prefix, number = match.groups()
                return (prefix, int(number))
            else:
                return (var, 0)
        
        var_names = sorted(list(self.variables), key=sort_key)
        var_mapping = {var: i for i, var in enumerate(var_names)}
        
        # Process equations to add np. prefix to special functions
        processed_eqs = []
        for eq in python_eqs:
            processed_eq = eq
            # Add np. prefix to transcendental functions
            for func_name in self.transcendental_functions.keys():
                # Replace function calls with np.function calls
                pattern = f'\\b{func_name}\\('
                replacement = f'np.{func_name}('
                processed_eq = re.sub(pattern, replacement, processed_eq)
            processed_eqs.append(processed_eq)
        
        # Create the ODE function code
        # Build the code line by line to avoid indentation issues
        code_lines = [
            "def ode_system(t, y):",
            '    """Generated ODE system function."""',
            "    import numpy as np",
            "",
            "    # Calculate derivatives",
            "    dydt = np.zeros_like(y)",
            ""
        ]
        
        # Add direct dydt assignments with y[i] substitution
        for i, eq in enumerate(processed_eqs):
            # Extract the right-hand side of the equation
            rhs = eq.split('=')[1].strip()
            
            # Replace variable names with y[i] references
            processed_rhs = rhs
            for j, var in enumerate(var_names):
                # Replace variable names with y[j] references
                # Use word boundary to ensure we only replace complete variable names
                # Also ensure we don't replace function names
                if var not in self.transcendental_functions:
                    pattern = r'\b' + re.escape(var) + r'\b'
                    processed_rhs = re.sub(pattern, f'y[{j}]', processed_rhs)
            
            code_lines.append(f"    dydt[{i}] = {processed_rhs}")
        
        code_lines.extend([
            "",
            "    return dydt"
        ])
        
        ode_code = "\n".join(code_lines)
        
        # Execute the code to create the function
        local_vars = {}
        exec(ode_code, {'np': np}, local_vars)
        
        return local_vars['ode_system'], var_names
    
    def split_complex_initial_conditions(self, complex_ic):
        """
        Split complex initial conditions into real and imaginary parts.
        
        This function handles the case where complex variables have been split into
        real and imaginary parts.
        Parameters:
        -----------
        complex_ic : array-like
            Complex initial conditions array
            
        Returns:
        --------
        real_ic : numpy.ndarray
            Real initial conditions array with real parts first, then imaginary parts
        """
        import numpy as np
        
        complex_ic = np.asarray(complex_ic)
        
        # Split into real and imaginary parts
        real_parts = np.real(complex_ic)
        imag_parts = np.imag(complex_ic)
            
        # Concatenate real parts first, then imaginary parts: [a1, a2, ..., a50, b1, b2, ..., b50]
        real_ic = np.concatenate([real_parts, imag_parts])
            
        return real_ic

    def simulate(self, initial_conditions, t_span, t_eval=None, method='RK45'):
        """Simulate the ODE system using string-generated functions."""
        
        # Create ODE function from strings if not already created
        if not hasattr(self, '_ode_function'):
            self._ode_function, self._var_names = self._create_ode_function_from_strings()
        
        sol = solve_ivp(self._ode_function, t_span, initial_conditions, t_eval=t_eval, method=method, dense_output=True)

        result = {
            'success': sol.success,
            't': sol.t,
            'y': sol.y,
            'message': sol.message,
            'variable_names': sorted(list(self.variables))
        }
        return result


    def score(self, X_data, X_dot_data):
        """
        Calculates R² in derivative space.
        """
        # Predict derivatives using the recovered model
        X_dot_pred = self.predict_derivatives(X_data)
        
        # Calculate R² in derivative space (exactly like SINDy)
        ss_res = np.sum((X_dot_data - X_dot_pred)**2)
        ss_tot = np.sum((X_dot_data - np.mean(X_dot_data))**2)
        
        score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return score

    def calculate_complexity(self):
        """
        Calculate model complexity - count non-zero coefficients in equation right-hand sides.
        
        Returns:
        --------
        complexity : int
            Number of non-zero coefficients
        """

        non_zero_count = 0
        
        for eq in self.equations:
            for term in eq['terms']:
                coeff = term['coefficient']
                if abs(coeff) > 1e-10:  # Non-zero threshold
                    non_zero_count += 1
        
        return non_zero_count


    def export_equations(self, format_type='text'):

        formatted_eqs = []

        if format_type == 'latex':
            for eq in self.equations:
                lhs = f"\\frac{{d{eq['lhs']}}}{{dt}}"
                rhs = eq['rhs'].replace('*', ' \\cdot ').replace('^', '^{') + '}'
                formatted_eqs.append(f"{lhs} = {rhs}")

        elif format_type == 'matlab':
            for eq in self.equations:
                lhs = f"d{eq['lhs']}_dt"
                rhs = eq['rhs'].replace('^', '.^')
                formatted_eqs.append(f"{lhs} = {rhs};")

        elif format_type == 'python':
            for eq in self.equations:
                lhs = f"d{eq['lhs']}_dt"
                rhs = eq['rhs'].replace('^', '**')
                formatted_eqs.append(f"{lhs} = {rhs}")

        else:  # default format
            for eq in self.equations:
                formatted_eqs.append(f"({eq['lhs']})' = {eq['rhs']}")

        return formatted_eqs

    def predict_derivatives(self, X_data):
        """
        Predict derivatives using the recovered model.
        """
        if not hasattr(self, '_ode_function'):
            self._ode_function, self._var_names = self._create_ode_function_from_strings()
            
        X_dot_pred = np.zeros_like(X_data)
            
        for i in range(X_data.shape[0]):
            # Get current state
            y = X_data[i, :]
                
            # Calculate derivatives using the ODE function
            dydt = self._ode_function(0, y)  # t=0 since we only need derivatives
                
            X_dot_pred[i, :] = dydt
            
        return X_dot_pred
