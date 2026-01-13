import numpy as np
import re

def parse_ode(equation_text):
    # Replace (xN)' with dxdt[N-1]
    equation_text = equation_text.replace("'", "")
    
    # General: Replace dxN with dxdt[N-1] for any N
    equation_text = re.sub(r'\bdx(\d+)\b', lambda m: f"dxdt[{int(m.group(1)) - 1}]", equation_text)
    
    # Replace xN with x[N-1]
    equation_text = re.sub(r'\bx(\d+)\b', lambda m: f"x[{int(m.group(1)) - 1}]", equation_text)
    
    # Replace cos and sin with np.cos and np.sin
    equation_text = equation_text.replace("cos", "np.cos").replace("sin", "np.sin")
    
    # Replace ^ with ** for power operation
    equation_text = equation_text.replace("^", "**")
    
    # Extract dxdt[N-1] assignment and expression
    lhs, rhs = equation_text.split("=")
    
    # Remove spaces around equals sign for better readability
    lhs = re.sub(r'x\[(\d+)\]', r'dxdt[\1]', lhs.strip())
    rhs = rhs.strip()
    
    return lhs, rhs

def generate_ode_function(equations):
    lines = []
    lines.append("import numpy as np")
    lines.append("def my_ode(t, x):")
    lines.append("    dxdt = np.zeros(len(x))")
    
    for eq in equations:
        lhs, rhs = parse_ode(eq)
        lines.append(f"    {lhs} = {rhs}")
    
    lines.append("    return dxdt")
    
    return "\n".join(lines)

# Read equations from ode.txt file
with open("identified_ode_explicit.txt", "r") as f:
    equations = [line.strip() for line in f if line.strip()]  # Read non-empty lines

# Generate the ODE function code
ode_function_code = generate_ode_function(equations)

# Save the generated ODE function code to a Python file
with open("multi_arm_ode.py", "w") as f:
    f.write(ode_function_code)

