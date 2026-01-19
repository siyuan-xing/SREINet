import numpy as np
import re

def parse_phi4_equation(equation_text):
    # Replace (xN)' with dxdt[N-1]
    equation_text = equation_text.replace("'", "")
    
    # Replace xN with x[N-1]
    equation_text = re.sub(r'\bx(\d+)\b', lambda m: f"x[{int(m.group(1)) - 1}]", equation_text)
    
    # replace "^" by "**"
    equation_text = equation_text.replace("^", "**")

    # Replace cos and sin with np.cos and np.sin
    #equation_text = equation_text.replace("cos", "np.cos").replace("sin", "np.sin")
    
    # Extract dxdt[N-1] assignment and expression
    lhs, rhs = equation_text.split("=")
    
    # Remove spaces around equals sign for better readability
    lhs = re.sub(r'x\[(\d+)\]', r'dxdt[\1]', lhs.strip())
    rhs = rhs.strip()
    
    return lhs, rhs

def generate_ode_function(equations):
    lines = []
    lines.append("import numpy as np")
    lines.append("def phi4_ode(t, x):")
    lines.append("    dxdt = np.zeros(len(x))")
    
    for eq in equations:
        lhs, rhs = parse_phi4_equation(eq)
        lines.append(f"    {lhs} = {rhs}")
    
    lines.append("    return dxdt")
    
    return "\n".join(lines)

#import excel file, named "nn_configuration.xlsx", using the sheet name "reconstructed_weights"
import pandas as pd
df = pd.read_excel("nn_configurations.xlsx", sheet_name="reconstructed_weights")


ode_function_code = generate_ode_function(df[0])

# Save the generated ODE function code to a Python file
with open("phi_ode.py", "w") as f:
    f.write(ode_function_code)