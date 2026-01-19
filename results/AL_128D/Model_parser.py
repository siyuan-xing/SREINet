import numpy as np
import re

def parse_AL_equation(equation_text):
    # Replace (xN)' with dxdt[N-1]
    equation_text = equation_text.replace("'", "")
    
    # Replace aN with a[N-1]
    equation_text = re.sub(r'\ba(\d+)\b', lambda m: f"a[{int(m.group(1)) - 1}]", equation_text)
    # Replace bN with b[N-1]
    equation_text = re.sub(r'\bb(\d+)\b', lambda m: f"b[{int(m.group(1)) - 1}]", equation_text)

    # replace "^" by "**"
    equation_text = equation_text.replace("^", "**")
    # replace "(a" or "(b)" by "a" or "b"
    equation_text = equation_text.replace("(a", "dadt")
    equation_text = equation_text.replace("(b", "dbdt")
    #equation_text = equation_text.replace("(d", "d")
    # replace "])" by "]"
    equation_text = equation_text.replace("])", "]")
    
    # Extract dxdt[N-1] assignment and expression
    lhs, rhs = equation_text.split("=")
    
    # Remove spaces around equals sign for better readability
    lhs = re.sub(r'x\[(\d+)\]', r'dxdt[\1]', lhs.strip())
    rhs = rhs.strip()
    
    return lhs, rhs

def generate_ode_function(equations):
    lines = []
    lines.append("import numpy as np")
    lines.append("def AL_ode(t ,x):")
    lines.append("    N = len(x)//2")
    lines.append("    a = x[:N]")
    lines.append("    b = x[N:]")
    lines.append("    dadt = np.zeros(N)")
    lines.append("    dbdt = np.zeros(N)")
     
    for eq in equations:
        lhs, rhs = parse_AL_equation(eq)
        lines.append(f"    {lhs} = {rhs}")
    
    lines.append("    return np.concatenate([dadt, dbdt])")
    
    return "\n".join(lines)

#import excel file, named "nn_configuration.xlsx", using the sheet name "reconstructed_weights"
import pandas as pd
df = pd.read_excel("nn_configurations.xlsx", sheet_name="reconstructed_weights")


ode_function_code = generate_ode_function(df[0])

# Save the generated ODE function code to a Python file
with open("AL_ode.py", "w") as f:
    f.write(ode_function_code)