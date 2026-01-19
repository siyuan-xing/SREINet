import numpy as np
#Credit: https://github.com/dynamicslab/pysindy/blob/master/docs/examples/index.rst


def linear_3D(t, x):
    return [-0.1 * x[0] +   2 * x[1] +   0 * x[2], 
              -2 * x[0] - 0.1 * x[1] +   0 * x[2], 
               0 * x[0] +   0 * x[1] - 0.3 * x[2]]

def lorenz(t, x, sigma=10.0, beta=2.66667, rho=28.0):
    """
    The Lorenz system.
        $$
        \begin{align*}
        \frac{dx}{dt} &= \sigma(y - x) \\
        \frac{dy}{dt} &= x(\rho - z) - y \\
        \frac{dz}{dt} &= xy - \beta z
        \end{align*}
        $$
        Here, $x$, $y$, and $z$ make up the system state, $t$ represents time, and $\sigma$, $\rho$, and $\beta$ are system parameters with the following typical values:
        - $\sigma > 0$ is the Prandtl number,
        - $\rho$ is the Rayleigh number, and
        - $\beta > 0$ is a geometric factor.
    """
    return  (sigma * (x[1] - x[0]), 
             x[0] * (rho - x[2]) - x[1], 
             x[0] * x[1] - beta * x[2])

def lorenz96(t, x, F=8):
    """
    The Lorenz-96 model with n variables. The model is governed by a set of $n$ differential equations:
        $$
        \dot{x}_j = x_{j-1} \cdot (x_{j+1} - x_{j-2}) - x_j + F, \quad j = 1, 2, \ldots, n
        $$
        where:
        - $ \dot{x}_j $ denotes the rate of change of the variable $x_j$ over time.
        - $ x_{j-1} $, $ x_{j+1} $, and $ x_{j-2} $ are the values of the neighboring sectors, implementing the model's spatial coupling.
        - $ F $ is a constant forcing term, representing external influences on the system (e.g., solar heating).
    """
    n = x.shape[0]
    dxdt = np.zeros(n)    
    for j in range(n):
        dxdt[j] = x[(j-1) % n] * (x[(j+1) % n] - x[(j-2) % n]) - x[j] + F
    return dxdt

def duffing(t, x, p=[0.2, 0.05, 1]):
    """
    The duffing oscillator.
    """
    return (x[1], -p[0] * x[1] - p[1] * x[0] - p[2] * x[0] ** 3)

def meanfield(t, x, mu=0.01):
    """ **Mean field model from Noack et al. 2003**
        $$
        \begin{align*}
        \frac{dx_0}{dt} &= \mu x_0 - x_1 - x_0 x_2 \\
        \frac{dx_1}{dt} &= \mu x_1 + x_0 - x_1 x_2 \\
        \frac{dx_2}{dt} &= -x_2 + x_0^2 + x_1^2
        \end{align*}
        $$
        where:
        - $ x_0 $, $ x_1 $, and $ x_2 $ represent state variables of the system, which could correspond to modes or significant dynamical features of the wake flow.
        - $ \mu $ is a parameter affecting the dynamics, which could represent aspects such as the Reynolds number or other control parameters in the context of fluid flow around a cylinder.
    """
    return (mu * x[0] - x[1] - x[0] * x[2], 
            mu * x[1] + x[0] - x[1] * x[2], 
            -x[2] + x[0] ** 2 + x[1] ** 2)

def mhd(t, x, nu=0.5, mu=0.4, sigma=3):
    """ **Carbone and Veltri triadic MHD model**
        $$
        \begin{align*}
        \dot{x_0} &= -2 \nu x_0 + 4.0 \left( x_1 x_2 - x_4 x_5 \right), \\
        \dot{x_1} &= -5 \nu x_1 - 7.0 \left( x_0 x_2 - x_3 x_5 \right), \\
        \dot{x_2} &= -9 \nu x_2 + 3.0 \left( x_0 x_1 - x_3 x_4 \right), \\
        \dot{x_3} &= -2 \mu x_3 + 2.0 \left( x_4 x_1 - x_2 x_3 \right), \\
        \dot{x_4} &= -5 \mu x_4 + \sigma x_5 + 5.0 \left( x_2 x_3 - x_0 x_5 \right), \\
        \dot{x}_5 &= -9 \mu x_5 + \sigma x_4 + 9.0 \left( x_3 x_0 - x_1 x_3 \right).
        \end{align*}
        $$
    """
    return  (-2 * nu * x[0] + 4.0 * (x[1] * x[2] - x[4] * x[5]),
            -5 * nu * x[1] - 7.0 * (x[0] * x[2] - x[3] * x[5]),
            -9 * nu * x[2] + 3.0 * (x[0] * x[1] - x[3] * x[4]),
            -2 * mu * x[4] + 2.0 * (x[5] * x[1] - x[2] * x[4]),
            -5 * mu * x[4] + sigma * x[5] + 5.0 * (x[2] * x[3] - x[0] * x[5]),
            -9 * mu * x[5] + sigma * x[4] + 9.0 * (x[4] * x[0] - x[1] * x[3])
            )
    
def lorenz_4d(t, x, r= 2, mu=0.1):
    """**Lorenz 4D**
        The equations of the Lorenz 4D system reads
        $$
        \begin{align*}
        \dot{x} &= -10 (x - y), \\
        \dot{y} &= r x - y - x z, \\
        \dot{z} &= -\frac{8}{3} z + \mu w + x y, \\
        \dot{w} &= -\frac{8}{3} w - \mu z.
        \end{align*}
        $$
        where, $x$, $y$, $z$, and $w$ represent the state variables of the system, which could be related to physical quantities such as velocity and temperature gradients in the context of atmospheric convection. 
        The parameters $r$ and $\mu$ play crucial roles in the system's dynamics, affecting the stability and nature of the attractors.
    """
    return (-10 * (x[0] - x[1]),
            r * x[0] - x[1] - x[0] * x[2],
            -8/3 * x[2] + mu * x[3] +  x[0] * x[1],
            -8/3 * x[3] - mu * x[2])

def shimizu_morioka(t, x, mu=0.1, a=2, b=0.5):
    """ **Shimizu-Morioka Model**
        The equations of the Shimizu-Morioka Model are as follows:
        $$
        \begin{align*}
        \dot{x} &= y, \\
        \dot{y} &= -a y + x - x z, \\
        \dot{z} &= w, \\
        \dot{w} &= -b w + \mu z + x^2,
        \end{align*}
        $$
        where $x$, $y$, $z$, and $w$ represent the state variables of the system. 
        The parameters $a$, $b$, and $\mu$ play crucial roles in determining the system's behavior, influencing the stability and type of orbits that the system exhibits.
    """
    return (x[1],
            -a * x[1] + x[0] - x[0] * x[2],
            x[3],
            -b * x[3] + mu * x[2] + x[0] ** 2)

def kuramoto(t,theta, wn:np.array, K=2.0, h=0.2):
    """
    The model parameters are obatined from the paper: https://arxiv.org/abs/1809.02448
    """
    N = theta.shape[0]
    theta_diffs = np.subtract.outer(theta, theta)
    coupling = np.sin(theta_diffs).sum(axis=1)
    forcing = np.sin(theta)
    return wn + K*coupling/N + h*forcing


def discrete_phi_quartic(t,x, C=2.0, N=100):
    #discrete phi^4 equation - free boundary conditions are used 
    # N: the number of grid points
    un = x[:N]
    un_dot = x[N:]


    un_ddot = np.zeros_like(un)

    un_plus_one = np.roll(un, -1)
    un_minus_one = np.roll(un, 1)

    un_ddot[1:N-1] = C * (un_plus_one[1:N-1] + un_minus_one[1:N-1] - 2 * un[1:N-1]) + 2*(un[1:N-1] - un[1:N-1]**3)

    #imposing free boundary conditions

    un_ddot[0] = C * (un[1] - un[0]) + 2 * (un[0] - un[0]**3)

    un_ddot[N-1] = C * (un[N-2] - un[N-1]) + 2 * (un[N-1] - un[N-1]**3)


    return np.concatenate([un_dot, un_ddot])


def DNLS(t,un, C=2.0, N=100):
    #discrete nonlinear Schrodinger equation - with free boundary conditions
    # N: the number of grid points

    un_plus_one = np.roll(un, -1)
    un_minus_one = np.roll(un, 1)

    un_dot = 1j*C * (un_plus_one + un_minus_one - 2 * un) + 1j*(un*np.abs(un)**2)

    return un_dot


def DNLS_split(t,un, C=2.0, N=100):
    #discrete nonlinear Schrodinger equation
    # N: the number of grid points
    an = un[:N]
    bn = un[N:]

    an_plus_one = np.roll(an, -1)
    an_minus_one = np.roll(an, 1)
    bn_plus_one = np.roll(bn, -1)
    bn_minus_one = np.roll(bn, 1)
    
    #un_dot = 1j*C * (un_plus_one + un_minus_one - 2 * un) + 1j*(un*np.abs(un)**2)

    an_dot = -C*(bn_plus_one + bn_minus_one - 2*bn) - bn*(an**2 +bn**2)
    bn_dot = C*(an_plus_one + an_minus_one - 2*an) + an*(an**2 +bn**2)

    return np.concatenate([an_dot, bn_dot])


def discrete_abolowitz_ladik(t, psi_n, N=20):
    
    psi_n_plus_one = np.roll(psi_n, -1)
    psi_n_minus_one = np.roll(psi_n, 1)

    un_dot = 0.5j * (psi_n_plus_one - 2 * psi_n + psi_n_minus_one) + 0.5j*(psi_n_plus_one + psi_n_minus_one)*np.abs(psi_n)**2 
    
    return un_dot



def fermi_pasta_ulam(t,x, b=0.7, N=10):
    un= x[:N]
    un_dot = x[N:]

    un_plus_one = np.roll(un, -1)
    un_minus_one = np.roll(un, 1)

    un_ddot = un_plus_one + un_minus_one - 2 * un + b*(un_plus_one - un)**3 - b*(un - un_minus_one)**3

    #impose boundary conditions
    #u_{-1} =0  
    un_ddot[0] = un[1] - 2 * un[0] + b*(un[1] - un[0])**3 - b*(un[0] - 0)**3

    #u_{N+1} =0
    un_ddot[-1] = un[-2] - 2 * un[-1] + b*(0 - un[-1])**3 - b*(un[-1] - un[-2])**3

    return np.concatenate([un_dot, un_ddot])