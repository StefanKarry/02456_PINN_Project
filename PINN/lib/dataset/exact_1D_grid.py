import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

# TODO: Adjust this file so it is easier to use in other files. Very complicated right now..
# Should not have hardcoded parameters which must be changed in multiple places in order to work properly.

# --- PARAMETERS ---

WAVE_SPEED = 1.4 #c [MUST MATCH WAVE_SPEED IN train_1d.py!!!]
DOMAIN_START = -1.0
DOMAIN_END = 1.0
N_TERMS = 50
T_MAX_PLOT = 2.0  # Max time for the y-axis

# Source 1
X_CENTER_S1 = 0.5
SIGMA_S1 = 0.2

# Source 2
X_CENTER_S2 = -0.3
SIGMA_S2 = 0.2

SOURCES = 1 # Number of sources to include in the initial condition <<<<<< Change this if coming from predict_1d.py

def initial_displacement(x, sources = 1):
    # Calculate Pulse 1
    psi_1 = np.exp(-((x - X_CENTER_S1) / SIGMA_S1)**2)

    if sources == 1:
        return psi_1

    else:
    # Calculate Pulse 2
        psi_2 = np.exp(-((x - X_CENTER_S2) / SIGMA_S2)**2)
        return psi_1 + psi_2

DOMAIN_LENGTH = DOMAIN_END - DOMAIN_START # Equals 2.0

def calculate_coefficients(sources = SOURCES):
    # A0 (Normalized by domain length)
    a0, _ = quad(lambda x: initial_displacement(x, sources), DOMAIN_START, DOMAIN_END)
    a0 = a0 / DOMAIN_LENGTH
    
    coeffs = [a0]

    for n in range(1, N_TERMS + 1):
        def integrand(x):
            basis = np.cos(n * np.pi * (x - DOMAIN_START) / DOMAIN_LENGTH)
            return initial_displacement(x, sources) * basis
            
        an, _ = quad(integrand, DOMAIN_START, DOMAIN_END)
        coeffs.append(an * (2.0 / DOMAIN_LENGTH)) 
    return coeffs

# Pre-calculate coefficients
print("Calculating Fourier coefficients...")
A_COEFFICIENTS = calculate_coefficients(sources=SOURCES)

def get_solution_grid(x_array, t_array):
    u_grid = np.full_like(x_array, A_COEFFICIENTS[0])

    for n in range(1, N_TERMS + 1):
        An = A_COEFFICIENTS[n]
    
        time_part = np.cos(WAVE_SPEED * n * np.pi * t_array / DOMAIN_LENGTH)

        space_part = np.cos(n * np.pi * (x_array - DOMAIN_START) / DOMAIN_LENGTH)
        
        u_grid += An * time_part * space_part
        
    return u_grid


# Define the Grid
resolution = 200
x_vals = np.linspace(DOMAIN_START, DOMAIN_END, resolution)
t_vals = np.linspace(0, T_MAX_PLOT, resolution)
X_Grid, T_Grid = np.meshgrid(x_vals, t_vals)

print("Computing solution grid...")
U_Grid = get_solution_grid(X_Grid, T_Grid)

# Create the plot
print("Plotting...")
fig, ax = plt.subplots(figsize=(8, 6))

cplot = ax.pcolormesh(X_Grid, T_Grid, U_Grid, cmap='turbo', shading='auto')

ax.set_xlim(DOMAIN_START, DOMAIN_END)
ax.set_ylim(0, T_MAX_PLOT)
ax.set_xlabel("Position ($x$)")
ax.set_ylabel("Time ($t$)")
ax.set_title("Exact Solution $u(x,t)$")

#Plotting the line indicating training domain limit
if T_MAX_PLOT > 1.0:
    ax.axhline(y=T_MAX_PLOT - (T_MAX_PLOT - 1.0), color='white', linestyle='--', label='Out of\nTraining\nDomain')
    ax.legend(loc = 'lower left')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

cbar = plt.colorbar(cplot, ax=ax)
cbar.set_label("Displacement $u(x,t)$")

plt.tight_layout()
plt.savefig("exact_solution_1D_heatmap.png", bbox_inches='tight', dpi=300)
