import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

# --- PARAMETERS ---
WAVE_SPEED = 1.4       # c
DOMAIN_START = -1.0
DOMAIN_END = 1.0
N_TERMS = 50           # Fourier series truncation
T_MAX_PLOT = 1.0       # Max time for the y-axis
X_CENTER = 1         # x_0
SIGMA = 0.2            # sigma

# --- PHYSICS & SOLUTION LOGIC (Same as before) ---
def initial_displacement(x):
    return np.exp(-((x - X_CENTER) / SIGMA)**2)

def calculate_coefficients():
    # A0
    a0, _ = quad(initial_displacement, DOMAIN_START, DOMAIN_END)
    # a0 *= 0.5
    
    coeffs = [a0]
    # An for n=1..N
    for n in range(1, N_TERMS + 1):
        def integrand(x):
            return initial_displacement(x) * np.cos(n * np.pi * x)
        an, _ = quad(integrand, DOMAIN_START, DOMAIN_END)
        coeffs.append(an)
    return coeffs

# Pre-calculate coefficients
print("Calculating Fourier coefficients...")
A_COEFFICIENTS = calculate_coefficients()

def get_solution_grid(x_array, t_array):
    """
    Calculates u(x,t) for 2D grids of x and t using the series solution.
    """
    # Initialize with A0 term
    u_grid = np.full_like(x_array, A_COEFFICIENTS[0])
    
    # Add series terms
    # Using broadcasting for efficiency: 
    # t_array varies along axis 0, x_array along axis 1
    for n in range(1, N_TERMS + 1):
        An = A_COEFFICIENTS[n]
        # cos(c * n * pi * t)
        time_part = np.cos(WAVE_SPEED * n * np.pi * t_array)
        # cos(n * pi * x)
        space_part = np.cos(n * np.pi * x_array)
        
        u_grid += An * time_part * space_part
        
    return u_grid

# --- PLOTTING ROUTINE ---

# 1. Define the Grid
resolution = 200
x_vals = np.linspace(DOMAIN_START, DOMAIN_END, resolution)
t_vals = np.linspace(0, T_MAX_PLOT, resolution)
X_Grid, T_Grid = np.meshgrid(x_vals, t_vals)

print("Computing solution grid...")
U_Grid = get_solution_grid(X_Grid, T_Grid)
print(U_Grid)
# 2. Create the Plot
print("Plotting...")
fig, ax = plt.subplots(figsize=(8, 6))

# pcolormesh creates the heatmap
# vmin/vmax center the colormap if needed, or let it scale auto
# 'shading="auto"' ensures correct grid alignment
cplot = ax.pcolormesh(X_Grid, T_Grid, U_Grid, cmap='viridis', shading='auto')

# 3. Style to match your image
ax.set_xlim(DOMAIN_START, DOMAIN_END)
ax.set_ylim(0, T_MAX_PLOT)
ax.set_xlabel("Position ($x$)")
ax.set_ylabel("Time ($t$)")
ax.set_title("Exact Solution $u(x,t)$ Space-Time Heatmap")

# Remove standard spines to match the 'clean' look of the provided image if desired
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add a colorbar to show amplitude
cbar = plt.colorbar(cplot, ax=ax)
cbar.set_label("Displacement $u$")

plt.tight_layout()
plt.show()