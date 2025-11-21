import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

# --- PARAMETERS ---

WAVE_SPEED = 1.4       # c
DOMAIN_START = -1.0
DOMAIN_END = 1.0
N_TERMS = 50           # Fourier series truncation
T_MAX_PLOT = 1.0   # Max time for the y-axis

# --- SOURCE 1 ---     
X_CENTER_S1 = 0.5         # x_0
SIGMA_S1 = 0.2            # sigma

# --- SOURCE 2 ---
X_CENTER_S2 = 0 
SIGMA_S2 = 0.2

SOURCES = 1 # Number of sources to include in the initial condition

# --- PHYSICS & SOLUTION LOGIC (Same as before) ---
def initial_displacement(x, sources = 1):
    # Calculate Pulse 1
    psi_1 = np.exp(-((x - X_CENTER_S1) / SIGMA_S1)**2)

    if sources == 1:
        return psi_1

    else:
    # Calculate Pulse 2
        psi_2 = np.exp(-((x - X_CENTER_S2) / SIGMA_S2)**2)
        # Linearity: The total displacement is just the sum
        return psi_1 + psi_2

DOMAIN_LENGTH = DOMAIN_END - DOMAIN_START # Equals 2.0

def calculate_coefficients(sources = SOURCES):
    # A0 (Normalized by domain length)
    a0, _ = quad(lambda x: initial_displacement(x, sources), DOMAIN_START, DOMAIN_END)
    a0 = a0 / DOMAIN_LENGTH  # Important normalization for A0
    
    coeffs = [a0]
    
    # An for n=1..N
    for n in range(1, N_TERMS + 1):
        def integrand(x):
            # CORRECTION HERE: Use (x - start) / length
            basis = np.cos(n * np.pi * (x - DOMAIN_START) / DOMAIN_LENGTH)
            return initial_displacement(x, sources) * basis
            
        an, _ = quad(integrand, DOMAIN_START, DOMAIN_END)
        # Standard Fourier Cosine normalization is 2/L
        coeffs.append(an * (2.0 / DOMAIN_LENGTH)) 
        
    return coeffs

# Pre-calculate coefficients
print("Calculating Fourier coefficients...")
A_COEFFICIENTS = calculate_coefficients(sources=SOURCES)

def get_solution_grid(x_array, t_array):
    u_grid = np.full_like(x_array, A_COEFFICIENTS[0])

    for n in range(1, N_TERMS + 1):
        An = A_COEFFICIENTS[n]
        
        # CORRECTION HERE: Time freq must also match the length scale
        # omega_n = c * n * pi / L
        time_part = np.cos(WAVE_SPEED * n * np.pi * t_array / DOMAIN_LENGTH)
        
        # CORRECTION HERE: Space basis must match the integration basis
        space_part = np.cos(n * np.pi * (x_array - DOMAIN_START) / DOMAIN_LENGTH)
        
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