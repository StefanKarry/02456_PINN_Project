import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- PARAMETERS ---
# Speed of the wave (c)
WAVE_SPEED = 1.4
# Domain limits [-L, L] -> [-1, 1]
DOMAIN_START = -1.0
DOMAIN_END = 1.0
# Truncation size for the infinite series (N)
N_TERMS = 50
# Time to simulate until
T_MAX = 8.0
# Position of the initial bump (x_0 in your image)
X_CENTER = 0.5 
# Gaussian width (sigma in your image)
# Note: The image uses exp(-((x-x0)/sigma)^2)
SIGMA = 0.2

# --- INITIAL CONDITION (psi(x)) ---
def initial_displacement(x):
    """
    Defines the initial displacement function, psi(x).
    Matches the formula in the image: psi(x) = exp(-((x - x0)/sigma)^2)
    """
    return np.exp(-((x - X_CENTER) / SIGMA)**2)

# --- FOURIER COEFFICIENT CALCULATION ---

def calculate_A0():
    """
    Calculates the A_0 coefficient (the average value).
    A_0 = 1/2 * integral(-1 to 1) of psi(x) dx
    """
    # quad returns (result, absolute_error)
    integral_result, _ = quad(initial_displacement, DOMAIN_START, DOMAIN_END)
    return 0.5 * integral_result

def calculate_An(n):
    """
    Calculates the A_n coefficients for n >= 1.
    A_n = integral(-1 to 1) of psi(x) * cos(n*pi*x) dx
    """
    if n == 0:
        # A_0 should be calculated separately, but handled here for safety
        return 0 
    
    def integrand(x):
        return initial_displacement(x) * np.cos(n * np.pi * x)
    
    integral_result, _ = quad(integrand, DOMAIN_START, DOMAIN_END)
    return integral_result

# --- SERIES SOLUTION FUNCTION ---

# Pre-calculate all coefficients up to N_TERMS
A_COEFFICIENTS = [calculate_A0()] + [calculate_An(n) for n in range(1, N_TERMS + 1)]

def exact_solution(x, t):
    """
    Calculates the exact solution u(x, t) as a truncated Fourier series.
    u(x, t) = A_0 + sum_{n=1}^{N} A_n * cos(c*n*pi*t) * cos(n*pi*x)
    """
    u_xt = A_COEFFICIENTS[0] # The A_0 term

    # Add the series terms (n=1 to N_TERMS)
    for n in range(1, N_TERMS + 1):
        An = A_COEFFICIENTS[n]
        # Angular frequency: omega_n = c * n * pi
        time_dependence = np.cos(WAVE_SPEED * n * np.pi * t)
        space_dependence = np.cos(n * np.pi * x)
        
        u_xt += An * time_dependence * space_dependence
        
    return u_xt



# --- VISUALIZATION (Animation) ---

print(f"--- 1D Wave Equation Solver ---")
print(f"Wave Speed (c): {WAVE_SPEED}")
print(f"Initial Pulse Center (x_0): {X_CENTER}")
print(f"Pulse Width (sigma): {SIGMA}")
print(f"Series Truncated at N = {N_TERMS} terms.")
print(f"A_0 (Average Displacement): {A_COEFFICIENTS[0]:.4f}")
print("Starting simulation...")

# Setup plot
fig, ax = plt.subplots(figsize=(10, 5))
x_grid = np.linspace(DOMAIN_START, DOMAIN_END, 500)
line, = ax.plot(x_grid, initial_displacement(x_grid), color='royalblue')

# Set plot limits
max_disp = np.max(initial_displacement(x_grid)) * 1.1 
min_disp = np.min(initial_displacement(x_grid)) * 1.1
ax.set_ylim(min(0, min_disp), max_disp) 
ax.set_xlim(DOMAIN_START, DOMAIN_END)
ax.set_title(f"Exact Solution: Gaussian Pulse $\sigma={SIGMA}$ at $x_0={X_CENTER}$")
ax.set_xlabel("Position $x$")
ax.set_ylabel("Displacement $u(x, t)$")
ax.grid(True, linestyle='--', alpha=0.6)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

def update(frame):
    """Update function for the animation."""
    t = frame / 50.0 # Time step
    
    # Calculate the solution at the current time t
    u_current = exact_solution(x_grid, t)
    
    # Update the plot line
    line.set_ydata(u_current)
    
    # Update the time display
    time_text.set_text(f'Time: $t = {t:.2f}$')
    
    return line, time_text

# Create the animation
# frames: controls the number of steps in the animation
# interval: delay between frames in milliseconds
ani = FuncAnimation(fig, update, frames=int(T_MAX * 50), 
                    interval=50, blit=True)

plt.show()

# --- Explanation of the Physics and Solution ---
print("\n--- Physics Interpretation ---")
print("1. **Boundary Conditions (BCs):** $\\frac{\\partial u}{\\partial x}(-1, t) = \\frac{\\partial u}{\\partial x}(1, t) = 0$.")
print("   - This means the slope (or strain, in a rod) is zero at the boundaries.")
print("   - Physically, this represents *free ends* for a vibrating string (though a bit non-physical for a simple string) or, more commonly, the ends of a longitudinally vibrating rod or a column of air with two *closed ends* (where the pressure is maximized, but the velocity is zero).")
print("   - The wave reflects perfectly at the boundaries without changing polarity (it reflects as a peak if it hits a peak).")
print("2. **Initial Condition:** $\\frac{\\partial u}{\\partial t}(x, 0) = 0$.")
print("   - The initial velocity is zero. The wave is only started by displacement, not by being 'plucked' or pushed.")
print("3. **Eigenfunctions:** The boundary conditions dictate the spatial pattern is $\\cos(n\\pi x)$, which is characteristic of fixed-fixed or free-free systems. The fundamental mode ($n=1$) has a full cosine wave across the domain.")
print("4. **Behavior:** The initial bump at $X_{CENTER}$ splits into two pulses that travel in opposite directions. When they hit the boundaries at $x=-1$ and $x=1$, they reflect back and combine to form standing wave patterns (the $\\cos(c n\\pi t)$ term).")