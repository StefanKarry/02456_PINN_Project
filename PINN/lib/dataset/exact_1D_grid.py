import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import sys
import argparse

# TODO: Adjust this file so it is easier to use in other files. Very complicated right now..
# Should not have hardcoded parameters which must be changed in multiple places in order to work properly.

# --- PARAMETERS ---
DEFAULT_PAR = {
    'WAVE_SPEED': 1.4, #c [MUST MATCH WAVE_SPEED IN train_1d.py!!!]
    'DOMAIN_START': -1.0,
    'DOMAIN_END': 1.0,
    'N_TERMS': 50,
    'T_MAX_PLOT': 2.0,  # Max time for the y-axis
    # Source 1
    'X_CENTER_S1': 0.5,
    'SIGMA_S1': 0.2,

    # Source 2
    'X_CENTER_S2': -0.3,
    'SIGMA_S2': 0.2,

    'SOURCES': 1, # Number of sources to include in the initial condition <<<<<< Change this if coming from predict_1d.py
    'OUTPUT_FILE': "exact_solution_1D_heatmap.png"
}

# Function made by Gemini to make script useable in collected jupyter notebook
def parse_arguments():
    """Parses command-line arguments and updates the configuration."""
    parser = argparse.ArgumentParser(
        description="Calculates and plots the exact solution to the 1D wave equation via Fourier series.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Core Parameters
    parser.add_argument('--wave_speed', type=float, default=DEFAULT_PAR['WAVE_SPEED'],
                        help=f"Wave propagation speed (c). Default: {DEFAULT_PAR['WAVE_SPEED']}")
    parser.add_argument('--n_terms', type=int, default=DEFAULT_PAR['N_TERMS'],
                        help=f"Number of Fourier terms to use. Default: {DEFAULT_PAR['N_TERMS']}")
    parser.add_argument('--t_max_plot', type=float, default=DEFAULT_PAR['T_MAX_PLOT'],
                        help=f"Maximum time (y-axis limit) for the plot. Default: {DEFAULT_PAR['T_MAX_PLOT']}")
    parser.add_argument('--sources', type=int, choices=[1, 2], default=DEFAULT_PAR['SOURCES'],
                        help=f"Number of initial Gaussian sources (1 or 2). Default: {DEFAULT_PAR['SOURCES']}")
    parser.add_argument('--output_file', type=str, default=DEFAULT_PAR['OUTPUT_FILE'],
                        help=f"Name of the output plot file. Default: {DEFAULT_PAR['OUTPUT_FILE']}")

    # Domain Parameters (often fixed for this problem type)
    parser.add_argument('--domain_start', type=float, default=DEFAULT_PAR['DOMAIN_START'],
                        help=f"Domain start position (x_min). Default: {DEFAULT_PAR['DOMAIN_START']}")
    parser.add_argument('--domain_end', type=float, default=DEFAULT_PAR['DOMAIN_END'],
                        help=f"Domain end position (x_max). Default: {DEFAULT_PAR['DOMAIN_END']}")

    # Source 1 Parameters
    parser.add_argument('--x_center_s1', type=float, default=DEFAULT_PAR['X_CENTER_S1'],
                        help=f"Center of Source 1. Default: {DEFAULT_PAR['X_CENTER_S1']}")
    parser.add_argument('--sigma_s1', type=float, default=DEFAULT_PAR['SIGMA_S1'],
                        help=f"Width (sigma) of Source 1. Default: {DEFAULT_PAR['SIGMA_S1']}")

    # Source 2 Parameters (only used if --sources 2)
    parser.add_argument('--x_center_s2', type=float, default=DEFAULT_PAR['X_CENTER_S2'],
                        help=f"Center of Source 2. Default: {DEFAULT_PAR['X_CENTER_S2']}")
    parser.add_argument('--sigma_s2', type=float, default=DEFAULT_PAR['SIGMA_S2'],
                        help=f"Width (sigma) of Source 2. Default: {DEFAULT_PAR['SIGMA_S2']}")


    args = parser.parse_args()
    
    # Merge parsed arguments back into the config dictionary
    config = DEFAULT_PAR.copy()
    for key in config.keys():
        # Keys in DEFAULT_PAR are uppercase, keys in args are lowercase.
        arg_key = key.lower()
        if hasattr(args, arg_key):
             config[key] = getattr(args, arg_key)
    
    return config

def initial_displacement(x, config):
    # Calculate Pulse 1
    psi_1 = np.exp(-((x - config['X_CENTER_S1']) / config['SIGMA_S1'])**2)

    if config['SOURCES'] == 1:
        return psi_1

    else:
    # Calculate Pulse 2
        psi_2 = np.exp(-((x - config['X_CENTER_S2']) / config['SIGMA_S2'])**2)
        return psi_1 + psi_2

def calculate_coefficients(config):
    DOMAIN_START = config['DOMAIN_START']
    DOMAIN_END = config['DOMAIN_END']
    DOMAIN_LENGTH = DOMAIN_END - DOMAIN_START
    N_TERMS = config['N_TERMS']
    # A0 (Normalized by domain length)
    a0, _ = quad(lambda x: initial_displacement(x, config), DOMAIN_START, DOMAIN_END)
    a0 = a0 / DOMAIN_LENGTH
    
    coeffs = [a0]

    for n in range(1, N_TERMS + 1):
        def integrand(x):
            basis = np.cos(n * np.pi * (x - DOMAIN_START) / DOMAIN_LENGTH)
            return initial_displacement(x, config) * basis
            
        an, _ = quad(integrand, DOMAIN_START, DOMAIN_END)
        coeffs.append(an * (2.0 / DOMAIN_LENGTH)) 
    return coeffs


def get_solution_grid(x_array, t_array, A_COEFFICIENTS=None, config=None):
    if A_COEFFICIENTS is None:
        A_COEFFICIENTS = calculate_coefficients(config)
    u_grid = np.full_like(x_array, A_COEFFICIENTS[0])

    for n in range(1, config['N_TERMS'] + 1):
        An = A_COEFFICIENTS[n]
    
        time_part = np.cos(config['WAVE_SPEED'] * n * np.pi * t_array / (config['DOMAIN_END'] - config['DOMAIN_START']))

        space_part = np.cos(n * np.pi * (x_array - config['DOMAIN_START']) / (config['DOMAIN_END'] - config['DOMAIN_START']))
        
        u_grid += An * time_part * space_part
        
    return u_grid


# Create the plot
def plot_solution(X_Grid, T_Grid, U_Grid, config):
    DOMAIN_START = config['DOMAIN_START']
    DOMAIN_END = config['DOMAIN_END']
    T_MAX_PLOT = config['T_MAX_PLOT']
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
    plt.savefig(f"exact_solution_1D_{config['SOURCES']}_sources_heatmap.png", bbox_inches='tight', dpi=300)




def main():
    config = parse_arguments()

    DOMAIN_LENGTH = config['DOMAIN_END'] - config['DOMAIN_START']

    # Pre-calculate coefficients
    print("Calculating Fourier coefficients...")
    A_COEFFICIENTS = calculate_coefficients(config)

    # Define the Grid
    # Define the Grid
    resolution = 200
    x_vals = np.linspace(config['DOMAIN_START'], config['DOMAIN_END'], resolution)
    t_vals = np.linspace(0, config['T_MAX_PLOT'], resolution)
    X_Grid, T_Grid = np.meshgrid(x_vals, t_vals)

    print("Computing solution grid...")
    U_Grid = get_solution_grid(X_Grid, T_Grid, A_COEFFICIENTS, config)

    print("Plotting...")
    plot_solution(X_Grid, T_Grid, U_Grid, config)
    plt.show()

if __name__ == "__main__":
    main()





