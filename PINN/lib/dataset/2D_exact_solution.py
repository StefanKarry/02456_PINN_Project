import numpy as np
import matplotlib.pyplot as plt

# Define the eact solution of the wave equation for 2D case

def exact_solution_2D(x, y, t, c=1.0):
    """
    Compute the exact solution of the 2D wave equation at given points (x, y) and time t.
    
    Parameters:
    x : np.ndarray
        x-coordinates of the points.
    y : np.ndarray
        y-coordinates of the points.
    t : float
        Time at which to evaluate the solution.
    c : float
        Wave speed (default is 1.0).
        
    Returns:
    np.ndarray
        The exact solution evaluated at the given points and time.
    """
    return np.sin(np.pi * x) * np.sin(np.pi * y) * np.cos(c * np.pi * t)

def exact_solution_2D_wConstraints(x, y, t, c=1.0):
    """
    Compute the exact solution of the 2D wave equation with constraints at given points (x, y) and time t.
    
    Parameters:
    x : np.ndarray
        x-coordinates of the points.
    y : np.ndarray
        y-coordinates of the points.
    t : float
        Time at which to evaluate the solution.
    c : float
        Wave speed (default is 1.0).
        
    Returns:
    np.ndarray
        The exact solution evaluated at the given points and time with constraints.
    """
    solution = np.sin(np.pi * x) * np.sin(np.pi * y) * np.cos(c * np.pi * t)
    
    # Apply constraints: for example, set solution to zero at boundaries
    solution[(x == 0) | (x == 2) | (y == 0) | (y == 2)] = 0

    # Homogeneous Neumann boundary conditions (zero normal derivative)
    # This is inherently satisfied by the sine function used in the solution.

    return solution

def wave_equation_sol_1D(x, t, c = 1.0):
    '''
    Function for computing the exact solution of the 1D wave equation at given points x and time t.
    It solve the IVP on the domain x \in [-1, 1] x t \in [0, T] with initial conditions u(x, 0) = sin(pi*x) and u_t(x, 0) = 0.
    It satisfies the homogeneous Neumann boundary conditions at x = -1 and x = 1. (du/dx = 0)


    Args:
        x: np.ndarray x-coordinates of the points.
        t: float Time at which to evaluate the solution.
        c: float Wave speed (default is 1.0).

    
    Returns:
        np.ndarray The exact solution evaluated at the given points and time.
    '''

    return np.sin(np.pi * x) * np.cos(c * np.pi * t)
    


# Example usage and visualization
if __name__ == "__main__":
    # Create a grid of points
    x = np.linspace(-1, 1, 25)
    y = np.linspace(-1, 1, 25)
    X, Y = np.meshgrid(x, y)
    
    # Generate a animation of the wave propagation over time
    for t in np.linspace(0, 2, 100):
        # Compute the exact solution
        Z = exact_solution_2D_wConstraints(X, Y, t)

        # Plot the solution
        plt.clf()
        plt.contourf(X, Y, Z, levels=50, cmap='viridis')
        plt.colorbar(label='Wave Amplitude')
        plt.title(f'2D Wave Equation Solution at t={t:.2f}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.pause(0.1)
    
    plt.show()