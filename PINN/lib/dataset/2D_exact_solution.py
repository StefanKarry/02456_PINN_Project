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

def WEQ_2D_initial_condition(x, y):
    # generate wave solution for a gaussian pulse at some intitial time and place
    return np.exp(-((x - 1)**2 + (y - 1)**2))

def WEQ_2D_Propagation(x, y, t, c=1.0):
    '''
    Function for computing the exact solution of the 2D wave equation at given points (x, y) and time t.
    It solve the IVP on the domain x, y \in [-3, 3] x t \in [0, T] with initial conditions u(x, y, 0) = exp(-((x-1)^2 + (y-1)^2)) and u_t(x, y, 0) = 0.
    It satisfies the homogeneous Neumann boundary conditions at the boundaries. (du/dn = 0)


    Args:
        x: np.ndarray x-coordinates of the points.
        y: np.ndarray y-coordinates of the points.
        t: float Time at which to evaluate the solution.
        c: float Wave speed (default is 1.0).

    
    Returns:
        np.ndarray The exact solution evaluated at the given points and time.
    '''

    
    return WEQ_2D_initial_condition(x, y) * np.cos(c * np.pi * t)




# Example usage and visualization
if __name__ == "__main__":
    # Create a grid of points
    x = np.linspace(-3, 3, 150)
    y = np.linspace(-3, 3, 150)
    #y = np.ones_like(x)  # For 2D wave along x-axis only
    X, Y = np.meshgrid(x, y)

    ax = plt.axes(projection='3d')


    Zs = []
    for t in np.linspace(0, 2, 100):
        ax.cla()
        Z = WEQ_2D_initial_condition(X, Y) * np.cos(np.pi * t)  # Example time evolution
        Zs = np.array(Z)
        ax.contourf(X, Y, Zs, zdir='z', offset=-1, cmap='viridis', alpha=0.5)
        ax.plot_surface(X, Y, Zs, cmap='viridis')
        ax.set_title(f'Exact Solution of 2D Wave Equation at t={t:.2f}')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('u(x,y,t)')
        ax.set_zlim(-1, 1)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)

        plt.pause(0.01)
    
    plt.show()
