import numpy as np

def wave_rectangle_exact(XY, t, f_handle, g_handle, c, Lx, Ly):
    """
    Exact spectral solution of u_tt = c^2 (u_xx + u_yy)
    on a rectangle [0,Lx]×[0,Ly] with Dirichlet boundaries.

    Inputs:
        XY        : N×2 array of (x,y) points
        t         : time
        f_handle  : function f(x,y), initial displacement
        g_handle  : function g(x,y), initial velocity
        Mmax,Nmax : max Fourier modes
        c         : wave speed
        Lx,Ly     : rectangle dimensions

    Output:
        u(XY,t)   : solution evaluated at XY
    """

    x = XY[:,0]
    y = XY[:,1]
    Npts = len(x)

    # Precompute basis functions values
    phi_vals = np.zeros((Mmax, Nmax, Npts))
    for m in range(1, Mmax+1):
        for n in range(1, Nmax+1):
            phi_vals[m-1,n-1,:] = (
                np.sin(m*np.pi*x/Lx) *
                np.sin(n*np.pi*y/Ly)
            )

    # Compute initial projection coefficients
    fmn = np.zeros((Mmax, Nmax))
    gmn = np.zeros((Mmax, Nmax))

    # Quadrature: sample grid
    Nx_quad, Ny_quad = 200, 200
    Xq = np.linspace(0,Lx,Nx_quad)
    Yq = np.linspace(0,Ly,Ny_quad)
    Xg, Yg = np.meshgrid(Xq, Yq)

    fvals = f_handle(Xg, Yg)
    gvals = g_handle(Xg, Yg)

    dx = Lx/(Nx_quad-1)
    dy = Ly/(Ny_quad-1)

    for m in range(1, Mmax+1):
        for n in range(1, Nmax+1):

            phi = np.sin(m*np.pi*Xg/Lx)*np.sin(n*np.pi*Yg/Ly)

            fmn[m-1,n-1] = 4/(Lx*Ly)*np.sum(fvals*phi)*dx*dy
            gmn[m-1,n-1] = 4/(Lx*Ly)*np.sum(gvals*phi)*dx*dy

    # Build final solution
    u = np.zeros(Npts)

    for m in range(1, Mmax+1):
        for n in range(1, Nmax+1):
            lam = (m*np.pi/Lx)**2 + (n*np.pi/Ly)**2
            omega = c*np.sqrt(lam)

            A = fmn[m-1,n-1]
            B = gmn[m-1,n-1] / omega

            phi_xy = phi_vals[m-1,n-1,:]

            u += (A*np.cos(omega*t) + B*np.sin(omega*t)) * phi_xy

    return u

# Example usage:
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Parameters
    Lx, Ly = 1.0, 1.0
    c = 1.0
    Mmax, Nmax = 20, 20
    t = 0.5

    # Initial conditions
    def f_handle(x,y):
        return np.sin(np.pi*x/Lx) * np.sin(np.pi*y/Ly)

    def g_handle(x,y):
        return 0.0*x*y

    # Create grid
    nX, nY = 200, 200
    x = np.linspace(-Lx, Lx, nX)
    y = np.linspace(-Ly, Ly, nY)
    X, Y = np.meshgrid(x, y, indexing="ij")
    XY = np.column_stack((X.ravel(), Y.ravel()))

    # Solve
    u = wave_rectangle_exact(XY, t, f_handle, g_handle,
                             Mmax, Nmax, c, Lx, Ly)
    U = u.reshape(X.shape)

    # Plot
    plt.figure(figsize=(6,4))
    im = plt.imshow(
        U,
        extent=(-Ly,Ly,-Lx,Lx),
        origin="lower",
        aspect="auto"
    )
    plt.xlabel(r"$y$")
    plt.ylabel(r"$x$")
    plt.title(r"$u(x,y,t=%.2f)$" % t)
    plt.colorbar(im, label="u")
    plt.show()
