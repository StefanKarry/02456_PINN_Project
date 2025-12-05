import numpy as np
from numpy.polynomial.legendre import leggauss   # Gauss–Legendre nodes/weights
from scipy.special import lpmv                   # associated Legendre P_l^m


def wave1_sphere_exact(XYZ, t, f_handle, g_handle, Lmax, c, R):
    """
    Exact spectral solution of u_tt = c^2 Δ_{S^2_R} u on the sphere.

    Parameters
    ----------
    XYZ : (N, 3) array
        Cartesian coordinates of query points on the sphere of radius R.
    t : float
        Time at which to evaluate the solution.
    f_handle : callable
        f_handle(x, y, z) -> initial displacement on the sphere.
    g_handle : callable
        g_handle(x, y, z) -> initial velocity on the sphere.
    Lmax : int
        Maximum spherical–harmonic degree (bandlimit / resolution).
    c : float
        Wave speed along the surface.
    R : float
        Sphere radius.

    Returns
    -------
    u : (N,) ndarray (real)
        Solution u(XYZ_i, t).
    """
    # ------------------ quadrature grid ------------------
    Ntheta = Lmax + 1          # exact for bandlimit Lmax in μ
    Nphi   = 2 * Lmax + 1      # exact for bandlimit Lmax in φ

    # μ = cosθ in [-1,1], with Gauss–Legendre weights
    mu, w_mu = gausslegendre(Ntheta)
    theta_q = np.arccos(mu)

    # uniform φ in [0, 2π)
    phi_q = np.linspace(0.0, 2.0 * np.pi, Nphi, endpoint=False)

    # sample f and g on quadrature grid (x,y,z)
    TH, PH = np.meshgrid(theta_q, phi_q, indexing='ij')   # Ntheta x Nphi
    XYZq = sph2cartR(TH.ravel(), PH.ravel(), R)           # (Ntheta*Nphi, 3)

    fq = f_handle(XYZq[:, 0], XYZq[:, 1], XYZq[:, 2]).reshape(Ntheta, Nphi)
    gq = g_handle(XYZq[:, 0], XYZq[:, 1], XYZq[:, 2]).reshape(Ntheta, Nphi)

    # ------------------ forward SHT: compute f_lm, g_lm ------------------
    dphi = 2.0 * np.pi / Nphi

    # m indices -Lmax,...,Lmax
    mvals = np.arange(-Lmax, Lmax + 1)
    n_m   = mvals.size

    # exp(-i m φ) for each φ, each m
    Ephi = np.exp(-1j * np.outer(phi_q, mvals))   # (Nphi, 2L+1)

    # Fourier in φ: Fhat(θ_i, m) = ∑_j f(θ_i,φ_j) e^{-i m φ_j} dφ
    Fhat = fq @ Ephi * dphi          # (Ntheta, 2L+1)
    Ghat = gq @ Ephi * dphi

    flm = np.zeros((Lmax + 1, n_m), dtype=complex)
    glm = np.zeros((Lmax + 1, n_m), dtype=complex)

    for ell in range(Lmax + 1):
        # Associated Legendre P_l^m(mu) for all m and quadrature nodes
        # P_l^m(μ) with Condon–Shortley phase is what MATLAB's legendre uses.
        for m in range(0, ell + 1):
            # normalization factor N_lm
            Nlm = np.sqrt((2.0 * ell + 1.0) / (4.0 * np.pi) *
                          factratio(ell - m, ell + m))

            col_pos = m + Lmax        # index for m
            col_neg = -m + Lmax       # index for -m

            plm_vec = lpmv(m, ell, mu)     # shape (Ntheta,)

            flm_pos = Nlm * np.dot(w_mu, plm_vec * Fhat[:, col_pos])
            glm_pos = Nlm * np.dot(w_mu, plm_vec * Ghat[:, col_pos])

            flm[ell, col_pos] = flm_pos
            glm[ell, col_pos] = glm_pos

            if m > 0:
                # Y_{l,-m} = (-1)^m conj(Y_{l,m})  =>  f_{l,-m} = (-1)^m conj(f_{l,m})
                flm[ell, col_neg] = (-1)**m * np.conj(flm_pos)
                glm[ell, col_neg] = (-1)**m * np.conj(glm_pos)

    # ------------------ exact modal time evolution ------------------
    ulm = np.zeros_like(flm, dtype=complex)
    for ell in range(Lmax + 1):
        omega = (c / R) * np.sqrt(ell * (ell + 1.0))
        if ell == 0 or np.abs(omega) < 1e-15:
            ulm[ell, :] = flm[ell, :] + glm[ell, :] * t
        else:
            ulm[ell, :] = (flm[ell, :] * np.cos(omega * t) +
                           (glm[ell, :] / omega) * np.sin(omega * t))

    # ------------------ synthesize u at requested XYZ nodes ------------------
    XYZ = enforce_radius(XYZ, R)      # project to radius R
    x, y, z = XYZ[:, 0], XYZ[:, 1], XYZ[:, 2]
    theta_pts, phi_pts = cart2sph_angles(x, y, z)
    mu_pts = np.cos(theta_pts)

    N = XYZ.shape[0]
    u = np.zeros(N, dtype=complex)

    # e^{i m φ} for all m, all points
    Em = np.exp(1j * np.outer(phi_pts, mvals))   # (N, 2L+1)

    for ell in range(Lmax + 1):
        Y_lm_all = np.zeros((N, n_m), dtype=complex)

        for m in range(0, ell + 1):
            Nlm = np.sqrt((2.0 * ell + 1.0) / (4.0 * np.pi) *
                          factratio(ell - m, ell + m))

            col_pos = m + Lmax
            col_neg = -m + Lmax

            plm_vec = lpmv(m, ell, mu_pts)   # (N,)
            Ypos = Nlm * plm_vec * Em[:, col_pos]

            Y_lm_all[:, col_pos] = Ypos
            if m > 0:
                Y_lm_all[:, col_neg] = (-1)**m * np.conj(Ypos)

        u += Y_lm_all @ ulm[ell, :]

    # For real initial data, the solution is real
    return u.real


# ======================= helpers =======================

def gausslegendre(n):
    """
    Gauss–Legendre nodes μ∈[-1,1] and weights (n-point).
    Uses numpy.polynomial.legendre.leggauss.
    """
    mu, w = leggauss(n)
    return mu, w


def factratio(a, b):
    """
    (a)! / (b)! for integers 0 <= a <= b.
    """
    if a == b:
        return 1.0
    # product from a+1 to b in denominator
    return 1.0 / np.prod(np.arange(a + 1, b + 1, dtype=float))


def sph2cartR(theta, phi, R):
    """
    Convert spherical angles (θ,φ) to Cartesian (x,y,z) on radius R.
    theta, phi can be arrays; returns shape (N,3).
    """
    x = R * np.sin(theta) * np.cos(phi)
    y = R * np.sin(theta) * np.sin(phi)
    z = R * np.cos(theta)
    return np.column_stack((x.ravel(), y.ravel(), z.ravel()))


def cart2sph_angles(x, y, z):
    """
    Return spherical angles θ∈[0,π], φ∈[0,2π) from Cartesian coordinates.
    """
    phi = np.arctan2(y, x)
    phi = np.where(phi < 0.0, phi + 2.0 * np.pi, phi)
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(np.clip(z / r, -1.0, 1.0))
    return theta, phi


def enforce_radius(XYZ, R):
    """
    Project points to the sphere of radius R (safety).
    """
    r = np.linalg.norm(XYZ, axis=1)
    scale = R / np.maximum(r, np.finfo(float).eps)
    return XYZ * scale[:, None]


# ======================= example usage =======================

if __name__ == "__main__":
    R    = 1.0
    c    = 1.4
    Lmax = 32
    t    = 0.6939

    # Initial conditions
    f_handle = lambda x, y, z: x**2 - y**2
    g_handle = lambda x, y, z: 0.0 * x

    # Evaluation grid (θ, φ)
    nTheta = 400
    nPhi   = 200
    theta  = np.linspace(0.0, np.pi, nTheta)
    phi    = np.linspace(0.0, 2.0 * np.pi, nPhi, endpoint=False)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    XYZ = sph2cartR(TH, PH, R)

    # Compute solution
    u = wave1_sphere_exact(XYZ, t, f_handle, g_handle, Lmax, c, R)
    U = u.reshape(TH.shape)

    #Saving data
    np.savez_compressed(f'wave_sphere_exact_{t:.3f}.npz', theta=theta, phi=phi, U=U)
    print(f"Saved data to wave_sphere_exact_{t:.3f}.npz")



    # Simple visualization (matplotlib)
    import matplotlib.pyplot as plt
    plt.figure()
    im = plt.imshow(U.real, extent=[phi[0], phi[-1], theta[0], theta[-1]],
                    origin='lower', aspect='auto', cmap='turbo')
    plt.xlabel(r'$\phi$')
    plt.ylabel(r'$\theta$')
    plt.colorbar(im, label=f'Displacement $u(\\theta, \\phi, t={t:.3f})$')
    plt.title(r'Exact Solution $u(\theta,\phi,t={:.3f})$'.format(t))
    plt.savefig(f'wave_sphere_exact_{t:.3f}.png', dpi=300, bbox_inches='tight')
    #plt.show()
