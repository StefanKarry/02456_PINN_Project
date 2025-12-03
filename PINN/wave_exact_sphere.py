import numpy as np
from scipy.special import lpmv, gammaln
from numpy.polynomial.legendre import leggauss

##### The exact spectral solution of the wave equation on the sphere #####
##### Used Copilot to convert from MATLAB to Python                  #####

def wave_sphere_exact(XYZ, t, f_handle, g_handle, Lmax, c, R):
    """
    Exact spectral solution of u_tt = c^2 Δ_{S^2_R} u on the sphere.

    Parameters
    ----------
    XYZ : array_like, shape (N,3)
        Cartesian coordinates of query points on (or near) the sphere of radius R.
    t : float
        Time at which to evaluate the solution.
    f_handle : callable
        Function f(x,y,z) giving initial displacement u(x,y,z,0).
    g_handle : callable
        Function g(x,y,z) giving initial velocity u_t(x,y,z,0).
    Lmax : int
        Maximum spherical-harmonic degree (bandlimit / resolution).
    c : float
        Wave speed along the surface.
    R : float
        Sphere radius.

    Returns
    -------
    u : ndarray, shape (N,)
        Solution u(XYZ_i, t) evaluated at the given points.
    """

    XYZ = np.asarray(XYZ, dtype=float)
    Npts = XYZ.shape[0]

    # ------------------ set up quadrature grid ------------------
    Ntheta = Lmax + 1
    Nphi   = 2 * Lmax + 1

    # Gauss–Legendre nodes μ in [-1,1] and weights
    mu, w_mu = leggauss(Ntheta)          # mu: (Ntheta,), w_mu: (Ntheta,)
    theta_q = np.arccos(mu)              # θ ∈ [0,π]
    phi_q   = np.linspace(0.0, 2.0*np.pi, Nphi, endpoint=False)
    dphi    = 2.0 * np.pi / Nphi

    # Sample f and g on quadrature grid
    TH, PH = np.meshgrid(theta_q, phi_q, indexing="ij")  # TH,PH: (Ntheta,Nphi)
    XYZq = sph2cartR(TH.ravel(), PH.ravel(), R)          # (Ntheta*Nphi,3)
    xq, yq, zq = XYZq[:, 0], XYZq[:, 1], XYZq[:, 2]

    fq = f_handle(xq, yq, zq).reshape(Ntheta, Nphi)
    gq = g_handle(xq, yq, zq).reshape(Ntheta, Nphi)

    # ------------------ forward SHT: compute f_lm, g_lm ------------------
    # flm, glm arrays: index [ell, m_index], where m_index = m + Lmax, m ∈ [-Lmax..Lmax]
    flm = np.zeros((Lmax + 1, 2 * Lmax + 1), dtype=complex)
    glm = np.zeros_like(flm)

    mvals = np.arange(-Lmax, Lmax + 1)  # [-Lmax, ..., Lmax]
    # Precompute Fourier sums in φ
    # Ephi[j,k] = exp(-i * m_k * φ_j), size (Nphi, 2Lmax+1)
    Ephi = np.exp(-1j * np.outer(phi_q, mvals))  # (Nphi, 2Lmax+1)

    # Fhat[i,k] = ∑_j fq[i,j] e^{-i m_k φ_j} dφ
    Fhat = fq @ Ephi * dphi          # (Ntheta, 2Lmax+1)
    Ghat = gq @ Ephi * dphi

    # Loop over ℓ, compute associated Legendre, project
    for ell in range(Lmax + 1):
        # P[m, i] = P_ell^m(mu_i) for m=0..ell, i=0..Ntheta-1
        P = np.zeros((ell + 1, Ntheta), dtype=float)
        for m in range(ell + 1):
            P[m, :] = lpmv(m, ell, mu)   # Condon–Shortley phase included

        for m in range(ell + 1):
            # normalization N_lm
            Nlm = sph_norm_factor(ell, m)

            col_pos = m + Lmax          # m index
            col_neg = -m + Lmax         # -m index

            plm_vec = P[m, :]           # (Ntheta,)
            # Projection: f_lm = ∑_i w_mu(i) P_l^m(mu_i) * Fhat(i,m) * Nlm
            flm_pos = Nlm * np.dot(w_mu, plm_vec * Fhat[:, col_pos])
            glm_pos = Nlm * np.dot(w_mu, plm_vec * Ghat[:, col_pos])

            flm[ell, col_pos] = flm_pos
            glm[ell, col_pos] = glm_pos

            if m > 0:
                # Y_l^{-m} = (-1)^m conj(Y_l^{m}) => f_{l,-m} = (-1)^m conj(f_{l,m})
                sign = (-1)**m
                flm[ell, col_neg] = sign * np.conj(flm_pos)
                glm[ell, col_neg] = sign * np.conj(glm_pos)

    # ------------------ exact modal time evolution ------------------
    ulm = np.zeros_like(flm)
    for ell in range(Lmax + 1):
        omega = (c / R) * np.sqrt(ell * (ell + 1))
        if ell == 0 or abs(omega) < 1e-15:
            ulm[ell, :] = flm[ell, :] + glm[ell, :] * t
        else:
            ulm[ell, :] = (
                flm[ell, :] * np.cos(omega * t)
                + glm[ell, :] * np.sin(omega * t) / omega
            )

    # ------------------ synthesize at requested XYZ nodes ------------------
    XYZp = enforce_radius(XYZ, R)
    x, y, z = XYZp[:, 0], XYZp[:, 1], XYZp[:, 2]
    theta_pts, phi_pts = cart2sphAngles(x, y, z)
    mu_pts = np.cos(theta_pts)

    # Precompute e^{i m φ} at all points
    Em = np.exp(1j * np.outer(phi_pts, mvals))  # (Npts, 2Lmax+1)

    u = np.zeros(Npts, dtype=complex)

    for ell in range(Lmax + 1):
        # P_l^m(mu_pts) for m=0..ell
        P = np.zeros((ell + 1, Npts), dtype=float)
        for m in range(ell + 1):
            P[m, :] = lpmv(m, ell, mu_pts)

        Y_lm_all = np.zeros((Npts, 2 * Lmax + 1), dtype=complex)

        for m in range(ell + 1):
            Nlm = sph_norm_factor(ell, m)

            col_pos = m + Lmax
            col_neg = -m + Lmax

            Ypos = Nlm * P[m, :] * Em[:, col_pos]  # (Npts,)
            Y_lm_all[:, col_pos] = Ypos

            if m > 0:
                sign = (-1)**m
                Y_lm_all[:, col_neg] = sign * np.conj(Ypos)

        u += Y_lm_all @ ulm[ell, :]

    # If initial data were real-valued, return real part
    if np.isrealobj(fq) and np.isrealobj(gq):
        u = u.real

    return u


# ======================= helper functions =======================

def sph_norm_factor(ell, m):
    """
    Normalization N_lm for complex spherical harmonics
    Y_l^m(θ,φ) = N_lm P_l^m(cosθ) e^{i m φ},
    N_lm = sqrt( (2l+1)/(4π) * (l-m)!/(l+m)! ).
    Uses gammaln for numerical stability.
    """
    # log((ell-m)!/(ell+m)!) = gammaln(ell-m+1) - gammaln(ell+m+1)
    log_fact_ratio = gammaln(ell - m + 1) - gammaln(ell + m + 1)
    return np.sqrt((2 * ell + 1) / (4.0 * np.pi) * np.exp(log_fact_ratio))


def sph2cartR(theta, phi, R):
    """
    Convert spherical angles (θ,φ) to (x,y,z) on radius R.
    theta: array-like, polar angle in [0,π]
    phi  : array-like, azimuth in [0,2π)
    """
    theta = np.asarray(theta)
    phi   = np.asarray(phi)
    x = R * np.sin(theta) * np.cos(phi)
    y = R * np.sin(theta) * np.sin(phi)
    z = R * np.cos(theta)
    return np.column_stack((x.ravel(), y.ravel(), z.ravel()))


def cart2sphAngles(x, y, z):
    """
    Convert Cartesian coordinates to spherical angles (θ,φ),
    where θ ∈ [0,π] (polar) and φ ∈ [0,2π) (azimuth).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    phi = np.arctan2(y, x)
    phi = np.where(phi < 0.0, phi + 2.0 * np.pi, phi)
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(np.clip(z / r, -1.0, 1.0))
    return theta, phi


def enforce_radius(XYZ, R):
    """
    Project points to the sphere of radius R (safe-guard).
    """
    XYZ = np.asarray(XYZ, dtype=float)
    r = np.linalg.norm(XYZ, axis=1)
    scale = R / np.maximum(r, np.finfo(float).eps)
    return XYZ * scale[:, None]

import numpy as np
import matplotlib.pyplot as plt

# Parameters
R = 1.0
c = 1.0
Lmax = 32
t = 0.7

# Initial conditions
f_handle = lambda x, y, z: x**2 - y**2       # initial displacement
g_handle = lambda x, y, z: 0.0 * x          # zero initial velocity

# Evaluation grid for visualization
nTheta = 200
nPhi   = 200
theta = np.linspace(0.0, np.pi, nTheta)
phi   = np.linspace(0.0, 2.0*np.pi, nPhi)
TH, PH = np.meshgrid(theta, phi, indexing="ij")
XYZ = sph2cartR(TH.ravel(), PH.ravel(), R)

# Solve
u = wave_sphere_exact(XYZ, t, f_handle, g_handle, Lmax, c, R)
U = u.reshape(TH.shape)

# # Plot
plt.figure(figsize=(6, 4))
im = plt.imshow(
    U.real,
    extent=(phi[0], phi[-1], theta[0], theta[-1]),
    origin="lower",
    aspect="auto"
)
plt.xlabel(r"$\phi$")
plt.ylabel(r"$\theta$")
plt.title(r"$u(\theta,\phi,t=0.7)$")
plt.colorbar(im, label="u")
plt.show()


