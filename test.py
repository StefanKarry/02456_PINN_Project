import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm
from scipy.special import lpmv, gammaln
from numpy.polynomial.legendre import leggauss


#######################################################################
#  EXACT SPECTRAL SOLUTION OF u_tt = c^2 Δ_{S^2_R} u ON THE SPHERE
#######################################################################

def wave_sphere_exact(XYZ, t, f_handle, g_handle, Lmax, c, R):
    """
    Exact spectral solution of u_tt = c^2 Δ_{S^2_R} u on the sphere.
    """
    XYZ = np.asarray(XYZ, dtype=float)
    Npts = XYZ.shape[0]

    # ------------------ quadrature grid ------------------
    Ntheta = Lmax + 1
    Nphi = 2 * Lmax + 1

    mu, w_mu = leggauss(Ntheta)              # Gauss–Legendre nodes, weights
    theta_q = np.arccos(mu)                  # θ ∈ [0, π]
    phi_q = np.linspace(0, 2*np.pi, Nphi, endpoint=False)
    dphi = 2 * np.pi / Nphi

    # Sample f, g on grid
    TH, PH = np.meshgrid(theta_q, phi_q, indexing="ij")  # (Ntheta, Nphi)
    XYZq = sph2cartR(TH.ravel(), PH.ravel(), R)          # (Ntheta*Nphi, 3)
    xq, yq, zq = XYZq[:, 0], XYZq[:, 1], XYZq[:, 2]

    fq = f_handle(xq, yq, zq).reshape(Ntheta, Nphi)
    gq = g_handle(xq, yq, zq).reshape(Ntheta, Nphi)

    # ------------------ forward SHT ------------------
    flm = np.zeros((Lmax + 1, 2 * Lmax + 1), dtype=complex)
    glm = np.zeros_like(flm)

    mvals = np.arange(-Lmax, Lmax + 1)
    Ephi = np.exp(-1j * np.outer(phi_q, mvals))  # (Nphi, 2Lmax+1)

    # Fourier transform in φ
    Fhat = fq @ Ephi * dphi    # (Ntheta, 2Lmax+1)
    Ghat = gq @ Ephi * dphi

    for ell in range(Lmax + 1):
        # Associated Legendre P_ell^m(mu_i) for m = 0..ell
        P = np.zeros((ell + 1, Ntheta))
        for m in range(ell + 1):
            P[m, :] = lpmv(m, ell, mu)

        for m in range(ell + 1):
            Nlm = sph_norm_factor(ell, m)
            col_pos = m + Lmax
            col_neg = -m + Lmax

            plm_vec = P[m, :]

            flm_pos = Nlm * np.dot(w_mu, plm_vec * Fhat[:, col_pos])
            glm_pos = Nlm * np.dot(w_mu, plm_vec * Ghat[:, col_pos])

            flm[ell, col_pos] = flm_pos
            glm[ell, col_pos] = glm_pos

            if m > 0:
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
            ulm[ell, :] = flm[ell, :] * np.cos(omega * t) + glm[ell, :] * np.sin(omega * t) / omega

    # ------------------ synthesize at XYZ ------------------
    XYZp = enforce_radius(XYZ, R)
    x, y, z = XYZp[:, 0], XYZp[:, 1], XYZp[:, 2]
    theta_pts, phi_pts = cart2sphAngles(x, y, z)
    mu_pts = np.cos(theta_pts)

    Em = np.exp(1j * np.outer(phi_pts, mvals))  # (Npts, 2Lmax+1)

    u = np.zeros(Npts, dtype=complex)

    for ell in range(Lmax + 1):
        P = np.zeros((ell + 1, Npts))
        for m in range(ell + 1):
            P[m, :] = lpmv(m, ell, mu_pts)

        Y_lm_all = np.zeros((Npts, 2 * Lmax + 1), dtype=complex)

        for m in range(ell + 1):
            Nlm = sph_norm_factor(ell, m)
            col_pos = m + Lmax
            col_neg = -m + Lmax

            Ypos = Nlm * P[m, :] * Em[:, col_pos]
            Y_lm_all[:, col_pos] = Ypos

            if m > 0:
                sign = (-1)**m
                Y_lm_all[:, col_neg] = sign * np.conj(Ypos)

        u += Y_lm_all @ ulm[ell, :]

    return u.real


#######################################################################
#  Helper functions
#######################################################################

def sph_norm_factor(ell, m):
    """
    Normalization N_lm for spherical harmonics.
    """
    log_ratio = gammaln(ell - m + 1) - gammaln(ell + m + 1)
    return np.sqrt((2 * ell + 1) / (4 * np.pi) * np.exp(log_ratio))

def sph2cartR(theta, phi, R):
    """
    Spherical angles (θ, φ) -> Cartesian (x, y, z) on sphere of radius R.
    theta, phi can be arrays of same shape.
    """
    theta = np.asarray(theta)
    phi = np.asarray(phi)
    x = R * np.sin(theta) * np.cos(phi)
    y = R * np.sin(theta) * np.sin(phi)
    z = R * np.cos(theta)
    return np.column_stack((x.ravel(), y.ravel(), z.ravel()))

def cart2sphAngles(x, y, z):
    """
    Cartesian (x,y,z) -> spherical angles (θ, φ).
    θ ∈ [0, π], φ ∈ [0, 2π).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    phi = np.arctan2(y, x)
    phi = np.where(phi < 0, phi + 2*np.pi, phi)
    r = np.sqrt(x**2 + y**2 + z**2)
    r = np.maximum(r, np.finfo(float).eps)
    theta = np.arccos(np.clip(z / r, -1.0, 1.0))
    return theta, phi

def enforce_radius(XYZ, R):
    """
    Project points onto the sphere of radius R.
    """
    XYZ = np.asarray(XYZ, dtype=float)
    r = np.linalg.norm(XYZ, axis=1)
    scale = R / np.maximum(r, np.finfo(float).eps)
    return XYZ * scale[:, None]


#######################################################################
#  INITIAL CONDITIONS
#######################################################################

R = 1.0
c = 1.4
Lmax = 20   # increase for smoother but more expensive solution

def f_handle(x, y, z):
    """
    Initial bump around the north pole (z ~ R).
    """
    cos_theta = z / R
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    sigma = 0.3
    return np.exp(- (theta**2) / (2 * sigma**2))

def g_handle(x, y, z):
    """
    Zero initial velocity.
    """
    return np.zeros_like(x)


#######################################################################
#  SPHERICAL GRID FOR VISUALISATION
#######################################################################

Ntheta_plot = 100     # increase for finer grid
Nphi_plot   = 160

theta = np.linspace(0.0, np.pi, Ntheta_plot)
phi   = np.linspace(0.0, 2.0*np.pi, Nphi_plot, endpoint=False)
TH, PH = np.meshgrid(theta, phi, indexing="ij")   # (Ntheta_plot, Nphi_plot)

XYZ_plot = sph2cartR(TH, PH, R)
X = XYZ_plot[:, 0].reshape(Ntheta_plot, Nphi_plot)
Y = XYZ_plot[:, 1].reshape(Ntheta_plot, Nphi_plot)
Z = XYZ_plot[:, 2].reshape(Ntheta_plot, Nphi_plot)


#######################################################################
#  TIME GRID
#######################################################################

T_final = 6.0
Nt = 120
t_vals = np.linspace(0.0, T_final, Nt)


#######################################################################
#  ANIMATION: FILLED SURFACE (NO HOLES)
#######################################################################

# Evaluate initial frame
u0 = wave_sphere_exact(XYZ_plot, t_vals[0], f_handle, g_handle, Lmax, c, R)
u0 = u0.reshape(Ntheta_plot, Nphi_plot)

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')

# Fix color scale from initial frame (or compute from all times if you want)
vmin, vmax = u0.min(), u0.max()
norm = plt.Normalize(vmin=vmin, vmax=vmax)

# Colors for faces: (Ntheta_plot-1) x (Nphi_plot-1) faces
colors0 = cm.viridis(norm(u0))
colors_faces0 = colors0[:-1, :-1, :]   # strip last row/col to match faces
surf = ax.plot_surface(
    X, Y, Z,
    facecolors=colors_faces0,
    rstride=1, cstride=1,
    shade=False,
    antialiased=False
)

ax.set_box_aspect([1, 1, 1])
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_title(f"Wave on sphere, t = {t_vals[0]:.2f}")

def update(frame):
    t = t_vals[frame]
    u = wave_sphere_exact(XYZ_plot, t, f_handle, g_handle, Lmax, c, R)
    u = u.reshape(Ntheta_plot, Nphi_plot)

    colors = cm.viridis(norm(u))
    colors_faces = colors[:-1, :-1, :].reshape(-1, 4)  # match #faces

    # Update the existing surface's facecolors
    surf.set_facecolors(colors_faces)

    ax.set_title(f"Wave on sphere, t = {t:.2f}")
    return surf,

ani = FuncAnimation(fig, update, frames=Nt, interval=50, blit=False)

plt.show()
