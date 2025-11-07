function u = wave_sphere_exact(XYZ, t, f_handle, g_handle, Lmax, c, R)
%WAVE_SPHERE_EXACT  Exact spectral solution of u_tt = c^2 Δ_{S^2_R} u
%   u = wave_sphere_exact(XYZ, t, f_handle, g_handle, Lmax, c, R)
%
% Inputs
%   XYZ      : N x 3 array of points on the sphere of radius R (each row: [x y z])
%   t        : time at which to evaluate the solution
%   f_handle : @(x,y,z) initial displacement on the sphere
%   g_handle : @(x,y,z) initial velocity on the sphere
%   Lmax     : maximum spherical-harmonic degree (bandlimit / resolution)
%   c        : wave speed along the surface
%   R        : sphere radius
%
% Output
%   u        : N x 1 vector, solution u(XYZ_i, t)
%
% Notes
%   - Uses complex Condon–Shortley spherical harmonics with L2-normalization:
%       Y_l^m(θ,φ) = N_lm P_l^m(cosθ) e^{i m φ},
%       N_lm = sqrt( (2l+1)/(4π) * (l-m)!/(l+m)! )
%   - Handles the ℓ=0 mode via the limit ω_0 → 0: u_00(t) = f_00 + g_00 t
%   - Quadrature: Gauss–Legendre in μ=cosθ (Nθ=Lmax+1), uniform in φ (Nφ=2Lmax+1)
%   - Requires only base MATLAB (uses built-in legendre)
%
% Example
%   R=1; c=1; Lmax=32; t=0.7;
%   f = @(x,y,z) x.^2 - y.^2;            % some smooth function on the sphere
%   g = @(x,y,z) 0*x;                    % zero initial velocity
%   % query a Fibonacci grid or any set of points:
%   th = linspace(0,pi,200)'; ph = linspace(0,2*pi,200);
%   [TH,PH] = ndgrid(th,ph); XYZ = sph2cartR(TH(:),PH(:),R);
%   u = wave_sphere_exact(XYZ,t,f,g,Lmax,c,R);
%   % reshape and visualize
%   U = reshape(u, size(TH)); imagesc(ph,th,real(U)); set(gca,'YDir','normal'); colorbar
%

% ------------------ set up quadrature grid ------------------
Ntheta = Lmax + 1;         % exact for bandlimit Lmax in μ
Nphi   = 2*Lmax + 1;       % exact for bandlimit Lmax in φ
[mu,w_mu] = gausslegendre(Ntheta);              % μ = cosθ in [-1,1]
theta_q = acos(mu);                             
phi_q   = linspace(0, 2*pi, Nphi+1); phi_q(end) = [];  % uniform [0,2π)

% Sample f and g on the quadrature grid (in xyz)
[TH,PH] = ndgrid(theta_q, phi_q);
XYZq    = sph2cartR(TH(:), PH(:), R);
fq = reshape(f_handle(XYZq(:,1), XYZq(:,2), XYZq(:,3)), Ntheta, Nphi);
gq = reshape(g_handle(XYZq(:,1), XYZq(:,2), XYZq(:,3)), Ntheta, Nphi);

% ------------------ forward SHT: compute f_lm, g_lm ------------------
% Integral convention: ∫_{S^2} f(θ,φ) Y_lm*(θ,φ) dΩ
% Using μ = cosθ, dΩ = dφ dμ, so weights are w_mu (Gauss-Legendre) and Δφ.
dphi = 2*pi / Nphi;
flm = zeros(Lmax+1, 2*Lmax+1);   % store for m=-L..L in columns (offset by +Lmax+1)
glm = zeros(Lmax+1, 2*Lmax+1);

% Precompute exp(-i m φ) sums (Fourier in φ) for each m needed
mvals = -Lmax:Lmax;
Ephi  = exp(-1i*(phi_q(:))*mvals);  % size Nphi x (2Lmax+1)
% For each θ_i, compute F_m(θ_i) = ∑_j f(θ_i,φ_j) e^{-i m φ_j} dφ
Fhat = (fq * Ephi) * dphi;   % size Ntheta x (2Lmax+1)
Ghat = (gq * Ephi) * dphi;

% Loop in ℓ to apply associated Legendre & normalization
for ell = 0:Lmax
    % Associated Legendre P_ell^m(μ) for m=0..ell at all μ_i
    % legendre returns (ell+1) x Ntheta, orders m=0..ell
    % We rely on the default (unnormalized) associated Legendre with Condon–Shortley phase.
    P = legendre(ell, mu);               % size (ell+1) x Ntheta
    if isvector(P), P = P(:).'; end      % ensure 2D
    % Normalize to N_lm and distribute to m >= 0 and m < 0 using symmetry
    for m = 0:ell
        Nlm = sqrt((2*ell+1)/(4*pi) * factratio(ell-m, ell+m));
        % column index for m in [-L..L] mapping
        col_pos = m + Lmax + 1;       % m
        col_neg = -m + Lmax + 1;      % -m
        % Projection: f_lm = ∫ f Y_lm* dΩ = ∑_i w_mu(i) P_l^m(μ_i) * Fhat(i,m) * (normalization)
        % Y_lm(θ,φ) = Nlm P_l^m(μ) e^{i m φ}  => Y_lm* = Nlm P_l^m(μ) e^{-i m φ}
        plm_vec = P(m+1,:).';                       % Ntheta x 1
        flm_pos = Nlm * (w_mu(:).' * (plm_vec .* Fhat(:, col_pos)));  % scalar
        glm_pos = Nlm * (w_mu(:).' * (plm_vec .* Ghat(:, col_pos)));
        flm(ell+1, col_pos) = flm_pos;
        glm(ell+1, col_pos) = glm_pos;
        if m > 0
            % Y_l^{-m} = (-1)^m conj(Y_l^{m})
            % Hence f_{l,-m} = (-1)^m conj(f_{l,m})
            flm(ell+1, col_neg) = (-1)^m * conj(flm_pos);
            glm(ell+1, col_neg) = (-1)^m * conj(glm_pos);
        end
    end
end

% ------------------ exact modal time evolution ------------------
ulm = zeros(size(flm));
for ell = 0:Lmax
    omega = (c / R) * sqrt(ell*(ell+1));
    if ell == 0 | abs(omega) < 1e-15
        ulm(ell+1, :) = flm(ell+1, :) + glm(ell+1, :) * t;
    else
        ulm(ell+1, :) = flm(ell+1, :) * cos(omega*t) + (glm(ell+1, :) / omega) * sin(omega*t);
    end
end

% ------------------ synthesize u at requested XYZ nodes ------------------
% Convert XYZ to spherical (θ,φ)
XYZ = enforce_radius(XYZ, R);   % project to radius R for safety
[x,y,z] = deal(XYZ(:,1), XYZ(:,2), XYZ(:,3));
[theta, phi] = cart2sphAngles(x,y,z);

% Evaluate Y_lm(θ,φ) and sum
N = size(XYZ,1);
u = complex(zeros(N,1));

% Precompute e^{i m φ} for all m, all points
Em = exp(1i * phi * mvals);   % size N x (2Lmax+1)

% For each ℓ, build P_l^m(cosθ) for m>=0 and accumulate
mu_pts = cos(theta);
for ell = 0:Lmax
    P = legendre(ell, mu_pts.');         % (ell+1) x N  (μ column)
    if isvector(P), P = P(:).'; end
    % Build Y_lm rows for m>=0 using normalization
    Y_lm_all = zeros(N, 2*Lmax+1);       % temp, columns m=-L..L
    for m = 0:ell
        Nlm = sqrt((2*ell+1)/(4*pi) * factratio(ell-m, ell+m));
        Ypos = Nlm * (P(m+1,:).').* Em(:, m + Lmax + 1); % N x 1
        Y_lm_all(:, m + Lmax + 1) = Ypos;
        if m > 0
            % Y_{l,-m} = (-1)^m conj(Y_{l,m})
            Y_lm_all(:, -m + Lmax + 1) = (-1)^m * conj(Ypos);
        end
    end
    % Accumulate u += sum_m u_lm(t) * Y_lm(θ,φ)
    u = u + Y_lm_all * (ulm(ell+1,:).');  % N x (2L+1) times (2L+1) x 1
end

% Return real part if inputs were real-valued (typical case)
if isreal(fq) && isreal(gq)
    u = real(u);
end
end

% ======================= helpers =======================

function [mu,w] = gausslegendre(n)
%GAUSSLEGENDRE  Gauss–Legendre nodes μ∈[-1,1] and weights (n-point)
% Golub–Welsch with symmetric tridiagonal Jacobi matrix
beta = 0.5 ./ sqrt(1 - (2*(1:n-1)).^(-2));
T = diag(beta,1) + diag(beta,-1);     % symmetric tridiagonal
[V,D] = eig(T);
mu = diag(D); [mu,idx] = sort(mu);    % nodes
V = V(:,idx);
w = 2 * (V(1,:).^2).';                % weights
end

function r = factratio(a,b)
%FACTRATIO  (a)! / (b)! for integers 0<=a<=b using product
% Stable for moderate sizes used here
if a==b, r = 1; return; end
r = 1/prod((a+1):b);
end

function XYZ = sph2cartR(theta, phi, R)
%SPH2CARTR  Convert spherical angles (θ,φ) to (x,y,z) on radius R
x = R .* sin(theta) .* cos(phi);
y = R .* sin(theta) .* sin(phi);
z = R .* cos(theta);
XYZ = [x(:), y(:), z(:)];
end

function [theta,phi] = cart2sphAngles(x,y,z)
%CART2SPHANGLES  Return θ∈[0,π], φ∈[0,2π)
phi = atan2(y,x);
phi(phi<0) = phi(phi<0) + 2*pi;
r = sqrt(x.^2 + y.^2 + z.^2);
theta = acos(max(-1,min(1, z ./ r)));
end

function XYZ = enforce_radius(XYZ, R)
%ENFORCE_RADIUS  Project points to the sphere of radius R (safe-guard)
r = sqrt(sum(XYZ.^2,2));
scale = R ./ max(r, eps);
XYZ = XYZ .* scale;
end
