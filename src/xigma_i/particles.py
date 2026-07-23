"""Stage 0: macroparticle source and ballistic pusher.

Produces, for a bunch of macroparticles pushed ballistically through the
laser pulse, per-timestep samples (gamma, theta_x, theta_y, a0, weight)
consumed by Stage 1 deposition (see deposition.py) to build the 4D overlap
table H[gamma, theta_x, theta_y, a0] described in plan.md.

Naming note: plan.md calls the transverse momentum angles theta_y/theta_z
(treating x as the beam axis). core.py's existing convention -- reused here
for consistency with the rest of the file -- has z as the beam axis and
theta_x/theta_y as the transverse momentum angles p_{x,y}/gamma. These are
the same two physical angles under different labels; the physics below is
unaffected by the choice of name.

Unlike particle_kernel (which uses the angle *grid cell* to supply theta and
an efficiency-motivated importance-sampled/truncated x0,y0,z0 domain), this
module draws every quantity -- position, angle, energy -- directly from its
true (untruncated) distribution. That trades sampling efficiency for a
normalisation that requires no correction factors (no f_th weighting, no
z_weight/dsx/dsy trims), which is the right trade for a slow, correctness-
first reference path (see plan.md "Build order").
"""
import numpy as np
from dataclasses import dataclass

from .core import GAUSS_WIDTH, LORENTZ_WIDTH

V_REL = 2.0  # relative-velocity factor for near-backscattering geometry, see core.py calculate_intersection


@dataclass
class Bunch:
    """Macroparticles with real per-particle energy and momentum angles.

    x0, y0, z0 are k0_las-normalised positions (same convention as core.py's
    `particles` array). gamma, theta_x, theta_y are true per-particle values
    -- not grid-supplied. weight is the number of physical electrons
    represented by each macroparticle (uniform across the bunch).
    """
    x0: np.ndarray
    y0: np.ndarray
    z0: np.ndarray
    gamma: np.ndarray
    theta_x: np.ndarray
    theta_y: np.ndarray
    weight: float

    @property
    def n_particles(self):
        return self.x0.shape[0]


def sample_bunch(compton, n_particles, gamma0, sigma_gamma0, *,
                  chirp=0.0, angle_energy_corr=0.0, rng=None):
    """Draw a macroparticle bunch from the beam's (possibly correlated) phase space.

    compton: a configured Compton instance (must have set_electron_parameters
        and set_laser_parameters already called; only electron-side and
        k0_las/delta_z attributes are used here).
    gamma0, sigma_gamma0: mean and RMS of the electron energy distribution.
    chirp: dimensionless energy-position correlation. gamma acquires an
        additional shift `chirp * gamma0 * z0_phys / sigma_ez`, so chirp=0.1
        means a 10% fractional energy change over one bunch RMS length.
    angle_energy_corr: correlation coefficient in [-1, 1] between theta_x and
        the (gamma - gamma0) / sigma_gamma0 residual. Used to test
        divergence-energy correlated bunches (validation 3).
    rng: numpy Generator, or None to create a fresh one.
    """
    rng = np.random.default_rng() if rng is None else rng

    k0 = compton.k0_las
    x0 = rng.normal(0.0, k0 * compton.sigma_ex, n_particles)
    y0 = rng.normal(0.0, k0 * compton.sigma_ey, n_particles)
    z0 = rng.normal(k0 * compton.delta_z, k0 * compton.sigma_ez, n_particles)

    sigma_thx = compton.emit_x / compton.sigma_ex
    sigma_thy = compton.emit_y / compton.sigma_ey

    corr = np.clip(angle_energy_corr, -1.0, 1.0)
    g_std = rng.normal(0.0, 1.0, n_particles)
    thx_std = corr * g_std + np.sqrt(max(0.0, 1.0 - corr**2)) * rng.normal(0.0, 1.0, n_particles)

    z0_phys = z0 / k0
    gamma = gamma0 * (1.0 + chirp * z0_phys / compton.sigma_ez) + sigma_gamma0 * g_std
    theta_x = sigma_thx * thx_std
    theta_y = sigma_thy * rng.normal(0.0, 1.0, n_particles)

    weight = compton.N_e / n_particles

    return Bunch(x0=x0, y0=y0, z0=z0, gamma=gamma, theta_x=theta_x, theta_y=theta_y, weight=weight)


def _time_window(compton, z0, xp=np):
    """Per-particle time window [t0, t1] (k0_las*c*t units) bounding where the
    particle is within ~2 Rayleigh ranges transversely and ~1 Gauss-width
    temporally of the pulse. Same bound as calculate_intersection's p_t0/p_t1,
    ported to plain numpy and evaluated per-particle rather than per-batch.

    xp: array module z0 belongs to (np or cp) -- array-module-agnostic so the
    same function serves both the numpy and cupy push_and_sample backends.
    """
    beta_ff = compton.beta_ff
    zT = compton.k0_las * compton.sigma_lz
    zR = (compton.k0_las * compton.sigma_lr0)**2 * (1.0 + beta_ff) * 2.0

    sigma_tau = GAUSS_WIDTH * zT
    sigma_raileigh = LORENTZ_WIDTH * zR

    t0 = (xp.maximum(-sigma_tau, (-z0 * (1 + beta_ff) - 2 * sigma_raileigh) / (1 - beta_ff)) - z0) / 2
    t1 = (xp.minimum(sigma_tau, (-z0 * (1 + beta_ff) + 2 * sigma_raileigh) / (1 - beta_ff)) - z0) / 2
    return t0, t1


def push_and_sample(compton, bunch, n_steps=200, backend='numpy'):
    """Ballistically push each macroparticle and emit one sample per particle.

    backend: 'numpy' (default) -- the original vectorised (n_particles,
        n_steps) broadcast, single-threaded. 'numba' -- CPU multithreading:
        a per-particle @numba.njit(parallel=True) loop (numpy.prange) that
        integrates each particle's trajectory without materialising the full
        (n_particles, n_steps) intermediate arrays, so it also uses far less
        memory at large n_particles*n_steps. Requires the numba package.
        'cupy' -- GPU offload: the same broadcast form as 'numpy', run with
        cupy arrays (array-module-agnostic, same pattern as deposition.py).
        Output arrays stay on-device (cupy), ready to feed straight into
        deposition.build_table without a host round-trip. Requires cupy and
        a CUDA device.

    Returns arrays (gamma, theta_x, theta_y, a0, weight) of length
    n_particles, ready for Stage 1 deposition. gamma/theta_x/theta_y are
    constant per particle (no pusher acceleration -- straight-line
    trajectories, matching particle_kernel).

    a0 here is NOT the instantaneous local field amplitude -- it is the
    trajectory-averaged effective intensity hat-a(zeta), Paper/xigma.tex
    eq. "ahattraj":

        ahat(zeta) = (TrXi/2) * integral[a^2(t)]^2 dt / integral a^2(t) dt

    with a^2(t;zeta) the instantaneous squared potential (what this function
    computes internally as a0_local(t)**2) and TrXi/2 = (1 + ellipticity**2)/2
    (eq. "Xi", generalised from linear polarisation to the ellipticity
    Compton.ellipticity set via set_laser_parameters). A single scalar per
    particle, not a distribution sampled along its own trajectory, because
    in this weakly-nonlinear regime (a0 <~ 1) the photon formation length
    spans the *whole* trajectory -- unlike the synchrotron regime, splitting
    the trajectory into short segments and radiating each independently is
    not valid here. See CLAUDE.md "Known bugs" / "Traps" for the full
    explanation; do not go back to per-timestep a0 deposition.

    weight[i] = sum over the particle's timesteps of
                v_rel * n_ph_shape(t, r) * dt * weight_macro * sigma_T *
                k0_las**2 * N_l
    i.e. the luminosity functional L(zeta) (Paper/xigma.tex eq. "lumfun"),
    the same physical content as particle_kernel's `f_cur` summed over time,
    scaled by the constants that calculate_intersection applies afterwards
    via `coef` -- minus the angular-grid normalisation
    (2*pi*sigma_thx*sigma_thy) and the position-truncation corrections
    (z_weight, dsx, dsy), neither of which apply here since positions/angles
    are drawn from their true distributions rather than an importance-sampled
    truncated domain.

    n_steps sets the trajectory-integration resolution for L and ahat (not
    the output array length, which is always n_particles).
    """
    if backend == 'numpy':
        return _push_and_sample_vectorized(compton, bunch, n_steps, np)
    if backend == 'cupy':
        import cupy as cp
        return _push_and_sample_vectorized(compton, bunch, n_steps, cp)
    if backend == 'numba':
        return _push_and_sample_numba(compton, bunch, n_steps)
    raise ValueError(f"backend must be 'numpy', 'numba', or 'cupy', got {backend!r}")


def _push_and_sample_vectorized(compton, bunch, n_steps, xp):
    """The (n_particles, n_steps) broadcast form of push_and_sample, shared
    by the 'numpy' and 'cupy' backends -- array-module-agnostic like
    deposition.py's deposit_nearest/deposit_cic, since every operation here
    is elementwise or a reduction along the n_steps axis (nothing that needs
    a hand-written kernel). For xp=cp, bunch's (host numpy) fields are
    transferred once at the top and results stay on-device.
    """
    from .core import sigma_T

    k0 = compton.k0_las
    beta_ff = compton.beta_ff
    w0 = k0 * compton.sigma_lr0
    zT = k0 * compton.sigma_lz
    z_rayleigh = 2 * w0 * w0 * (1.0 + beta_ff)

    x0, y0, z0, gamma, theta_x, theta_y = (
        xp.asarray(a) for a in
        (bunch.x0, bunch.y0, bunch.z0, bunch.gamma, bunch.theta_x, bunch.theta_y))

    vx, vy = theta_x, theta_y
    vz = xp.sqrt(xp.maximum(0.0, 1.0 - vx**2 - vy**2))
    dt0 = z0 / vz

    t0_local, t1_local = _time_window(compton, z0, xp)
    span = xp.maximum(0.0, t1_local - t0_local)
    dt = span / n_steps

    step = (xp.arange(n_steps) + 0.5) / n_steps  # midpoint rule, shape (n_steps,)
    t = t0_local[:, None] + step[None, :] * span[:, None]  # (n, n_steps)

    x = x0[:, None] + vx[:, None] * (t + dt0[:, None])
    y = y0[:, None] + vy[:, None] * (t + dt0[:, None])
    z = z0[:, None] + vz[:, None] * t

    sigma_l_sq = w0 * w0 * (1.0 + (z - beta_ff * t)**2 / z_rayleigh**2)
    env = xp.exp(-((z + t) / zT)**2 / 2) / xp.sqrt(2 * np.pi) / zT
    n_ph_shape = xp.exp(-(x**2 + y**2) / sigma_l_sq / 2) / (2 * np.pi) / sigma_l_sq * env

    peak_shape = 1.0 / (2 * np.pi * w0 * w0) / (np.sqrt(2 * np.pi) * zT)
    a0_local = compton.a0 * xp.sqrt(xp.clip(n_ph_shape / peak_shape, 0.0, None))

    contribution = V_REL * n_ph_shape * dt[:, None] * bunch.weight * sigma_T * k0**2 * compton.N_l

    L = contribution.sum(axis=1)  # eq. "lumfun", per-particle deposited weight

    a_sq = a0_local**2  # a^2(t;zeta), eq. "ahattraj"
    denom = a_sq.sum(axis=1)
    F_pol = (1.0 + compton.ellipticity**2) / 2.0  # TrXi/2, eq. "Xi"
    # xp.where instead of np.divide(..., where=) -- cupy's ufunc `where=`
    # kwarg support is version-dependent; xp.where is safe on both.
    ahat = xp.where(denom > 0, F_pol * (a_sq**2).sum(axis=1) / xp.maximum(denom, 1e-300), 0.0)

    return gamma, theta_x, theta_y, ahat, L


_numba_kernel_cache = None


def _get_numba_kernel():
    """Lazily compiles and caches the numba kernel so importing particles.py
    doesn't require numba to be installed unless backend='numba' is used.
    """
    global _numba_kernel_cache
    if _numba_kernel_cache is not None:
        return _numba_kernel_cache
    try:
        import numba
    except ImportError as e:
        raise ImportError("backend='numba' requires the numba package (pip install numba)") from e

    @numba.njit(parallel=True, fastmath=True, cache=True)
    def kernel(x0, y0, z0, vx, vy, vz, t0_local, t1_local, n_steps,
               beta_ff, w0, zT, z_rayleigh, particle_weight, v_rel, sigma_T_,
               k0_sq, N_l, a0_compton, F_pol):
        n = x0.shape[0]
        L = np.empty(n, dtype=np.float64)
        ahat = np.empty(n, dtype=np.float64)
        two_pi = 2.0 * np.pi
        sqrt_two_pi = np.sqrt(two_pi)
        peak_shape = 1.0 / (two_pi * w0 * w0) / (sqrt_two_pi * zT)

        for i in numba.prange(n):
            span = t1_local[i] - t0_local[i]
            if span < 0.0:
                span = 0.0
            dt = span / n_steps
            dt0 = z0[i] / vz[i]

            contribution_sum = 0.0
            a_sq_sum = 0.0
            a_sq_sq_sum = 0.0
            for j in range(n_steps):
                step = (j + 0.5) / n_steps
                t = t0_local[i] + step * span

                x = x0[i] + vx[i] * (t + dt0)
                y = y0[i] + vy[i] * (t + dt0)
                z = z0[i] + vz[i] * t

                zr_term = z - beta_ff * t
                sigma_l_sq = w0 * w0 * (1.0 + zr_term * zr_term / (z_rayleigh * z_rayleigh))
                env = np.exp(-((z + t) / zT) ** 2 / 2.0) / sqrt_two_pi / zT
                n_ph_shape = np.exp(-(x * x + y * y) / sigma_l_sq / 2.0) / two_pi / sigma_l_sq * env

                ratio = n_ph_shape / peak_shape
                if ratio < 0.0:
                    ratio = 0.0
                a0_local = a0_compton * np.sqrt(ratio)

                contribution_sum += v_rel * n_ph_shape * dt * particle_weight * sigma_T_ * k0_sq * N_l

                a_sq = a0_local * a0_local
                a_sq_sum += a_sq
                a_sq_sq_sum += a_sq * a_sq

            L[i] = contribution_sum
            ahat[i] = F_pol * a_sq_sq_sum / a_sq_sum if a_sq_sum > 0.0 else 0.0

        return L, ahat

    _numba_kernel_cache = kernel
    return kernel


def _push_and_sample_numba(compton, bunch, n_steps):
    """Per-particle @numba.njit(parallel=True) form of push_and_sample: same
    physics as _push_and_sample_vectorized, but integrated with an explicit
    inner loop over n_steps instead of a materialised (n_particles, n_steps)
    array, parallelised across particles (numba.prange) instead of relying
    on numpy's (single-threaded, for elementwise ops) vectorisation. Wins
    both wall-clock (multiple CPU cores) and peak memory (no O(n_particles *
    n_steps) temporaries) at large problem sizes.
    """
    from .core import sigma_T

    k0 = compton.k0_las
    beta_ff = compton.beta_ff
    w0 = k0 * compton.sigma_lr0
    zT = k0 * compton.sigma_lz
    z_rayleigh = 2 * w0 * w0 * (1.0 + beta_ff)

    vz = np.sqrt(np.maximum(0.0, 1.0 - bunch.theta_x**2 - bunch.theta_y**2))
    t0_local, t1_local = _time_window(compton, bunch.z0, np)

    kernel = _get_numba_kernel()
    L, ahat = kernel(
        np.ascontiguousarray(bunch.x0), np.ascontiguousarray(bunch.y0),
        np.ascontiguousarray(bunch.z0), np.ascontiguousarray(bunch.theta_x),
        np.ascontiguousarray(bunch.theta_y), vz, t0_local, t1_local, n_steps,
        beta_ff, w0, zT, z_rayleigh, bunch.weight, V_REL, sigma_T, k0**2,
        compton.N_l, compton.a0, (1.0 + compton.ellipticity**2) / 2.0,
    )

    return bunch.gamma, bunch.theta_x, bunch.theta_y, ahat, L
