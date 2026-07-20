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


def _time_window(compton, z0):
    """Per-particle time window [t0, t1] (k0_las*c*t units) bounding where the
    particle is within ~2 Rayleigh ranges transversely and ~1 Gauss-width
    temporally of the pulse. Same bound as calculate_intersection's p_t0/p_t1,
    ported to plain numpy and evaluated per-particle rather than per-batch.
    """
    beta_ff = compton.beta_ff
    zT = compton.k0_las * compton.sigma_lz
    zR = (compton.k0_las * compton.sigma_lr0)**2 * (1.0 + beta_ff) * 2.0

    sigma_tau = GAUSS_WIDTH * zT
    sigma_raileigh = LORENTZ_WIDTH * zR

    t0 = (np.maximum(-sigma_tau, (-z0 * (1 + beta_ff) - 2 * sigma_raileigh) / (1 - beta_ff)) - z0) / 2
    t1 = (np.minimum(sigma_tau, (-z0 * (1 + beta_ff) + 2 * sigma_raileigh) / (1 - beta_ff)) - z0) / 2
    return t0, t1


def push_and_sample(compton, bunch, n_steps=200):
    """Ballistically push each macroparticle and emit per-timestep samples.

    Returns flat arrays (gamma, theta_x, theta_y, a0, weight) of length
    n_particles * n_steps, ready for Stage 1 deposition. gamma/theta_x/theta_y
    are constant per particle (no pusher acceleration -- straight-line
    trajectories, matching particle_kernel); a0 and weight vary along the
    trajectory as the particle crosses the pulse.

    weight[i] = v_rel * n_ph_shape(t, r) * dt * weight_macro * sigma_T *
                k0_las**2 * N_l
    i.e. the same physical content as particle_kernel's `f_cur`, scaled by
    the constants that calculate_intersection applies afterwards via `coef`
    -- minus the angular-grid normalisation (2*pi*sigma_thx*sigma_thy) and
    the position-truncation corrections (z_weight, dsx, dsy), neither of
    which apply here since positions/angles are drawn from their true
    distributions rather than an importance-sampled truncated domain.
    """
    from .core import sigma_T

    k0 = compton.k0_las
    beta_ff = compton.beta_ff
    w0 = k0 * compton.sigma_lr0
    zT = k0 * compton.sigma_lz
    z_rayleigh = 2 * w0 * w0 * (1.0 + beta_ff)

    n = bunch.n_particles
    vx = bunch.theta_x
    vy = bunch.theta_y
    vz = np.sqrt(np.maximum(0.0, 1.0 - vx**2 - vy**2))
    dt0 = bunch.z0 / vz

    t0_local, t1_local = _time_window(compton, bunch.z0)
    span = np.maximum(0.0, t1_local - t0_local)
    dt = span / n_steps

    step = (np.arange(n_steps) + 0.5) / n_steps  # midpoint rule, shape (n_steps,)
    t = t0_local[:, None] + step[None, :] * span[:, None]  # (n, n_steps)

    x = bunch.x0[:, None] + vx[:, None] * (t + dt0[:, None])
    y = bunch.y0[:, None] + vy[:, None] * (t + dt0[:, None])
    z = bunch.z0[:, None] + vz[:, None] * t

    sigma_l_sq = w0 * w0 * (1.0 + (z - beta_ff * t)**2 / z_rayleigh**2)
    env = np.exp(-((z + t) / zT)**2 / 2) / np.sqrt(2 * np.pi) / zT
    n_ph_shape = np.exp(-(x**2 + y**2) / sigma_l_sq / 2) / (2 * np.pi) / sigma_l_sq * env

    peak_shape = 1.0 / (2 * np.pi * w0 * w0) / (np.sqrt(2 * np.pi) * zT)
    a0_local = compton.a0 * np.sqrt(np.clip(n_ph_shape / peak_shape, 0.0, None))

    contribution = V_REL * n_ph_shape * dt[:, None] * bunch.weight * sigma_T * k0**2 * compton.N_l

    gamma_flat = np.broadcast_to(bunch.gamma[:, None], (n, n_steps)).reshape(-1)
    theta_x_flat = np.broadcast_to(bunch.theta_x[:, None], (n, n_steps)).reshape(-1)
    theta_y_flat = np.broadcast_to(bunch.theta_y[:, None], (n, n_steps)).reshape(-1)
    a0_flat = a0_local.reshape(-1)
    weight_flat = contribution.reshape(-1)

    return gamma_flat, theta_x_flat, theta_y_flat, a0_flat, weight_flat
