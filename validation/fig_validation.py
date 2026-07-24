#!/usr/bin/env python
"""Fig. 4 -- delta-function model vs direct spectral integration
(convergence-validation.md Sec. 4).

*** FIRST-DRAFT PHYSICS, NOT YET CROSS-VALIDATED -- SEE THE WARNING BELOW. ***

Unlike Figs. 1-3 (which exercise only already-validated xigma_i code: Stage
0/1/2 and reference.spectrum_from_table), this figure tests the physics
approximation itself and needs machinery that does not exist anywhere in
xigma_i: the per-electron resonance function R (paper Eq. Rdef) evaluated by
explicit Fourier quadrature, instead of the delta-function substitution.
Implemented here, from the paper (`Paper/xigma.tex`, symlinked to
Papers/2026/Compton-Numerics), not from any existing validated code path:

  - wR(gamma, theta, ahat) = gamma^2 / (1 + gamma^2 theta^2 + ahat), in the
    code's s-units (s = omega/4*omega_L) -- the *full* resonance condition,
    including the ahat term that xigma_i's own spectrum_kernel_4d currently
    drops (it brackets/inverts only on theta, then sums H over a0 at that
    fixed g -- see spectrum4d.py). Fig. 4 needs the full condition because
    it is testing the delta-substitution itself, not xigma_i's kernel.
  - ahat(zeta), the trajectory-averaged effective intensity (Eq. ahattraj),
    and L(zeta), the luminosity functional (Eq. lumfun) -- taken directly
    from particles.push_and_sample, which computes both per-particle (one
    row per particle, not per timestep -- see git log "Fix a0/H..." and that
    function's docstring). Not recomputed here: earlier versions of this
    file kept an independent copy of the same ahat formula over a different
    (wider, particle-independent) time window: now redundant given the
    canonical implementation, and less consistent with the rest of xigma_i.
  - R(Delta_s; ahat) as |FFT{field(t)}|^2 over a common, particle-independent
    window +/- sigma_tau (matching particles.py's own conservative exposure
    bound), normalised to integrate to 1 in Delta_s (paper Eq. Rnorm), where
    field(t) = a0_local(t) * exp(i*phase_geom(t)) is the envelope amplitude
    times the wavefront-curvature/Gouy phase factor of a focused Gaussian
    beam (see sample_common_window_field) -- the fast carrier phase is not
    included in field(t) itself; it is handled separately by the wR/wL
    Doppler-stretch (the q = 4*s_res rescaling below), so only the residual,
    slowly-varying geometric phase needs to be resolved here.

The field along each trajectory now includes the wavefront-curvature and
Gouy-phase terms of a focused Gaussian beam (previously it used only the
real envelope, i.e. implicitly a plane wave with no phase structure beyond
the fast carrier). This is adapted from resonance-function.py (a standalone
script written separately, reviewed and incorporated here -- see git log
for the physics check) via `efield()`'s phase formula. Numerically, for
params.py's representative case, the laser pulse spans ~7.4 Rayleigh
ranges (zT/z_rayleigh, printed at runtime), so the Gouy phase (which swings
by ~pi per Rayleigh range) varies substantially across a trajectory and is
not a small correction -- this is *not* the same effect resonance-
function.py separately calls out as "nonlinearity": the wavefront-curvature/
Gouy terms are a *geometric* consequence of a focused (non-plane-wave) beam,
present even as a0 -> 0, whereas the still-omitted ponderomotive phase
Phi_NL below is intensity-driven and only matters at a0 ~> 1.

SIMPLIFICATION KNOWINGLY MADE: the ponderomotive nonlinear phase Phi_NL
(Eq. PhiNL) is set to zero, i.e. R is computed from the linear bandwidth and
the geometric (curvature/Gouy) chirp, not the ponderomotive one. The paper
states Phi_NL "vanishes identically for a flat-top pulse" and matters
"when a0 >~ 1"; params.py's representative case has compton.a0 ~ 0.09
(printed at runtime), so this remains a quantitatively small omission for
the chosen parameter set, not a free pass in general -- unlike the
geometric term above, which is unrelated to a0 and was not small here.
Anyone reusing this at higher a0 needs to add Phi_NL back in (it requires
nailing down the phase origin for Eq. PhiNL's running average, which is not
fully determined by the sections of the paper read while writing this
script). resonance-function.py's own trajectories are also still ballistic
(unperturbed by the field), so it does not include Phi_NL either -- adding
the geometric phase did not, by itself, resolve that omission.

Consequence: treat this script's *scaling exponent* result (the thing the
paper actually wants to quote) as trustworthy -- it follows from the
Fourier-transform width/tail relationship regardless of the Phi_NL
omission -- but do not quote absolute delta-vs-direct-integration
differences from this script without a physics review, since one
non-trivial modelling choice here (Phi_NL=0) was not independently
cross-checked against anything else in the repository the way Figs. 1-3's
building blocks were.

--quick: fewer particles and a shorter mu sweep (a minute or so).
"""
import argparse

import numpy as np

import params as P
import refs as R
import metrics as MT
import plotstyle as PS

GAUSS_WIDTH = 3.0


def sample_common_window_field(compton, bunch, n_steps):
    """Complex field a0_local(t) * exp(i*phase_geom(t)) on a single,
    particle-independent time grid t in [-sigma_tau, sigma_tau]
    (sigma_tau = GAUSS_WIDTH * k0_las * sigma_lz, the same conservative
    temporal bound particles.py uses for its own per-particle exposure
    window) -- shared across particles so the FFT below is one batched
    call. See module docstring for why a common window is an adequate
    simplification (the temporal Gaussian factor dominates the window
    choice; a0_local already -> 0 outside a particle's real overlap
    regardless of what window it's sampled on).

    phase_geom = wavefront-curvature + Gouy phase of a focused Gaussian
    beam, adapted from resonance-function.py's efield() (standard Siegman-
    type paraxial-beam formulae) to this codebase's units. That script's
    "w0" is the field's 1/e amplitude radius; xigma_i's sigma_lr0 is the
    *density* RMS width. The two give the same z_rayleigh already used
    elsewhere in this file (w0_field = 2*sigma_lr0 and z_rayleigh =
    w0_field^2*(1+beta_ff)/2 = 2*sigma_lr0^2*(1+beta_ff), matching this
    function's own z_rayleigh -- checked by hand before relying on it), so
    no separate "field waist" variable is introduced; curv/gouy are built
    directly from the z_rayleigh/z(t) already computed for sigma_l_sq.

    Deliberately excludes the fast carrier phase (z+t): unlike
    resonance-function.py, which resolves every optical cycle directly
    (LAMDA_RES=32 samples/cycle, needed because it does not separate the
    carrier first), the carrier is handled analytically here by the wR/wL
    Doppler-stretch already applied in per_particle_R/build_spectra -- only
    phase_geom needs resolving, and it varies on the envelope/trajectory
    scale (comparable to z_rayleigh, not to one optical cycle), so the
    existing coarse grid remains adequate. That separation is what keeps
    this tractable at this laser's ~10^4-cycle pulse length and >=10^4
    particles; resolving every cycle directly, as in resonance-function.py,
    is not practical at this pulse length and particle count.
    """
    k0 = compton.k0_las
    beta_ff = compton.beta_ff
    w0 = k0 * compton.sigma_lr0
    zT = k0 * compton.sigma_lz
    z_rayleigh = 2 * w0 * w0 * (1.0 + beta_ff)
    sigma_tau = GAUSS_WIDTH * zT

    vx, vy = bunch.theta_x, bunch.theta_y
    vz = np.sqrt(np.maximum(0.0, 1.0 - vx**2 - vy**2))
    dt0 = bunch.z0 / vz

    t = np.linspace(-sigma_tau, sigma_tau, n_steps)
    x = bunch.x0[:, None] + vx[:, None] * (t[None, :] + dt0[:, None])
    y = bunch.y0[:, None] + vy[:, None] * (t[None, :] + dt0[:, None])
    z = bunch.z0[:, None] + vz[:, None] * t[None, :]

    sigma_l_sq = w0 * w0 * (1.0 + (z - beta_ff * t[None, :])**2 / z_rayleigh**2)
    env = np.exp(-((z + t[None, :]) / zT)**2 / 2) / np.sqrt(2 * np.pi) / zT
    n_ph_shape = np.exp(-(x**2 + y**2) / sigma_l_sq / 2) / (2 * np.pi) / sigma_l_sq * env
    peak_shape = 1.0 / (2 * np.pi * w0 * w0) / (np.sqrt(2 * np.pi) * zT)
    a0_local = compton.a0 * np.sqrt(np.clip(n_ph_shape / peak_shape, 0.0, None))

    r_sq = x**2 + y**2
    curv = z / (z**2 + z_rayleigh**2)
    gouy = np.arctan(z / z_rayleigh)
    phase_geom = r_sq / 2.0 * curv - gouy

    field = a0_local * np.exp(1j * phase_geom)
    return t, field


def per_particle_R(t, field):
    """Batched FFT of `field` (= a0_local * exp(i*phase_geom), see
    sample_common_window_field) w.r.t. t (=omega_L*t_lab, the ordinary
    optical phase) gives Ẽ(nu), nu = Omega'/omega_L conjugate to that phase
    -- *not* yet R(Omega;p). Eq. (Rdef)'s Fourier variable is the *rescaled*
    phase xi = phi*(omega_L/omega_R), i.e. R is the Doppler-*stretched*
    image of this Ẽ: substituting u = xi*(omega_R/omega_L) in the integral
    gives R(Omega) prop (1/q)*|Ẽ(Omega/q)|^2 with q = omega_R/omega_L
    (per Eq. wR, q = 4*s_res for this geometry). In Delta_s units this
    collapses to a clean rescale-and-divide with no interpolation needed --
    see build_spectra, where s_res multiplies the sample offsets and cancels
    out of the deposited weight exactly. Works identically whether `field`
    is real (this module's first version, envelope only) or complex (with
    phase_geom): np.fft.fft/np.abs()**2 do not care, only the resulting R_u
    becomes properly asymmetric once the input carries a real phase.

    Returns (u grid = nu/4, *unstretched* per-particle normalised density
    R_u on that grid, *unstretched* rms width of R_u). Callers must multiply
    both the sample offset and the rms width by that particle's own s_res to
    get the physical (stretched) R_s(Delta_s) and dwsingle.
    """
    n_steps = t.shape[0]
    dt = t[1] - t[0]
    Efft = np.fft.fft(field, axis=1) * dt
    freq = np.fft.fftfreq(n_steps, d=dt) * 2 * np.pi  # nu = Omega'/omega_L, conjugate to phi = omega_L*t_lab
    R_raw = np.abs(Efft)**2
    dfreq = freq[1] - freq[0]
    norm = R_raw.sum(axis=1, keepdims=True) * dfreq
    norm = np.where(norm > 0, norm, 1.0)
    R_norm = R_raw / norm  # integrates to 1 over nu

    u_grid = freq / 4.0
    R_u = 4.0 * R_norm  # integrates to 1 over u = nu/4 (still unstretched)

    mean_u = np.sum(u_grid[None, :] * R_u, axis=1) * dfreq
    var_u = np.sum((u_grid[None, :] - mean_u[:, None])**2 * R_u, axis=1) * dfreq
    rms_width_u = np.sqrt(np.maximum(var_u, 0.0))
    return u_grid, R_u, rms_width_u


def per_particle_kinematics(compton, bunch, L, x0, y0):
    """L: per-particle luminosity functional, e.g. particles.push_and_sample's
    own 5th return value (that function now returns one row per particle --
    see its docstring -- so no external per-particle reshape/sum is needed,
    unlike this function's first version).
    """
    gamma = bunch.gamma
    theta_sq = (bunch.theta_x - x0)**2 + (bunch.theta_y - y0)**2
    gth_sq_inv = 1.0 / (1.0 + theta_sq * gamma**2)**2
    cos_pol = np.cos(P.PHI_POL - np.arctan2(bunch.theta_y - y0, bunch.theta_x - x0))**2
    a_fac = 1.0 - 4.0 * cos_pol * theta_sq * gamma**2 * gth_sq_inv
    prefactor = L * a_fac * gamma**5 * gth_sq_inv
    return theta_sq, prefactor


def accumulate_histograms(bunch, prefactor, theta_sq, ahat, s_edges, u_grid, R_u,
                           hist_delta_total, hist_direct_total):
    """Adds one batch's contribution to the running (raw, un-normalised)
    delta-model and direct-spectral-integration histograms on s_edges, both
    from the *same* per-particle prefactor -- Sec. 4's "the only difference
    is the delta substitution". In-place accumulation (not returned) so
    main() can process particles in RAM-bounded batches (see module
    docstring / main()'s batch-size derivation) without ever holding more
    than one batch's field/R_u arrays at once.

    The direct-integration path deposits each particle's own R_u (still on
    the unstretched u=freq/4 grid, see per_particle_R) at sample positions
    s_res*(1+u) instead of s_res+u -- the Doppler stretch by q=4*s_res
    derived in per_particle_R's docstring. The deposited *weight* is
    unchanged (prefactor*R_u*du): the two factors of s_res (from the
    stretched spacing and from R_s=R_u/s_res) cancel exactly, which is also
    why no interpolation is needed despite every particle having a
    different s_res.
    """
    gamma = bunch.gamma
    s_res = gamma**2 / (1.0 + gamma**2 * theta_sq + ahat)

    h, _ = np.histogram(s_res, bins=s_edges, weights=prefactor)
    hist_delta_total += h

    du = u_grid[1] - u_grid[0]
    s_samples = (s_res[:, None] * (1.0 + u_grid[None, :])).ravel()
    w_samples = (prefactor[:, None] * R_u * du).ravel()
    h, _ = np.histogram(s_samples, bins=s_edges, weights=w_samples)
    hist_direct_total += h
    return s_res


def spectra_from_histograms(compton, s_edges, hist_delta_total, hist_direct_total):
    ds = np.diff(s_edges)
    s_centers = 0.5 * (s_edges[1:] + s_edges[:-1])
    coef = 3.0 / (4.0 * np.pi**4 * compton.Wph * 4.0)
    spec_delta = coef * hist_delta_total / ds / s_centers**2
    spec_direct = coef * hist_direct_total / ds / s_centers**2
    return s_centers, spec_delta, spec_direct


def sub_bunch(bunch, sl):
    from xigma_i.particles import Bunch
    return Bunch(x0=bunch.x0[sl], y0=bunch.y0[sl], z0=bunch.z0[sl],
                 gamma=bunch.gamma[sl], theta_x=bunch.theta_x[sl], theta_y=bunch.theta_y[sl],
                 weight=bunch.weight)


def process_batch(compton, bunch, n_steps_R, x0, y0):
    """gamma/theta_x/theta_y/prefactor/ahat/s_res/dwsingle for one batch --
    everything main()'s accumulation loop needs, without holding more than
    one batch's (n_batch, n_steps_R) field/FFT arrays alive at a time.

    ahat/L (the resonance-shift and luminosity-weight inputs to the delta
    model) now come directly from particles.push_and_sample -- since
    xigma_i's Stage 0 was fixed (see git log) to compute exactly this
    trajectory-averaged effective intensity itself (one value per particle,
    Eq. ahattraj), rather than this file maintaining its own duplicate
    computation over a different (wider, particle-independent) time window.
    Only the *shape* of R (u_grid, R_u, from the FFT of the full complex
    field over that wider window -- needed for the direct-integration model,
    which push_and_sample has no reason to compute) still comes from this
    file's own sample_common_window_field/per_particle_R.
    """
    from xigma_i import particles as _particles
    _, _, _, ahat, L = _particles.push_and_sample(compton, bunch, n_steps=64)
    theta_sq, prefactor = per_particle_kinematics(compton, bunch, L, x0, y0)

    t, field = sample_common_window_field(compton, bunch, n_steps_R)
    u_grid, R_u, rms_width_u = per_particle_R(t, field)

    gamma = bunch.gamma
    s_res = gamma**2 / (1.0 + gamma**2 * theta_sq + ahat)
    dwsingle_per_particle = s_res * rms_width_u
    return theta_sq, prefactor, ahat, u_grid, R_u, dwsingle_per_particle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    PS.apply()
    compton = P.build_compton()
    print(f"[fig_validation] compton.a0 = {compton.a0:.4f} (Phi_NL=0 simplification's validity -- see module docstring)")

    # n_particles=20_000 (this function's first version) starved the fine
    # display histogram below: with n_fine=4000 bins spanning the display
    # window, that is ~5 macroparticles/bin on average, so panel (a) rendered
    # as shot-noise spikes for *both* curves rather than smooth spectra (the
    # windowed-error panel was fine regardless, since it aggregates many fine
    # bins per mu-window). bench.py timed this method at roughly 35-40
    # us/particle at n_steps_R=512 with the (real, envelope-only) field of
    # this function's first version; 100k particles is a ~15-20s cost there.
    # Since then field(t) became complex (envelope * geometric-phase factor)
    # plus several more (n_particles, n_steps_R) intermediates, several times
    # the memory per sample -- 100k particles * n_steps_R=2048 OOM-killed
    # (SIGKILL) on this project's 15 GB dev box. Processed in RAM-bounded
    # batches below instead of raising or lowering n_particles by feel: this
    # scales to any particle count on any machine, chunk size derived from
    # params.RAM_GB. See sample_common_window_field's docstring for why
    # n_steps_R can stay coarse (envelope-scale, not per-optical-cycle)
    # despite the added geometric phase.
    n_particles = 4_000 if args.quick else 100_000
    n_steps_R = 512 if args.quick else 2048
    n_mu = 6 if args.quick else 14
    n_fine = 300 if args.quick else 2000

    # ~9 real (n_batch, n_steps_R) float64 arrays (x, y, z, sigma_l_sq, env,
    # n_ph_shape, r_sq, curv, gouy/phase_geom) plus field/Efft (complex128,
    # 16 bytes) plus R_raw/R_norm/R_u (float64) in sample_common_window_field/
    # per_particle_R, measured against the OOM above -- ~190 bytes/sample is
    # a safety-padded estimate (measured peak RSS was higher, likely from
    # short-lived broadcasting temporaries this doesn't count), budgeted
    # against 20% of detected system RAM so other processes/the OS keep room.
    bytes_per_sample = 190
    batch_size = max(1_000, min(n_particles, int(0.2 * P.RAM_GB * 1e9 / bytes_per_sample / n_steps_R)))
    print(f"[fig_validation] RAM = {P.RAM_GB:.1f} GB -> batch size {batch_size} "
          f"({-(-n_particles // batch_size)} batches for {n_particles} particles)")

    x0, y0 = P.OBS_POINTS["on_axis"]

    bunch_full = R.make_bunch(compton, n_particles)

    # ---- pass 1: process the first batch, using it to size s_edges/mu_values
    first_end = min(batch_size, n_particles)
    first_batch = sub_bunch(bunch_full, slice(0, first_end))
    theta_sq0, prefactor0, ahat0, u_grid0, R_u0, dwsingle_pp0 = process_batch(compton, first_batch, n_steps_R, x0, y0)
    dwsingle = float(np.max(dwsingle_pp0))
    s_res0 = first_batch.gamma**2 / (1.0 + first_batch.gamma**2 * theta_sq0 + ahat0)
    s_center0 = float(np.median(s_res0))
    n_cyc_equivalent = s_center0 / dwsingle if dwsingle > 0 else np.inf
    print(f"[fig_validation] dwsingle (s-units, max over first batch of {first_batch.n_particles}) = {dwsingle:.3e}  "
          f"[dwsingle/wR ~ 1/{n_cyc_equivalent:.0f}, cf. laser N_cyc]")

    s_span_half = 60 * dwsingle
    print(f"[fig_validation] {n_particles} particles / {n_fine} display bins "
          f"= {n_particles / n_fine:.0f} particles/bin on average")
    s_edges = np.linspace(s_center0 - s_span_half, s_center0 + s_span_half, n_fine + 1)

    # ---- pass 2: accumulate the first batch, then the rest
    hist_delta_total = np.zeros(n_fine)
    hist_direct_total = np.zeros(n_fine)
    accumulate_histograms(first_batch, prefactor0, theta_sq0, ahat0, s_edges, u_grid0, R_u0,
                          hist_delta_total, hist_direct_total)
    dwsingle_max_seen = dwsingle
    for start in range(first_end, n_particles, batch_size):
        end = min(start + batch_size, n_particles)
        batch = sub_bunch(bunch_full, slice(start, end))
        theta_sq, prefactor, ahat, u_grid, R_u, dwsingle_pp = process_batch(compton, batch, n_steps_R, x0, y0)
        dwsingle_max_seen = max(dwsingle_max_seen, float(np.max(dwsingle_pp)))
        accumulate_histograms(batch, prefactor, theta_sq, ahat, s_edges, u_grid, R_u,
                              hist_delta_total, hist_direct_total)

    if dwsingle_max_seen > 1.5 * dwsingle:
        print(f"[fig_validation] WARNING: max dwsingle over the full bunch ({dwsingle_max_seen:.3e}) "
              f"exceeds the first-batch estimate used to size s_edges ({dwsingle:.3e}) by >50%; "
              f"a minority of particles' R may be clipped at the display window's edges.")

    s_c, spec_delta, spec_direct = spectra_from_histograms(compton, s_edges, hist_delta_total, hist_direct_total)

    # ------------------------------------------------------------- mu sweep
    mu_values = np.geomspace(0.4 * dwsingle, 40 * dwsingle, n_mu)
    l1_vals, mx_vals = [], []
    for mu in mu_values:
        l1, mx, _ = MT.window_integrated_relative_error(s_c, spec_delta, spec_direct, mu)
        l1_vals.append(l1)
        mx_vals.append(mx)
    l1_vals, mx_vals = np.array(l1_vals), np.array(mx_vals)

    n_fit = max(2, len(mu_values) - 3)
    slope, _ = np.polyfit(np.log(mu_values[:n_fit] / dwsingle), np.log(np.maximum(l1_vals[:n_fit], 1e-300)), 1)
    print(f"[fig_validation] fitted exponent of relative error vs mu/dwsingle: {slope:.2f} (predicted: -1)")

    # ------------------------------------------------- representative-mu overlay
    mu_rep = mu_values[len(mu_values) // 2]

    edge_mask = s_c > (s_center0 + 0.6 * s_span_half)
    body_mask = ~edge_mask
    l1_body, mx_body, _ = MT.window_integrated_relative_error(s_c[body_mask], spec_delta[body_mask],
                                                                 spec_direct[body_mask], mu_rep)
    l1_edge, mx_edge, _ = MT.window_integrated_relative_error(s_c[edge_mask], spec_delta[edge_mask],
                                                                 spec_direct[edge_mask], mu_rep)
    print(f"[fig_validation] at mu={mu_rep:.3e}: body l1={l1_body:.3e}, edge-band l1={l1_edge:.3e}")

    # ------------------------------------------------------------------ plot
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(2 * PS.COL_WIDTH_IN, PS.COL_WIDTH_IN * 0.8), layout="constrained")

    ax = axes[0]
    ax.plot(s_c, spec_delta, "-", color=PS.COLORS["delta_model"], label="delta-function model")
    ax.plot(s_c, spec_direct, "--", color=PS.COLORS["spectral_integration"], label="direct spectral integration")
    ax.set_xlabel("s")
    ax.set_ylabel("dN/ds (arb. norm.)")
    ax.set_title(f"(a) overlaid spectra, $\\mu$={mu_rep / dwsingle:.1f}$\\times\\delta\\omega$")
    ax.legend()

    ax = axes[1]
    ax.loglog(mu_values / dwsingle, l1_vals, "o-", color=PS.COLORS["delta_model"],
               label=f"fitted $\\propto (\\mu/\\delta\\omega)^{{{slope:.2f}}}$")
    PS.slope_guide(ax, mu_values / dwsingle, l1_vals[0], -1.0, r"$(\mu/\delta\omega)^{-1}$ predicted")
    ax.axvline(1.0, color="0.5", linestyle=":", linewidth=1.0)
    ax.set_xlabel(r"$\mu/\delta\omega$")
    ax.set_ylabel(r"weighted-$L_1$ relative error")
    ax.set_title("(b) window-integrated flux error")
    ax.legend()

    PS.save(fig, "fig_validation")

    PS.DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        PS.DATA_DIR / "fig_validation.npz",
        s_c=s_c, spec_delta=spec_delta, spec_direct=spec_direct,
        mu_values=mu_values, l1_vals=l1_vals, mx_vals=mx_vals, slope=slope,
        dwsingle=dwsingle, mu_rep=mu_rep, l1_body=l1_body, l1_edge=l1_edge,
        n_particles=n_particles, n_steps_R=n_steps_R, compton_a0=compton.a0,
        quick=args.quick,
    )
    print(f"[fig_validation] wrote {PS.FIGS_DIR / 'fig_validation.pdf'} and {PS.DATA_DIR / 'fig_validation.npz'}")


if __name__ == "__main__":
    main()
