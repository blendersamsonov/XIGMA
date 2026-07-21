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
  - ahat(zeta) = (Tr(Xi)/2) * <a0_local^4>_t / <a0_local^2>_t per particle
    (Eq. ahattraj), Tr(Xi)/2 = 1/2 for the linear polarisation used
    throughout xigma_i (paper, Sec. "Coherence matrix...": Xi=diag(1,0)).
  - R(Delta_s; ahat) as |FFT{a0_local(t)}|^2 over a common, particle-
    independent window +/- sigma_tau (matching particles.py's own
    conservative exposure bound), normalised to integrate to 1 in Delta_s
    (paper Eq. Rnorm).

SIMPLIFICATION KNOWINGLY MADE: the ponderomotive nonlinear phase Phi_NL
(Eq. PhiNL) is set to zero, i.e. R is computed from the *linear* bandwidth
only. The paper states this term "vanishes identically for a flat-top pulse"
and matters "when a0 >~ 1"; params.py's representative case has
compton.a0 ~ 0.09 (printed at runtime), so this is a quantitatively small
omission for the chosen parameter set, not a free pass in general. Anyone
reusing this at higher a0 needs to add Phi_NL back in (it requires nailing
down the phase origin for Eq. PhiNL's running average, which is not fully
determined by the sections of the paper read while writing this script).

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
TR_XI_HALF = 0.5  # linear polarisation, Xi = diag(1, 0); see module docstring


def sample_common_window_envelope(compton, bunch, n_steps):
    """a0_local(t) on a single, particle-independent time grid t in
    [-sigma_tau, sigma_tau] (sigma_tau = GAUSS_WIDTH * k0_las * sigma_lz, the
    same conservative temporal bound particles.py uses for its own
    per-particle exposure window) -- shared across particles so the FFT
    below is one batched call. See module docstring for why a common window
    is an adequate simplification (the temporal Gaussian factor dominates
    the window choice; a0_local already -> 0 outside a particle's real
    overlap regardless of what window it's sampled on).
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
    return t, a0_local


def per_particle_R(t, a0_local):
    """Batched FFT of the envelope w.r.t. t (=omega_L*t_lab, the ordinary
    optical phase) gives Ẽ(nu), nu = Omega'/omega_L conjugate to that phase
    -- *not* yet R(Omega;p). Eq. (Rdef)'s Fourier variable is the *rescaled*
    phase xi = phi*(omega_L/omega_R), i.e. R is the Doppler-*stretched*
    image of this Ẽ: substituting u = xi*(omega_R/omega_L) in the integral
    gives R(Omega) prop (1/q)*|Ẽ(Omega/q)|^2 with q = omega_R/omega_L
    (per Eq. wR, q = 4*s_res for this geometry). In Delta_s units this
    collapses to a clean rescale-and-divide with no interpolation needed --
    see build_spectra, where s_res multiplies the sample offsets and cancels
    out of the deposited weight exactly.

    Returns (u grid = nu/4, *unstretched* per-particle normalised density
    R_u on that grid, *unstretched* rms width of R_u). Callers must multiply
    both the sample offset and the rms width by that particle's own s_res to
    get the physical (stretched) R_s(Delta_s) and dwsingle.
    """
    n_steps = t.shape[0]
    dt = t[1] - t[0]
    Efft = np.fft.fft(a0_local, axis=1) * dt
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


def per_particle_kinematics(compton, bunch, gamma_flat, weight_flat, n_steps, n_particles, x0, y0):
    L = weight_flat.reshape(n_particles, n_steps).sum(axis=1)
    gamma = bunch.gamma
    theta_sq = (bunch.theta_x - x0)**2 + (bunch.theta_y - y0)**2
    gth_sq_inv = 1.0 / (1.0 + theta_sq * gamma**2)**2
    cos_pol = np.cos(P.PHI_POL - np.arctan2(bunch.theta_y - y0, bunch.theta_x - x0))**2
    a_fac = 1.0 - 4.0 * cos_pol * theta_sq * gamma**2 * gth_sq_inv
    prefactor = L * a_fac * gamma**5 * gth_sq_inv
    return theta_sq, prefactor


def build_spectra(compton, bunch, prefactor, theta_sq, ahat, s_edges, u_grid, R_u):
    """Delta-model and direct-spectral-integration histograms on s_edges,
    both from the *same* per-particle prefactor -- Sec. 4's "the only
    difference is the delta substitution".

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
    ds = np.diff(s_edges)
    s_centers = 0.5 * (s_edges[1:] + s_edges[:-1])
    coef = 3.0 / (4.0 * np.pi**4 * compton.Wph * 4.0)

    hist_delta, _ = np.histogram(s_res, bins=s_edges, weights=prefactor)
    spec_delta = coef * hist_delta / ds / s_centers**2

    du = u_grid[1] - u_grid[0]
    s_samples = (s_res[:, None] * (1.0 + u_grid[None, :])).ravel()
    w_samples = (prefactor[:, None] * R_u * du).ravel()
    hist_direct, _ = np.histogram(s_samples, bins=s_edges, weights=w_samples)
    spec_direct = coef * hist_direct / ds / s_centers**2

    return s_centers, s_res, spec_delta, spec_direct


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
    # us/particle at n_steps_R=512; 100k particles is a ~15-20s cost here,
    # not the "prohibitive" full-spectral-scan regime the paper means by that
    # term (that refers to doing this at every output point of a 2D angular
    # scan, not once for one direction).
    n_particles = 4_000 if args.quick else 100_000
    n_steps_R = 512 if args.quick else 2048
    n_mu = 6 if args.quick else 14
    n_fine = 300 if args.quick else 2000

    x0, y0 = P.OBS_POINTS["on_axis"]

    bunch = R.make_bunch(compton, n_particles)
    from xigma_i import particles as _particles
    gamma_flat, tx_flat, ty_flat, a0_flat, weight_flat = _particles.push_and_sample(compton, bunch, n_steps=64)

    theta_sq, prefactor = per_particle_kinematics(compton, bunch, gamma_flat, weight_flat, 64, n_particles, x0, y0)

    t, a0_local = sample_common_window_envelope(compton, bunch, n_steps_R)
    u_grid, R_u, rms_width_u = per_particle_R(t, a0_local)
    ahat = TR_XI_HALF * np.mean(a0_local**4, axis=1) / np.maximum(np.mean(a0_local**2, axis=1), 1e-300)

    gamma = bunch.gamma
    s_res = gamma**2 / (1.0 + gamma**2 * theta_sq + ahat)
    dwsingle_per_particle = s_res * rms_width_u  # Doppler-stretched width, see per_particle_R docstring
    dwsingle = float(np.max(dwsingle_per_particle))
    n_cyc_equivalent = float(np.median(s_res)) / dwsingle if dwsingle > 0 else np.inf
    print(f"[fig_validation] dwsingle (s-units, max over bunch) = {dwsingle:.3e}  "
          f"[dwsingle/wR ~ 1/{n_cyc_equivalent:.0f}, cf. laser N_cyc]")

    # ---------------------------------------------------------- fine s_edges
    s_span_half = 60 * dwsingle
    s_center0 = float(np.median(s_res))
    print(f"[fig_validation] {n_particles} particles / {n_fine} display bins "
          f"= {n_particles / n_fine:.0f} particles/bin on average")
    s_edges = np.linspace(s_center0 - s_span_half, s_center0 + s_span_half, n_fine + 1)

    s_c, s_res_check, spec_delta, spec_direct = build_spectra(
        compton, bunch, prefactor, theta_sq, ahat, s_edges, u_grid, R_u)

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
