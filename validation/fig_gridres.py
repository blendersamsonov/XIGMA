#!/usr/bin/env python
"""Fig. 2 -- table resolution (convergence-validation.md Sec. 2).

Two panels sharing a y-axis: error vs gamma-axis spacing and error vs
a0^2-axis ("ahat") spacing, each expressed in units of the energy interval
mu*gamma/(2*omega) of one reporting-resolution element (Sec. "validity4" of
the paper). Since reference.spectrum_from_table is an *exact* grid
quadrature (a Riemann sum over the table's own cells, no sampling noise
whatsoever), "fixed high M so quadrature noise is below the effect being
measured" is achieved trivially here by not using any Monte-Carlo/QMC
quadrature at all for this figure -- there is nothing to converge in M.

Normalisation derivation (not in any existing module -- derived here from
Eq. wR, s = gamma^2/(1+gamma^2 theta^2 + ahat)): at the on-axis peak
(theta=0, ahat<<1), ds/dgamma = 2*gamma0, so one resolution element in s
(width mu) corresponds to d_gamma = mu/(2*gamma0). Likewise ds/d(ahat) =
-gamma0^2, so d(ahat) = mu/gamma0^2. ahat's role in the resonance condition
is played by a0^2 (up to the TrXi/2=1/2 polarisation factor this module
does not need, since it cancels out of the *spacing* ratio), so the table's
a0-axis is converted to a0^2 spacing via d(a0^2) = 2*a0_typical*d(a0),
evaluated at a0_typical = compton.a0 (bin nearest the laser peak amplitude,
where the population is concentrated).

The *same* fixed particle sample set is deposited into every grid variant
(only the scanned axis's bin count changes; the sampling noise from a
finite particle draw is thereby common to every point of a scan and does
not contaminate the resolution comparison -- that is Fig. 3's subject).
This separation is only clean as long as N_p stays large enough, *at every
bin count tested*, that per-cell shot noise is negligible next to the
discretisation error being measured -- refining the grid at *fixed* N_p
eventually starves individual cells and reintroduces exactly the Fig.-3
noise this figure is trying to exclude, showing up as the error curve
turning back *up* at the finest bin counts instead of flattening (a
bias/variance crossover, not a bug in the resolution scan itself). This
script prints the finest trial's occupancy so that crossover -- if the
particle count used isn't large enough for the bin-count range requested --
is visible rather than silently mistaken for noise in the resolution effect.

KNOWN, EXPECTED RESULT -- not a bug: panel (b)'s error is *exactly* flat
against a0-axis bin count (see the printed note at runtime). spectrum_from_
table's (and spectrum_kernel_4d's) resonance-condition inversion,
`g_sq = 1/(1/s - r_sq)`, does not depend on a0/ahat at all -- a0 only
selects which table cells contribute to the coarse population weighting, and
summing H over the *entire* a0 axis gives the same total regardless of how
many bins that axis is divided into (interpolation commutes with the sum,
same argument as quadrature.py's optimisation). Physically, Eq. wR *does*
include ahat in the resonance denominator; xigma_i's current implementation
elects not to use it there (see spectrum4d.py's module docstring). Worth
flagging to whoever maintains that code before quoting this panel as
evidence about the *paper's* resolution criterion for a0/ahat, since it is
currently evidence about this implementation choice instead.

--quick: fewer particles/coarser fine-reference table and a short bin-count
sweep (minutes); full mode uses params.py's FINE_*/DEFAULT_* settings and a
wider sweep.
"""
import argparse
import time

import numpy as np
import cupy as cp

import params as P
import refs as R
import metrics as M
import plotstyle as PS
from xigma_i import deposition


def build_variant_table(samples, base_grid, axis, n_bins_axis):
    """Redeposit `samples` into a grid identical to base_grid except that
    axis `axis` (0=gamma, 3=a0) has n_bins_axis bins instead of base_grid's.
    """
    gamma, tx, ty, a0, w = samples
    edges = [base_grid.gamma_edges, base_grid.theta_x_edges, base_grid.theta_y_edges, base_grid.a0_edges]
    edges[axis] = np.linspace(edges[axis][0], edges[axis][-1], n_bins_axis + 1)
    grid = deposition.Grid4D(*edges)
    H_raw, occupancy, n_discarded = R.deposit_in_chunks(grid, gamma, tx, ty, a0, w)
    return _finish_table(H_raw, occupancy, n_discarded, grid, gamma.shape[0])


def _finish_table(H_raw, occupancy, n_discarded, grid, n_samples):
    H_raw = H_raw.get()
    occupancy = occupancy.get()
    cp.get_default_memory_pool().free_all_blocks()
    H_density = H_raw / grid.bin_volume
    bracket = deposition.gamma_bracket(H_density, grid, q=1e-4)
    return deposition.Table(H=H_density, grid=grid, scheme="cic", n_particle_samples=n_samples,
                             total_weight=float(H_raw.sum()), n_discarded=n_discarded,
                             occupancy=occupancy, gamma_bracket=bracket)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    PS.apply()
    compton = P.build_compton()

    if args.quick:
        fine_overrides = dict(n_bins=(96, 96, 96, 24))
        fine_n_particles, fine_n_steps = 150_000, 48
        n_particles, n_steps = 60_000, 32
        base_n_bins = (96, 48, 48, 16)
        gamma_bin_counts = np.array([8, 16, 32, 64, 128])
        a0_bin_counts = np.array([2, 4, 8, 16, 32])
    else:
        print(f"[fig_gridres] hardware tier: {P.SCALE_TIER} (VALIDATION_SCALE to override)")
        fine_overrides = dict(n_bins=P.FINE_N_BINS)
        fine_n_particles, fine_n_steps = P.FINE_N_PARTICLES, P.FINE_N_STEPS
        n_particles, n_steps = P.GRIDRES_N_PARTICLES, P.GRIDRES_N_STEPS
        gg, gt = P.DEFAULT_N_BINS[0], P.DEFAULT_N_BINS[1]
        base_n_bins = (gg, gt, gt, P.DEFAULT_N_BINS[3])
        gamma_bin_counts = np.array([8, 16, 32, 64, 96, 128, 192, 256])
        gamma_bin_counts = gamma_bin_counts[gamma_bin_counts <= 2 * gg]
        a0_bin_counts = np.array([2, 4, 8, 12, 16, 24, 32, 48])

    print(f"[fig_gridres] building fine reference table "
          f"(N_p={fine_n_particles}, n_steps={fine_n_steps}, n_bins={fine_overrides['n_bins']}) ...")
    t0 = time.time()
    table_fine = R.cached_table(compton, fine_n_particles, fine_n_steps, **fine_overrides)
    print(f"  done in {time.time() - t0:.1f}s")

    x0, y0 = P.OBS_POINTS["on_axis"]
    s = P.s_grid(96)
    ref_spec = R.ref_quadrature(table_fine, compton, x0, y0, s)
    mu = P.DEFAULT_MU

    print(f"[fig_gridres] drawing fixed particle sample set (N_p={n_particles}, n_steps={n_steps}) ...")
    samples = R.make_samples(compton, n_particles, n_steps)
    base_grid = deposition.Grid4D.from_samples(*samples[:4], n_bins=base_n_bins)

    d_gamma_unit = mu / (2.0 * P.GAMMA0)          # one resolution element, gamma units
    d_a0sq_unit = mu / P.GAMMA0**2                # one resolution element, a0^2=ahat units

    def scan(axis, bin_counts, unit, spacing_fn, report_occupancy=False):
        l1s, mxs = [], []
        for nb in bin_counts:
            table = build_variant_table(samples, base_grid, axis, int(nb))
            spec = R.ref_quadrature(table, compton, x0, y0, s)
            l1, mx, _ = M.window_integrated_relative_error(s, spec, ref_spec, mu)
            l1s.append(l1)
            mxs.append(mx)
            if report_occupancy and nb == bin_counts[-1]:
                occ = table.occupancy
                populated = occ[occ > 0]
                median_occ = float(np.median(populated)) if populated.size else 0.0
                print(f"    [occupancy check @ finest bin count={nb}] "
                      f"median deposits/populated cell = {median_occ:.1f}, "
                      f"empty-cell fraction = {1.0 - populated.size / occ.size:.3f}")
                if median_occ < 20:
                    print(f"    WARNING: median occupancy {median_occ:.1f} is low -- shot noise from finite "
                          f"N_p may be contaminating this scan's finest point(s) (see module docstring). "
                          f"Consider raising GRIDRES_N_PARTICLES or narrowing the bin-count range.")
        spacing = spacing_fn(bin_counts)
        return np.array(l1s), np.array(mxs), spacing / unit

    print("[fig_gridres] scanning gamma-axis bin count ...")
    l1_gamma, mx_gamma, x_gamma = scan(
        0, gamma_bin_counts, d_gamma_unit,
        lambda nb: (base_grid.gamma_edges[-1] - base_grid.gamma_edges[0]) / nb,
        report_occupancy=True)

    print("[fig_gridres] scanning a0-axis bin count ...")

    def a0_spacing(nb):
        da0 = (base_grid.a0_edges[-1] - base_grid.a0_edges[0]) / nb
        return 2.0 * compton.a0 * da0  # d(a0^2) = 2 a0 d(a0), at the laser's peak a0

    l1_a0, mx_a0, x_a0 = scan(3, a0_bin_counts, d_a0sq_unit, a0_spacing)
    a0_flat = bool(np.allclose(l1_a0, l1_a0[0]))
    if a0_flat:
        print("[fig_gridres] a0-axis error is exactly flat across bin count -- expected with the current "
              "implementation, not a bug; see module docstring (spectrum_from_table's g_sq does not use a0/ahat).")

    print(f"[fig_gridres] gamma-axis: error at coarsest/finest = {l1_gamma[0]:.3e} / {l1_gamma[-1]:.3e}")
    print(f"[fig_gridres] a0-axis:    error at coarsest/finest = {l1_a0[0]:.3e} / {l1_a0[-1]:.3e}")

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(2 * PS.COL_WIDTH_IN, PS.COL_WIDTH_IN * 0.8),
                              sharey=True, layout="constrained")

    ax = axes[0]
    ax.loglog(x_gamma, l1_gamma, "o-", color=PS.COLORS["gamma"])
    ax.axvline(1.0, color="0.5", linestyle=":", linewidth=1.0)
    ax.set_xlabel(r"$\gamma$-axis spacing / $(\mu\gamma/2\omega)$")
    ax.set_ylabel(r"weighted-$L_1$ relative error")
    ax.set_title(r"(a) $\gamma$ resolution")

    ax = axes[1]
    ax.loglog(x_a0, l1_a0, "s-", color=PS.COLORS["a0"])
    ax.axvline(1.0, color="0.5", linestyle=":", linewidth=1.0)
    ax.set_xlabel(r"$\hat{a}$-axis spacing / $(\mu\gamma/2\omega)$")
    ax.set_title(r"(b) $\hat{a}$ resolution")
    if a0_flat:
        ax.text(0.5, 0.5, "flat: current spectrum_from_table\ndoes not use $\\hat{a}$ in the\nresonance condition (see script docstring)",
                transform=ax.transAxes, ha="center", va="center", fontsize=6.5, color="0.35",
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7", alpha=0.85))

    PS.save(fig, "fig_gridres")

    PS.DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        PS.DATA_DIR / "fig_gridres.npz",
        gamma_bin_counts=gamma_bin_counts, a0_bin_counts=a0_bin_counts,
        x_gamma=x_gamma, l1_gamma=l1_gamma, mx_gamma=mx_gamma,
        x_a0=x_a0, l1_a0=l1_a0, mx_a0=mx_a0,
        mu=mu, s=s, quick=args.quick,
        n_particles=n_particles, n_steps=n_steps, base_n_bins=base_n_bins,
        fine_n_particles=fine_n_particles, fine_n_steps=fine_n_steps, fine_n_bins=fine_overrides["n_bins"],
    )
    print(f"[fig_gridres] wrote {PS.FIGS_DIR / 'fig_gridres.pdf'} and {PS.DATA_DIR / 'fig_gridres.npz'}")


if __name__ == "__main__":
    main()
