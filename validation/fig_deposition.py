#!/usr/bin/env python
"""Fig. 3 -- deposition statistics and time step (convergence-validation.md
Sec. 3).

(a) error vs number of deposited trajectories N_p at fixed grid, nearest vs
cloud-in-cell. Independent particle draws per N_p (the point is *statistical*
convergence, so each N_p needs its own noise realisation -- unlike Fig. 2,
which deliberately reused one particle set across grid variants to remove
this same noise from the comparison).

Deliberately uses a *coarser* dedicated grid for this panel than the
production DEFAULT_N_BINS (half the bins per axis, floor 16). At the fine
production grid, nearest-cell's binning bias (set by cell width, N_p-
independent) can be small enough that shot noise stays above it across the
whole practical N_p range, so both schemes look statistics-limited
(propto N_p^-1/2) throughout and never show CLAUDE.md's expected "NGP
saturates, CIC keeps falling" -- the two effects need to be checked at a
resolution coarse enough (plan.md: "with realistic particle counts there
are only a few deposits per cell") for the bias floor to actually sit above
what CIC's shot noise reaches within the tested N_p. Prints per-scheme
median cell occupancy at the smallest N_p tested so this can be judged
directly rather than inferred from the curve shape alone.

(b) error vs trajectory time step dt (i.e. n_steps, at fixed N_p and grid).
Per CLAUDE.md/plan.md: a0 (the a0-axis coordinate) is one number per
particle-timestep as currently implemented in particles.push_and_sample --
n_steps controls the resolution of the trajectory integrals L(zeta) and
<a^2>(zeta) that ultimately populate that axis, which is what this panel's
scan actually tests; it is not re-splitting the emission itself (there is no
per-time-step emission model here, only a per-time-step *sampling* of the
already-defined weight/a0 integrands).

--quick: shorter N_p/n_steps sweeps and a coarser fine-reference table.
"""
import argparse
import time

import numpy as np

import params as P
import refs as R
import metrics as M
import plotstyle as PS
from xigma_i import deposition


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--rebuild-cache", action="store_true",
                    help="ignore any cached fine-reference table under data/cache/ and rebuild+overwrite it "
                         "(see refs.cached_table's docstring). Equivalent to VALIDATION_REBUILD_CACHE=1.")
    args = ap.parse_args()

    PS.apply()
    compton = P.build_compton()

    if args.quick:
        fine_overrides = dict(n_bins=(96, 96, 96, 24))
        fine_n_particles, fine_n_steps = 150_000, 48
        base_n_bins = (64, 48, 48, 16)
        n_steps_fixed = 32
        np_values = np.array([2_000, 8_000, 32_000, 128_000], dtype=int)
        n_p_fixed_for_dt = 80_000
        n_steps_values = np.array([4, 8, 16, 32, 64])
    else:
        print(f"[fig_deposition] hardware tier: {P.SCALE_TIER} (VALIDATION_SCALE to override)")
        fine_overrides = dict(n_bins=P.FINE_N_BINS)
        fine_n_particles, fine_n_steps = P.FINE_N_PARTICLES, P.FINE_N_STEPS
        base_n_bins = P.DEFAULT_N_BINS
        n_steps_fixed = P.DEFAULT_N_STEPS
        np_values = np.array(P.DEPOSITION_NP_VALUES, dtype=int)
        n_p_fixed_for_dt = P.DEPOSITION_NP_FIXED_FOR_DT
        n_steps_values = np.array([2, 4, 8, 16, 32, 64, 128])

    print(f"[fig_deposition] building fine reference table "
          f"(N_p={fine_n_particles}, n_steps={fine_n_steps}, n_bins={fine_overrides['n_bins']}) ...")
    t0 = time.time()
    table_fine = R.cached_table(compton, fine_n_particles, fine_n_steps, force=args.rebuild_cache, **fine_overrides)
    print(f"  done in {time.time() - t0:.1f}s")

    x0, y0 = P.OBS_POINTS["on_axis"]
    s = P.s_grid(96)
    ref_spec = R.ref_quadrature(table_fine, compton, x0, y0, s)
    mu = P.DEFAULT_MU

    # A fixed grid, extents derived from a modest, independent draw -- just
    # needs reasonable phase-space coverage, not the full size of the largest
    # N_p being tested (that would materialise a redundant, possibly huge
    # array purely to read off min/max+margin).
    grid_n_particles = min(np_values.max(), 200_000)
    grid_samples = R.make_samples(compton, grid_n_particles, n_steps_fixed, seed=P.DEFAULT_SEED - 1)
    base_grid = deposition.Grid4D.from_samples(*grid_samples[:4], n_bins=base_n_bins)

    # Coarser dedicated grid for panel (a) -- see module docstring.
    scheme_n_bins = tuple(max(16, nb // 2) for nb in base_n_bins)
    scheme_grid = deposition.Grid4D.from_samples(*grid_samples[:4], n_bins=scheme_n_bins)
    print(f"[fig_deposition] panel (a) grid: {scheme_n_bins} (production grid {base_n_bins} is used for panel (b))")

    # ---------------------------------------------------------- (a) N_p scan
    # Streamed (chunked draw+push+deposit) rather than R.make_samples, which
    # materialises N_p*n_steps host+GPU arrays at once -- fine at quick-mode
    # sizes but not at the largest N_p in params.py's medium/large tiers.
    # grid= pins the same fixed grid built above; chunking overhead at small
    # N_p is negligible, so this is used unconditionally, not just above a cutoff.
    print("[fig_deposition] (a) N_p scan, nearest vs cic ...")
    l1_nearest, l1_cic = [], []
    for i, n_p in enumerate(np_values):
        for scheme, store in (("nearest", l1_nearest), ("cic", l1_cic)):
            table = R.build_table_streaming(compton, int(n_p), n_steps_fixed, grid=scheme_grid, scheme=scheme,
                                             chunk_particles=P.STREAM_CHUNK_PARTICLES, seed=P.DEFAULT_SEED + i)
            spec = R.ref_quadrature(table, compton, x0, y0, s)
            l1, mx, _ = M.window_integrated_relative_error(s, spec, ref_spec, mu)
            store.append(l1)
            if n_p == np_values[0]:
                occ = table.occupancy
                populated = occ[occ > 0]
                median_occ = float(np.median(populated)) if populated.size else 0.0
                print(f"    [occupancy check @ N_p={n_p}, {scheme}] median deposits/populated cell = {median_occ:.1f}")
        print(f"    N_p={n_p}: nearest={l1_nearest[-1]:.3e}  cic={l1_cic[-1]:.3e}")
    l1_nearest, l1_cic = np.array(l1_nearest), np.array(l1_cic)

    # ---------------------------------------------------------- (b) dt scan
    # Expected shape, x-axis = dt (not n_steps): flat/converged at *small* dt
    # (large n_steps), rising at *large* dt (small n_steps, under-resolved
    # trajectory integral) -- i.e. the curve should look converged on the
    # left and increasing on the right of a log-x plot. The pulse used here
    # is multi-cycle (N_cyc ~ 10^3-10^4, a picosecond-class pulse), and only
    # the *envelope* integrals L(zeta)/<a^2>(zeta) are being resolved here,
    # not the fast optical cycle (see this script's module docstring) -- so
    # a converged dt of order 10^2-10^3 laser periods is the expected scale,
    # not a sign that something is wrong.
    print("[fig_deposition] (b) time-step (n_steps) scan at fixed N_p ...")
    l1_dt = []
    for n_steps in n_steps_values:
        table = R.build_table_streaming(compton, n_p_fixed_for_dt, int(n_steps), grid=base_grid, scheme="cic",
                                         chunk_particles=P.STREAM_CHUNK_PARTICLES, seed=P.DEFAULT_SEED + 100)
        spec = R.ref_quadrature(table, compton, x0, y0, s)
        l1, mx, _ = M.window_integrated_relative_error(s, spec, ref_spec, mu)
        l1_dt.append(l1)
        print(f"    n_steps={n_steps}: l1={l1:.3e}")
    l1_dt = np.array(l1_dt)

    # dt in laser periods: trajectory time window / n_steps, in units of 2*pi/omega_las.
    # The window itself varies (slightly) per particle; report the bunch-averaged span.
    t0_w, t1_w = particles_time_window_stats(compton)
    dt_values = (t1_w - t0_w) / n_steps_values / (2 * np.pi)

    n_conv = min(3, len(n_steps_values))
    dt_floor = float(np.median(l1_dt[-n_conv:]))
    dt_converged_idx = int(np.argmax(l1_dt <= 1.2 * dt_floor))
    dt_converged = float(dt_values[dt_converged_idx])
    print(f"[fig_deposition] time step converged (within 20% of floor {dt_floor:.3e}) "
          f"at dt <~ {dt_converged:.3e} laser periods")

    # ------------------------------------------------------------------ plot
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(2 * PS.COL_WIDTH_IN, PS.COL_WIDTH_IN * 0.8), layout="constrained")

    ax = axes[0]
    ax.loglog(np_values, l1_nearest, "s-", color=PS.COLORS["nearest"], label="nearest")
    ax.loglog(np_values, l1_cic, "o-", color=PS.COLORS["cic"], label="cloud-in-cell")
    PS.slope_guide(ax, np_values, l1_cic[0], -0.5, r"$N_p^{-1/2}$ guide")
    ax.set_xlabel(r"deposited trajectories, $N_p$")
    ax.set_ylabel(r"weighted-$L_1$ relative error")
    ax.set_title("(a) deposition scheme")
    ax.legend()

    ax = axes[1]
    ax.loglog(dt_values, l1_dt, "o-", color=PS.COLORS["gamma"])
    ax.axvline(dt_converged, color="0.5", linestyle=":", linewidth=1.0)
    ax.set_xlabel(r"trajectory time step $\Delta t$ [laser periods]")
    ax.set_title("(b) trajectory time step")

    PS.save(fig, "fig_deposition")

    PS.DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        PS.DATA_DIR / "fig_deposition.npz",
        np_values=np_values, l1_nearest=l1_nearest, l1_cic=l1_cic,
        n_steps_values=n_steps_values, dt_values=dt_values, l1_dt=l1_dt,
        dt_floor=dt_floor, dt_converged=dt_converged,
        mu=mu, s=s, quick=args.quick, base_n_bins=base_n_bins, scheme_n_bins=scheme_n_bins,
        n_steps_fixed=n_steps_fixed, n_p_fixed_for_dt=n_p_fixed_for_dt,
        fine_n_particles=fine_n_particles, fine_n_steps=fine_n_steps, fine_n_bins=fine_overrides["n_bins"],
    )
    print(f"[fig_deposition] wrote {PS.FIGS_DIR / 'fig_deposition.pdf'} and {PS.DATA_DIR / 'fig_deposition.npz'}")


def particles_time_window_stats(compton):
    """Bunch-averaged [t0, t1] trajectory window (see particles._time_window),
    used only to express n_steps as a physical dt in laser periods.
    """
    from xigma_i import particles
    z0 = np.linspace(-3 * compton.k0_las * compton.sigma_ez, 3 * compton.k0_las * compton.sigma_ez, 4096)
    t0, t1 = particles._time_window(compton, z0)
    return float(np.mean(t0)), float(np.mean(t1))


if __name__ == "__main__":
    main()
