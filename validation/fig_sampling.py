#!/usr/bin/env python
"""Fig. 1 -- sample-count convergence (convergence-validation.md Sec. 1).

Sweeps M (samples per output point) over several decades for Fibonacci
(low-discrepancy) vs pseudorandom sampling at equal budget, at a fixed table,
and plots the relative error in window-integrated flux (metrics.py) against
M, log-log, with reference slopes -1/2 and -1.

See quadrature.py's module docstring for why the M-scan is done with a
from-scratch CPU box-quadrature harness on the same physical integrand
rather than by instrumenting spectrum_kernel_4d directly (the compiled
kernel's own M-like knob only spans ~2-3 decades and has no pseudorandom
variant).

Run at all three representative observation points (on-axis, mid-spectrum,
near the Compton edge; params.OBS_POINTS) and reports each separately, since
the point of the figure is convergence behaviour, not the spectrum shape.

--quick: smaller reference/production tables and a shorter M sweep with no
pseudorandom repeats, runs in a couple of minutes; full mode repeats the
pseudorandom draw and uses the full-size tables from params.py.
"""
import argparse
import time

import numpy as np

import params as P
import refs as R
import metrics as M
import quadrature as Q
import plotstyle as PS


def fit_exponent(x, y):
    """Least-squares log-log slope, fit only over points still clearly above
    the systematic floor.

    A fixed "drop the last 3 points" rule (this function's first version)
    does not adapt to *when* a given curve reaches its floor: Fibonacci
    reaches it far sooner than pseudorandom at the same M (that is the whole
    point of the figure), so a fixed cutoff leaves mostly-flat points in
    Fibonacci's fit and mostly-decaying points in pseudorandom's, diluting
    the reported Fibonacci exponent toward the pseudorandom one -- exactly
    the "both curves report the same slope" confusion seen in practice.
    Estimate the floor from the last 3 points instead, and fit only the
    points at least 2x above it (falling back to all points if fewer than 2
    qualify, e.g. a curve that never really converges within the sweep).
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    floor = float(np.median(y[-3:]))
    mask = y > 2.0 * floor
    if mask.sum() < 2:
        mask = np.ones_like(y, dtype=bool)
    lx, ly = np.log(x[mask]), np.log(np.maximum(y[mask], 1e-300))
    slope, intercept = np.polyfit(lx, ly, 1)
    return slope, intercept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--rebuild-cache", action="store_true",
                    help="ignore any cached table under data/cache/ and rebuild+overwrite it -- needed "
                         "when running across machines kept in sync by a file-sync tool rather than git, "
                         "since the cache key is a hash of parameters, not of the code that built it, and "
                         "so does not itself detect a code change (e.g. a bug fix in refs.py/deposition.py). "
                         "Equivalent to setting VALIDATION_REBUILD_CACHE=1 for this run.")
    args = ap.parse_args()

    PS.apply()
    compton = P.build_compton()

    if args.quick:
        fine_overrides = dict(n_bins=(96, 96, 96, 24))
        fine_n_particles, fine_n_steps = 150_000, 48
        default_overrides = dict(n_bins=(64, 64, 64, 16))
        default_n_particles, default_n_steps = 40_000, 32
        m_values = np.array([8, 32, 128, 512, 2048, 8192], dtype=int)
        n_repeats_random = 1
        n_s = 40
    else:
        print(f"[fig_sampling] hardware tier: {P.SCALE_TIER} (VALIDATION_SCALE to override)")
        fine_overrides = dict(n_bins=P.FINE_N_BINS)
        fine_n_particles, fine_n_steps = P.FINE_N_PARTICLES, P.FINE_N_STEPS
        default_overrides = dict(n_bins=P.DEFAULT_N_BINS)
        default_n_particles, default_n_steps = P.DEFAULT_N_PARTICLES, P.DEFAULT_N_STEPS
        m_values = np.array(P.SAMPLING_M_VALUES, dtype=int)
        n_repeats_random = 4
        n_s = 64

    print(f"[fig_sampling] building fine reference table "
          f"(N_p={fine_n_particles}, n_steps={fine_n_steps}, n_bins={fine_overrides['n_bins']}) ...")
    t0 = time.time()
    table_fine = R.cached_table(compton, fine_n_particles, fine_n_steps, force=args.rebuild_cache, **fine_overrides)
    print(f"  done in {time.time() - t0:.1f}s")

    print(f"[fig_sampling] building production table "
          f"(N_p={default_n_particles}, n_steps={default_n_steps}, n_bins={default_overrides['n_bins']}) ...")
    t0 = time.time()
    table = R.cached_table(compton, default_n_particles, default_n_steps, force=args.rebuild_cache, **default_overrides)
    print(f"  done in {time.time() - t0:.1f}s")

    s = P.s_grid(n_s)
    mu = P.DEFAULT_MU
    rng = np.random.default_rng(P.DEFAULT_SEED)

    results = {}
    for name, (x0, y0) in P.OBS_POINTS.items():
        print(f"[fig_sampling] observation point {name} = ({x0:.3e}, {y0:.3e}) ...")
        ref_spec = R.ref_quadrature(table_fine, compton, x0, y0, s)

        l1_fib, mx_fib = [], []
        l1_rnd, mx_rnd = [], []
        for m in m_values:
            spec_fib = Q.box_quadrature_spectrum(table, compton, x0, y0, s, int(m), "fibonacci", P.PHI_POL)
            l1, mx, _ = M.window_integrated_relative_error(s, spec_fib, ref_spec, mu)
            l1_fib.append(l1)
            mx_fib.append(mx)

            l1_r, mx_r = [], []
            for _ in range(n_repeats_random):
                spec_rnd = Q.box_quadrature_spectrum(table, compton, x0, y0, s, int(m), "pseudorandom",
                                                       P.PHI_POL, rng=rng)
                l1_, mx_, _ = M.window_integrated_relative_error(s, spec_rnd, ref_spec, mu)
                l1_r.append(l1_)
                mx_r.append(mx_)
            l1_rnd.append(float(np.mean(l1_r)))
            mx_rnd.append(float(np.mean(mx_r)))

        results[name] = dict(l1_fib=np.array(l1_fib), mx_fib=np.array(mx_fib),
                              l1_rnd=np.array(l1_rnd), mx_rnd=np.array(mx_rnd))

    # ---- fit exponents and identify the systematic floor (on-axis point) ----
    primary = results["on_axis"]
    slope_fib, _ = fit_exponent(m_values, primary["l1_fib"])
    slope_rnd, _ = fit_exponent(m_values, primary["l1_rnd"])
    floor = float(np.median(primary["l1_fib"][-3:]))
    # M at which the Fibonacci curve first comes within 20% of the floor
    m_flatten = int(m_values[np.argmax(primary["l1_fib"] <= 1.2 * floor)])

    print(f"[fig_sampling] fitted exponents (on-axis, weighted-L1): "
          f"Fibonacci ~ M^{slope_fib:.2f}, pseudorandom ~ M^{slope_rnd:.2f}")
    print(f"[fig_sampling] systematic floor (table-resolution-limited): {floor:.3e}, "
          f"reached near M={m_flatten}")

    # ---------------------------------------------------------------- plot
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.loglog(m_values, primary["l1_fib"], "o-", color=PS.COLORS["fibonacci"],
               label=f"Fibonacci ($\\propto M^{{{slope_fib:.2f}}}$)")
    ax.loglog(m_values, primary["l1_rnd"], "s-", color=PS.COLORS["pseudorandom"],
               label=f"pseudorandom ($\\propto M^{{{slope_rnd:.2f}}}$)")
    PS.slope_guide(ax, m_values, primary["l1_fib"][0], -1.0, r"$M^{-1}$ guide")
    PS.slope_guide(ax, m_values, primary["l1_rnd"][0], -0.5, r"$M^{-1/2}$ guide")
    ax.axhline(floor, color="0.5", linestyle=":", linewidth=1.0, label="table-resolution floor")
    ax.set_xlabel("samples per output point, $M$")
    ax.set_ylabel(r"weighted-$L_1$ relative error, window-integrated flux")
    ax.set_title("on-axis observation point")
    ax.legend()
    PS.save(fig, "fig_sampling")

    # -------------------------------------------------------------- save data
    PS.DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        PS.DATA_DIR / "fig_sampling.npz",
        m_values=m_values, mu=mu, s=s,
        **{f"{name}_{k}": v for name, d in results.items() for k, v in d.items()},
        slope_fib=slope_fib, slope_rnd=slope_rnd, floor=floor, m_flatten=m_flatten,
        quick=args.quick,
        fine_n_particles=fine_n_particles, fine_n_steps=fine_n_steps, fine_n_bins=fine_overrides["n_bins"],
        default_n_particles=default_n_particles, default_n_steps=default_n_steps,
        default_n_bins=default_overrides["n_bins"],
    )
    print(f"[fig_sampling] wrote {PS.FIGS_DIR / 'fig_sampling.pdf'} and {PS.DATA_DIR / 'fig_sampling.npz'}")


if __name__ == "__main__":
    main()
