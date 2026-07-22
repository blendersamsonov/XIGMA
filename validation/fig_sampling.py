#!/usr/bin/env python
"""Fig. 1 -- sample-count convergence (convergence-validation.md Sec. 1).

Four curves at equal sample budget M, crossing two independent design axes:
whether the (theta_x, theta_y) integral is importance-sampled at all, and
whether the point-generation sequence is low-discrepancy (Fibonacci/
golden-ratio) or classical Monte Carlo:

  - "fibonacci"          -- box_quadrature_spectrum, no importance sampling,
                            a Fibonacci lattice over the whole table box.
  - "pseudorandom"        -- box_quadrature_spectrum, no importance sampling,
                            i.i.d. pseudorandom points (numpy Generator).
  - "importance, kernel"  -- the real, compiled spectrum_kernel_4d
                            (calculate_angular_spectrum_4d): production
                            importance sampling with its own golden-ratio/
                            stratified-CDF point generation. Its own M-like
                            knob (samples_per_point) spans only ~2-3 decades
                            before cost becomes impractical (quadrature.py's
                            module docstring), so this curve covers a
                            narrower M range than the other three; plotted
                            at samples_per_point*spectrum4d.SAMPLES_TOTAL
                            (the kernel's actual per-point sample count) so
                            the x-axis stays directly comparable.
  - "importance, hash"    -- quadrature.py's own from-scratch importance
                            sampler (importance_quadrature_spectrum), using
                            the *same* H_marginal proposal density as the
                            kernel, but with a stateless GPU-friendly hash
                            function standing in for the kernel's
                            golden-ratio sequence -- imitating classical/
                            white-noise Monte Carlo under importance
                            sampling. See quadrature.py's module docstring
                            for why this is a from-scratch estimator rather
                            than a second rawkernel, and for the design
                            history (a first, annulus-restricted version of
                            this sampler had a real ~40% coverage bug on
                            this codebase's actual, very elongated tables --
                            caught by comparing against ref_quadrature at
                            increasing M and seeing the bias *not* shrink).

The no-importance-sampling curves run at all three representative
observation points (params.OBS_POINTS) and report each separately, since
the point of the figure is convergence behaviour, not the spectrum shape.
The two importance-sampled curves run on-axis only, to keep runtime
bounded: both cost substantially more per M than the plain box quadrature
(the kernel needs a real GPU launch per samples_per_point value; the
hash/Fibonacci importance sampler pays the same per-a0-bin quadrature cost
`_integrand` already imposes on box_quadrature_spectrum, see quadrature.py's
module docstring -- none of this is GPU-accelerated, so a bigger GPU does
not speed up the on-CPU curves the way it does the other figures here).
importance_quadrature_spectrum's own "fibonacci" mode is also computed (not
plotted) purely as an internal cross-check: since it uses the same H_marginal
proposal as the compiled kernel, its convergence should track the kernel
curve's, confirming the "hash" comparison isolates the point-generation
scheme rather than a different importance density.

All four report relative error in window-integrated flux (metrics.py)
against the same fine-table reference, log-log against M, with reference
slopes -1/2 and -1.

--quick: smaller reference/production tables, a shorter M sweep, no
pseudorandom/hash repeats, runs in a few minutes; full mode repeats the
pseudorandom and hash draws and uses the full-size tables from params.py.
"""
import argparse
import time

import numpy as np
import cupy as cp

import params as P
import refs as R
import metrics as M
import quadrature as Q
import plotstyle as PS
from xigma_i import spectrum4d


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
        kernel_samples_per_point_values = np.array([1, 4, 16, 64], dtype=int)
    else:
        print(f"[fig_sampling] hardware tier: {P.SCALE_TIER} (VALIDATION_SCALE to override)")
        fine_overrides = dict(n_bins=P.FINE_N_BINS)
        fine_n_particles, fine_n_steps = P.FINE_N_PARTICLES, P.FINE_N_STEPS
        default_overrides = dict(n_bins=P.DEFAULT_N_BINS)
        default_n_particles, default_n_steps = P.DEFAULT_N_PARTICLES, P.DEFAULT_N_STEPS
        m_values = np.array(P.SAMPLING_M_VALUES, dtype=int)
        n_repeats_random = 4
        n_s = 64
        kernel_samples_per_point_values = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=int)

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

    # -------------------------- importance-sampled curves, on-axis only ----
    # (see module docstring: substantially costlier per M than the plain box
    # quadrature above, and neither is GPU-accelerated on the quadrature.py
    # side, so this is run only at the one observation point the figure
    # actually plots.)
    x0, y0 = P.OBS_POINTS["on_axis"]
    ref_spec_axis = R.ref_quadrature(table_fine, compton, x0, y0, s)

    print("[fig_sampling] importance-sampled curve: production kernel (importance + Fibonacci) ...")
    l1_kernel, mx_kernel = [], []
    theta_x_gpu = cp.asarray([x0], dtype=cp.float32)
    theta_y_gpu = cp.asarray([y0], dtype=cp.float32)
    s_gpu = cp.asarray(s, dtype=cp.float32)
    kernel_m_values = kernel_samples_per_point_values * spectrum4d.SAMPLES_TOTAL
    for spp in kernel_samples_per_point_values:
        spec, _, _ = spectrum4d.calculate_angular_spectrum_4d(
            compton, table, s_gpu, theta_x_gpu, theta_y_gpu, P.PHI_POL, samples_per_point=int(spp))
        spec = np.asarray(spec, dtype=np.float64).reshape(-1)
        l1, mx, _ = M.window_integrated_relative_error(s, spec, ref_spec_axis, mu)
        l1_kernel.append(l1)
        mx_kernel.append(mx)
    l1_kernel, mx_kernel = np.array(l1_kernel), np.array(mx_kernel)

    print("[fig_sampling] importance-sampled curve: importance + hash (classical Monte Carlo) ...")
    l1_imp_fib, mx_imp_fib = [], []
    l1_imp_hash, mx_imp_hash = [], []
    for m in m_values:
        spec_imp_fib = Q.importance_quadrature_spectrum(table, compton, x0, y0, s, int(m), "fibonacci", P.PHI_POL)
        l1, mx, _ = M.window_integrated_relative_error(s, spec_imp_fib, ref_spec_axis, mu)
        l1_imp_fib.append(l1)
        mx_imp_fib.append(mx)

        l1_h, mx_h = [], []
        for rep in range(n_repeats_random):
            spec_imp_hash = Q.importance_quadrature_spectrum(table, compton, x0, y0, s, int(m), "hash", P.PHI_POL,
                                                               salt0=3 * rep)
            l1_, mx_, _ = M.window_integrated_relative_error(s, spec_imp_hash, ref_spec_axis, mu)
            l1_h.append(l1_)
            mx_h.append(mx_)
        l1_imp_hash.append(float(np.mean(l1_h)))
        mx_imp_hash.append(float(np.mean(mx_h)))
    l1_imp_fib, mx_imp_fib = np.array(l1_imp_fib), np.array(mx_imp_fib)
    l1_imp_hash, mx_imp_hash = np.array(l1_imp_hash), np.array(mx_imp_hash)

    # ---- fit exponents and identify the systematic floor (on-axis point) ----
    primary = results["on_axis"]
    slope_fib, _ = fit_exponent(m_values, primary["l1_fib"])
    slope_rnd, _ = fit_exponent(m_values, primary["l1_rnd"])
    slope_kernel, _ = fit_exponent(kernel_m_values, l1_kernel)
    slope_imp_fib, _ = fit_exponent(m_values, l1_imp_fib)
    slope_imp_hash, _ = fit_exponent(m_values, l1_imp_hash)
    floor = float(np.median(primary["l1_fib"][-3:]))
    # M at which the Fibonacci curve first comes within 20% of the floor
    m_flatten = int(m_values[np.argmax(primary["l1_fib"] <= 1.2 * floor)])

    print(f"[fig_sampling] fitted exponents (on-axis, weighted-L1): "
          f"Fibonacci ~ M^{slope_fib:.2f}, pseudorandom ~ M^{slope_rnd:.2f}, "
          f"importance+kernel ~ M^{slope_kernel:.2f}, importance+hash ~ M^{slope_imp_hash:.2f}")
    print(f"[fig_sampling] cross-check: importance sampler's own Fibonacci mode ~ M^{slope_imp_fib:.2f} "
          f"(not plotted -- should track importance+kernel's slope above, confirming the hash comparison "
          f"isolates the point-generation scheme rather than a different importance density)")
    print(f"[fig_sampling] systematic floor (table-resolution-limited): {floor:.3e}, "
          f"reached near M={m_flatten}")

    # ---------------------------------------------------------------- plot
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.loglog(m_values, primary["l1_fib"], "o-", color=PS.COLORS["fibonacci"],
               label=f"Fibonacci ($\\propto M^{{{slope_fib:.2f}}}$)")
    ax.loglog(m_values, primary["l1_rnd"], "s-", color=PS.COLORS["pseudorandom"],
               label=f"pseudorandom ($\\propto M^{{{slope_rnd:.2f}}}$)")
    ax.loglog(kernel_m_values, l1_kernel, "^-", color=PS.COLORS["importance_kernel"],
               label=f"importance + Fibonacci, kernel ($\\propto M^{{{slope_kernel:.2f}}}$)")
    ax.loglog(m_values, l1_imp_hash, "d-", color=PS.COLORS["importance_hash"],
               label=f"importance + hash ($\\propto M^{{{slope_imp_hash:.2f}}}$)")
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
        kernel_samples_per_point_values=kernel_samples_per_point_values, kernel_m_values=kernel_m_values,
        l1_kernel=l1_kernel, mx_kernel=mx_kernel,
        l1_imp_fib=l1_imp_fib, mx_imp_fib=mx_imp_fib,
        l1_imp_hash=l1_imp_hash, mx_imp_hash=mx_imp_hash,
        slope_fib=slope_fib, slope_rnd=slope_rnd, slope_kernel=slope_kernel,
        slope_imp_fib=slope_imp_fib, slope_imp_hash=slope_imp_hash,
        floor=floor, m_flatten=m_flatten,
        quick=args.quick,
        fine_n_particles=fine_n_particles, fine_n_steps=fine_n_steps, fine_n_bins=fine_overrides["n_bins"],
        default_n_particles=default_n_particles, default_n_steps=default_n_steps,
        default_n_bins=default_overrides["n_bins"],
    )
    print(f"[fig_sampling] wrote {PS.FIGS_DIR / 'fig_sampling.pdf'} and {PS.DATA_DIR / 'fig_sampling.npz'}")


if __name__ == "__main__":
    main()
