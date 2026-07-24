#!/usr/bin/env python
"""bench.py -- table/numbers, no figure (convergence-validation.md Sec. 5).

Wall-clock (and table memory footprint) for:
  - Stage 1 (deposition, deposit_cic) vs N_p; Stage 0 (push_and_sample,
    trajectory integration) vs n_steps -- these are two different functions
    now (n_steps no longer affects deposit_cic's input size at all, since
    the a0/H fix -- see git log "Fix a0/H..." -- made push_and_sample return
    one row per particle regardless of n_steps). Stage 0 additionally
    compares every backend push_and_sample supports on this machine
    (git log "particles.py: add numba/cupy backends to push_and_sample"):
    'numpy' (single-threaded, always available), 'cupy' (GPU, always
    available in this repo's environment), 'numba' (CPU multithreaded,
    only if the numba package is importable -- skipped otherwise).
  - Stage 2 (spectrum_kernel_4d, the real GPU kernel) vs number of output
    points and vs samples_per_point
  - direct binning (reference.direct_binning_spectrum) vs N_p * N_directions,
    numpy vs cupy backend -- cost baseline only; its normalisation was
    root-caused and fixed (git log "reference.py: root-cause and fix
    direct_binning_spectrum's normalization"), but a real, still-open
    ~2*pi-adjacent residual remains (see reference.py's module docstring),
    irrelevant to a timing measurement regardless. Timed with a0=0 (a
    plain Stage-0 bunch has no ahat -- that only comes from push_and_sample,
    not needed to benchmark this function's own cost).
  - direct per-electron spectral *integration* (fig_validation.py's FFT-based
    R, the Sec. 4 "prohibitive for a full spectral scan" method) vs N_p --
    stays numpy-only: fig_validation.py is deliberately CPU/RAM-bound (see
    params.RAM_GB's docstring), not part of the table-based GPU pipeline.

Every timing is wall-clock via cp.cuda.Event where the operation is a GPU
kernel launch and time.perf_counter() otherwise, averaged over a few repeats
after one untimed warm-up call (GPU kernels JIT-compile on first launch).

--quick: fewer/smaller sweep points, still exercises every code path.
"""
import argparse
import time

import numpy as np
import cupy as cp

import params as P
import refs as R
import plotstyle as PS
from xigma_i import particles, deposition, reference, spectrum4d


def timed(fn, n_repeat=3, warmup=1):
    for _ in range(warmup):
        fn()
    cp.cuda.Stream.null.synchronize()
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        fn()
        cp.cuda.Stream.null.synchronize()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def _push_and_sample_backends():
    """'numpy' and 'cupy' are always available in this repo's environment;
    'numba' only if the package is importable (push_and_sample's own lazy
    import, see particles.py) -- checked the same way here so this
    benchmark degrades gracefully on a machine without it rather than
    crashing partway through.
    """
    backends = ['numpy', 'cupy']
    try:
        import numba  # noqa: F401
        backends.append('numba')
    except ImportError:
        pass
    return backends


def bench_deposition(compton, n_p_values, n_steps_fixed, n_steps_values, n_p_fixed):
    print("[bench] Stage 1 (deposition) vs N_p ...")
    t_vs_np = []
    for n_p in n_p_values:
        bunch = R.make_bunch(compton, int(n_p))
        # backend='cupy': this call only generates test data for the timed
        # deposit_cic call below (deposit_cic itself has no backend choice --
        # xp=cp is unconditional), so running it on-device here avoids a
        # redundant host round-trip; Grid4D.from_samples accepts cupy input
        # directly.
        gamma, tx, ty, a0, w = particles.push_and_sample(compton, bunch, n_steps=n_steps_fixed, backend='cupy')
        grid = deposition.Grid4D.from_samples(gamma, tx, ty, a0, n_bins=P.DEFAULT_N_BINS)
        dt = timed(lambda: deposition.deposit_cic(grid, gamma, tx, ty, a0, w, accumulate_dtype=cp.float64, xp=cp),
                   n_repeat=3)
        t_vs_np.append(dt)
        print(f"    N_p={n_p:>10d}: {dt * 1e3:.2f} ms  ({gamma.shape[0] / dt:.3e} samples/s)")
        del gamma, tx, ty, a0, w
        cp.get_default_memory_pool().free_all_blocks()

    # push_and_sample (Stage 0), not deposit_cic, vs n_steps: since the a0/H
    # fix (git log "Fix a0/H..."), push_and_sample returns one row per
    # particle regardless of n_steps -- n_steps is purely the trajectory-
    # integration resolution used internally for each particle's ahat/L, so
    # deposit_cic's input size (and therefore its own cost) no longer depends
    # on n_steps at all. What *does* still scale with n_steps is
    # push_and_sample's own trajectory-integration cost, timed here instead,
    # per backend (see module docstring/`_push_and_sample_backends`).
    backends = _push_and_sample_backends()
    print(f"[bench] Stage 0 (push_and_sample, trajectory integration) vs n_steps, "
          f"backends={backends} ...")
    t_vs_nsteps = {}
    for backend in backends:
        t_vs_nsteps[backend] = []
        for n_steps in n_steps_values:
            bunch = R.make_bunch(compton, int(n_p_fixed))
            dt = timed(lambda: particles.push_and_sample(compton, bunch, n_steps=int(n_steps), backend=backend),
                       n_repeat=3)
            t_vs_nsteps[backend].append(dt)
            print(f"    backend={backend:<6s} n_steps={n_steps:>6d}: {dt * 1e3:.2f} ms  "
                  f"({n_p_fixed * n_steps / dt:.3e} particle-steps/s)")
        t_vs_nsteps[backend] = np.array(t_vs_nsteps[backend])

    return np.array(t_vs_np), t_vs_nsteps


def bench_stage2(compton, table, n_points_values, samples_values):
    print("[bench] Stage 2 (spectrum_kernel_4d) vs number of output points ...")
    t_vs_npts = []
    for n_pts in n_points_values:
        theta_x = cp.linspace(0.0, 1.0 / P.GAMMA0, int(n_pts), dtype=cp.float32)
        theta_y = cp.array([0.0], dtype=cp.float32)
        s = P.s_grid(32)
        dt = timed(lambda: spectrum4d.calculate_angular_spectrum_4d(
            compton, table, cp.asarray(s, dtype=cp.float32), theta_x, theta_y, P.PHI_POL,
            samples_per_point=P.DEFAULT_SAMPLES_PER_POINT), n_repeat=3)
        t_vs_npts.append(dt)
        print(f"    n_points={n_pts * s.size:>8d}: {dt * 1e3:.2f} ms")

    print("[bench] Stage 2 vs samples_per_point ...")
    t_vs_samples = []
    theta_x = cp.linspace(0.0, 1.0 / P.GAMMA0, 32, dtype=cp.float32)
    theta_y = cp.array([0.0], dtype=cp.float32)
    s = P.s_grid(32)
    for spp in samples_values:
        dt = timed(lambda: spectrum4d.calculate_angular_spectrum_4d(
            compton, table, cp.asarray(s, dtype=cp.float32), theta_x, theta_y, P.PHI_POL,
            samples_per_point=int(spp)), n_repeat=3)
        t_vs_samples.append(dt)
        print(f"    samples_per_point={spp:>6d}: {dt * 1e3:.2f} ms")

    return np.array(t_vs_npts), np.array(t_vs_samples)


def bench_direct_binning(compton, n_p_values, n_directions):
    print("[bench] direct binning (cost baseline, normalisation now fixed -- see module docstring "
          "for the still-open ~2*pi residual) vs N_p * N_directions, numpy vs cupy ...")
    s_edges = P.s_grid(64)
    directions = [P.OBS_POINTS["on_axis"], P.OBS_POINTS["mid"]][:max(1, n_directions)]
    backends = ['numpy', 'cupy']
    times = {b: [] for b in backends}
    for n_p in n_p_values:
        bunch = R.make_bunch(compton, int(n_p))
        weight = np.full(bunch.n_particles, bunch.weight)
        # a0=0: a plain Stage-0 bunch (R.make_bunch) has no ahat -- that's a
        # push_and_sample output, not needed here since this only benchmarks
        # direct_binning_spectrum's own cost, not correctness (see docstring).
        a0 = np.zeros(bunch.n_particles)

        for backend in backends:
            def run():
                for (x0, y0) in directions:
                    reference.direct_binning_spectrum(bunch.gamma, bunch.theta_x, bunch.theta_y,
                                                        weight, a0, x0, y0, s_edges, P.PHI_POL, backend=backend)
            t0 = time.perf_counter()
            run()
            dt = time.perf_counter() - t0
            times[backend].append(dt)
            print(f"    backend={backend:<6s} N_p={n_p:>10d} x {len(directions)} directions: {dt * 1e3:.2f} ms")
    return {b: np.array(v) for b, v in times.items()}


def bench_spectral_integration(compton, n_p_values):
    print("[bench] direct spectral integration (Fig. 4's FFT-based R) vs N_p ...")
    import fig_validation as FV
    times = []
    for n_p in n_p_values:
        bunch = R.make_bunch(compton, int(n_p))

        def run():
            t, field = FV.sample_common_window_field(compton, bunch, 512)
            FV.per_particle_R(t, field)
        t0 = time.perf_counter()
        run()
        dt = time.perf_counter() - t0
        times.append(dt)
        print(f"    N_p={n_p:>10d}: {dt * 1e3:.2f} ms")
    return np.array(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--rebuild-cache", action="store_true",
                    help="ignore any cached table under data/cache/ and rebuild+overwrite it "
                         "(see refs.cached_table's docstring). Equivalent to VALIDATION_REBUILD_CACHE=1.")
    args = ap.parse_args()

    compton = P.build_compton()

    if args.quick:
        dep_np_values = np.array([2_000, 20_000, 200_000])
        dep_n_steps_values = np.array([8, 32, 128])
        dep_n_steps_fixed, dep_n_p_fixed = 32, 40_000
        stage2_npts_values = np.array([8, 32, 128])
        stage2_samples_values = np.array([4, 16, 64])
        db_np_values = np.array([2_000, 20_000])
        si_np_values = np.array([500, 2_000])
        table_n_bins = (64, 64, 64, 16)
        table_n_p, table_n_steps = 40_000, 32
    else:
        print(f"[bench] hardware tier: {P.SCALE_TIER} (VALIDATION_SCALE to override)")
        # bench_deposition/bench_direct_binning time a single un-chunked call
        # each (raw kernel/path throughput, not correctness at arbitrary N_p),
        # so N_p here comes from params.BENCH_NP_VALUES, capped independently
        # of the (streamed, unbounded) deposition-scheme scan in fig_deposition.py.
        dep_np_values = np.array(P.BENCH_NP_VALUES)
        dep_n_steps_values = np.array([8, 16, 32, 64, 128])
        dep_n_steps_fixed, dep_n_p_fixed = P.DEFAULT_N_STEPS, min(P.DEFAULT_N_PARTICLES, 200_000)
        stage2_npts_values = np.array([8, 32, 128, 512])
        stage2_samples_values = np.array([4, 16, 64, 256])
        db_np_values = dep_np_values
        si_np_values = np.array([2_000, 10_000, 20_000])
        table_n_bins = P.DEFAULT_N_BINS
        table_n_p, table_n_steps = P.DEFAULT_N_PARTICLES, P.DEFAULT_N_STEPS

    t_dep_np, t_dep_nsteps_by_backend = bench_deposition(compton, dep_np_values, dep_n_steps_fixed,
                                                          dep_n_steps_values, dep_n_p_fixed)

    print(f"[bench] building a table for the Stage-2 benchmark "
          f"(N_p={table_n_p}, n_steps={table_n_steps}, n_bins={table_n_bins}) ...")
    table = R.cached_table(compton, table_n_p, table_n_steps, n_bins=table_n_bins, force=args.rebuild_cache)

    t_stage2_npts, t_stage2_samples = bench_stage2(compton, table, stage2_npts_values, stage2_samples_values)
    t_direct_binning_by_backend = bench_direct_binning(compton, db_np_values, n_directions=2)
    t_spectral_integration = bench_spectral_integration(compton, si_np_values)

    footprint_bytes = int(np.prod(P.DEFAULT_N_BINS)) * 4  # float32
    footprint_fine_bytes = int(np.prod(P.FINE_N_BINS)) * 4
    print(f"[bench] table footprint (float32): DEFAULT_N_BINS={P.DEFAULT_N_BINS} -> "
          f"{footprint_bytes / 1e6:.1f} MB; FINE_N_BINS={P.FINE_N_BINS} -> {footprint_fine_bytes / 1e6:.1f} MB")

    PS.DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        PS.DATA_DIR / "bench.npz",
        dep_np_values=dep_np_values, t_dep_np=t_dep_np,
        dep_n_steps_values=dep_n_steps_values,
        **{f"t_dep_nsteps_{backend}": arr for backend, arr in t_dep_nsteps_by_backend.items()},
        stage2_npts_values=stage2_npts_values, t_stage2_npts=t_stage2_npts,
        stage2_samples_values=stage2_samples_values, t_stage2_samples=t_stage2_samples,
        db_np_values=db_np_values,
        **{f"t_direct_binning_{backend}": arr for backend, arr in t_direct_binning_by_backend.items()},
        si_np_values=si_np_values, t_spectral_integration=t_spectral_integration,
        footprint_bytes=footprint_bytes, footprint_fine_bytes=footprint_fine_bytes,
        table_n_bins=table_n_bins, table_n_p=table_n_p, table_n_steps=table_n_steps,
        quick=args.quick,
    )
    print(f"[bench] wrote {PS.DATA_DIR / 'bench.npz'}")


if __name__ == "__main__":
    main()
