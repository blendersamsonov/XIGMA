#!/usr/bin/env python
"""bench.py -- table/numbers, no figure (convergence-validation.md Sec. 5).

Wall-clock (and table memory footprint) for:
  - Stage 1 (deposition) vs N_p and vs n_steps
  - Stage 2 (spectrum_kernel_4d, the real GPU kernel) vs number of output
    points and vs samples_per_point
  - direct binning (reference.direct_binning_spectrum) vs N_p * N_directions
    -- cost baseline only; see refs.py/reference.py for its open
    normalisation-accuracy caveat, irrelevant to a timing measurement
  - direct per-electron spectral *integration* (fig_validation.py's FFT-based
    R, the Sec. 4 "prohibitive for a full spectral scan" method) vs N_p

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


def bench_deposition(compton, n_p_values, n_steps_fixed, n_steps_values, n_p_fixed):
    print("[bench] Stage 1 (deposition) vs N_p ...")
    t_vs_np = []
    for n_p in n_p_values:
        bunch = R.make_bunch(compton, int(n_p))
        gamma, tx, ty, a0, w = particles.push_and_sample(compton, bunch, n_steps=n_steps_fixed)
        grid = deposition.Grid4D.from_samples(gamma, tx, ty, a0, n_bins=P.DEFAULT_N_BINS)
        args = [cp.asarray(x) for x in (gamma, tx, ty, a0, w)]
        dt = timed(lambda: deposition.deposit_cic(grid, *args, accumulate_dtype=cp.float64, xp=cp), n_repeat=3)
        t_vs_np.append(dt)
        print(f"    N_p={n_p:>10d}: {dt * 1e3:.2f} ms  ({gamma.shape[0] / dt:.3e} samples/s)")
        del args
        cp.get_default_memory_pool().free_all_blocks()

    print("[bench] Stage 1 (deposition) vs n_steps ...")
    t_vs_nsteps = []
    for n_steps in n_steps_values:
        bunch = R.make_bunch(compton, int(n_p_fixed))
        gamma, tx, ty, a0, w = particles.push_and_sample(compton, bunch, n_steps=int(n_steps))
        grid = deposition.Grid4D.from_samples(gamma, tx, ty, a0, n_bins=P.DEFAULT_N_BINS)
        args = [cp.asarray(x) for x in (gamma, tx, ty, a0, w)]
        dt = timed(lambda: deposition.deposit_cic(grid, *args, accumulate_dtype=cp.float64, xp=cp), n_repeat=3)
        t_vs_nsteps.append(dt)
        print(f"    n_steps={n_steps:>6d}: {dt * 1e3:.2f} ms  ({gamma.shape[0] / dt:.3e} samples/s)")
        del args
        cp.get_default_memory_pool().free_all_blocks()

    return np.array(t_vs_np), np.array(t_vs_nsteps)


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
    print("[bench] direct binning (cost baseline) vs N_p * N_directions ...")
    times = []
    s_edges = P.s_grid(64)
    directions = [P.OBS_POINTS["on_axis"], P.OBS_POINTS["mid"]][:max(1, n_directions)]
    for n_p in n_p_values:
        bunch = R.make_bunch(compton, int(n_p))

        def run():
            for (x0, y0) in directions:
                reference.direct_binning_spectrum(bunch.gamma, bunch.theta_x, bunch.theta_y,
                                                    np.full(bunch.n_particles, bunch.weight),
                                                    compton, x0, y0, s_edges, P.PHI_POL)
        t0 = time.perf_counter()
        run()
        dt = time.perf_counter() - t0
        times.append(dt)
        print(f"    N_p={n_p:>10d} x {len(directions)} directions: {dt * 1e3:.2f} ms")
    return np.array(times)


def bench_spectral_integration(compton, n_p_values):
    print("[bench] direct spectral integration (Fig. 4's FFT-based R) vs N_p ...")
    import fig_validation as FV
    times = []
    for n_p in n_p_values:
        bunch = R.make_bunch(compton, int(n_p))

        def run():
            t, a0_local = FV.sample_common_window_envelope(compton, bunch, 512)
            FV.per_particle_R(t, a0_local)
        t0 = time.perf_counter()
        run()
        dt = time.perf_counter() - t0
        times.append(dt)
        print(f"    N_p={n_p:>10d}: {dt * 1e3:.2f} ms")
    return np.array(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
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

    t_dep_np, t_dep_nsteps = bench_deposition(compton, dep_np_values, dep_n_steps_fixed,
                                               dep_n_steps_values, dep_n_p_fixed)

    print(f"[bench] building a table for the Stage-2 benchmark "
          f"(N_p={table_n_p}, n_steps={table_n_steps}, n_bins={table_n_bins}) ...")
    table = R.cached_table(compton, table_n_p, table_n_steps, n_bins=table_n_bins)

    t_stage2_npts, t_stage2_samples = bench_stage2(compton, table, stage2_npts_values, stage2_samples_values)
    t_direct_binning = bench_direct_binning(compton, db_np_values, n_directions=2)
    t_spectral_integration = bench_spectral_integration(compton, si_np_values)

    footprint_bytes = int(np.prod(P.DEFAULT_N_BINS)) * 4  # float32
    footprint_fine_bytes = int(np.prod(P.FINE_N_BINS)) * 4
    print(f"[bench] table footprint (float32): DEFAULT_N_BINS={P.DEFAULT_N_BINS} -> "
          f"{footprint_bytes / 1e6:.1f} MB; FINE_N_BINS={P.FINE_N_BINS} -> {footprint_fine_bytes / 1e6:.1f} MB")

    PS.DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        PS.DATA_DIR / "bench.npz",
        dep_np_values=dep_np_values, t_dep_np=t_dep_np,
        dep_n_steps_values=dep_n_steps_values, t_dep_nsteps=t_dep_nsteps,
        stage2_npts_values=stage2_npts_values, t_stage2_npts=t_stage2_npts,
        stage2_samples_values=stage2_samples_values, t_stage2_samples=t_stage2_samples,
        db_np_values=db_np_values, t_direct_binning=t_direct_binning,
        si_np_values=si_np_values, t_spectral_integration=t_spectral_integration,
        footprint_bytes=footprint_bytes, footprint_fine_bytes=footprint_fine_bytes,
        table_n_bins=table_n_bins, table_n_p=table_n_p, table_n_steps=table_n_steps,
        quick=args.quick,
    )
    print(f"[bench] wrote {PS.DATA_DIR / 'bench.npz'}")


if __name__ == "__main__":
    main()
