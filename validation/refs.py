"""Shared bunch/table construction plus the reference paths used across
figures (plan.md Sec. 0).

Two distinct references, per the plan -- never conflated:

  - `ref_quadrature`: `reference.spectrum_from_table` (a brute-force Riemann
    sum over an H table's own cells -- no sampling/QMC noise at all) applied
    to a "very high sample budget / fine grid" table (`get_fine_table`).
    Reference for Figs. 1-3, i.e. for testing the *numerical* convergence of
    the quadrature/table used by the production pipeline, at a fixed physical
    case.
  - `ref_direct_binning`: `reference.direct_binning_spectrum`, iterating real
    macroparticles with no table and no quadrature grid at all. Used here
    only as a cost baseline (bench.py) -- its absolute normalisation has an
    open, unresolved bug (see reference.py's module docstring, "WHAT'S STILL
    OPEN"), so it is not used as a correctness reference in any figure here.

A third, `ref_spectral_integration` (full per-electron resonance function,
no delta substitution), is Fig. 4 only and lives in fig_validation.py itself
since it is not reused elsewhere and needs the paper's Eq. (Rdef)/(wR)/
(ahattraj), not anything already in xigma_i.

Tables are cached to `data/cache/` keyed by every parameter that affects
them, since they are the expensive, reused artifact (plan.md Sec. 0: "Cache
all reference runs to disk").
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import cupy as cp

from xigma_i import particles, deposition, reference

import params as P

CACHE_DIR = Path(__file__).resolve().parent / "data" / "cache"


def _cache_key(kind, **kwargs):
    blob = json.dumps(kwargs, sort_keys=True, default=str)
    h = hashlib.sha1(blob.encode()).hexdigest()[:16]
    return CACHE_DIR / f"{kind}_{h}.npz"


def make_bunch(compton, n_particles, *, gamma0=P.GAMMA0, sigma_gamma0=P.SIGMA_GAMMA0,
               chirp=0.0, angle_energy_corr=0.0, seed=P.DEFAULT_SEED):
    rng = np.random.default_rng(seed)
    return particles.sample_bunch(compton, n_particles, gamma0, sigma_gamma0,
                                   chirp=chirp, angle_energy_corr=angle_energy_corr, rng=rng)


def make_samples(compton, n_particles, n_steps, *, gamma0=P.GAMMA0, sigma_gamma0=P.SIGMA_GAMMA0,
                  chirp=0.0, angle_energy_corr=0.0, seed=P.DEFAULT_SEED):
    """Stage 0+ballistic-push samples materialised all at once (host numpy
    arrays), for scripts that need to deposit the *same* particle set into
    several different grids (Fig. 2's grid-resolution scan; Fig. 3a/b's
    deposition-scheme/timestep scans) -- use build_table_streaming instead
    when the point is a single, very large table and per-particle reuse
    across grids doesn't matter.
    """
    bunch = make_bunch(compton, n_particles, gamma0=gamma0, sigma_gamma0=sigma_gamma0,
                        chirp=chirp, angle_energy_corr=angle_energy_corr, seed=seed)
    return particles.push_and_sample(compton, bunch, n_steps=n_steps)


def build_table_streaming(compton, n_particles, n_steps, *, n_bins=P.DEFAULT_N_BINS,
                           scheme=P.DEFAULT_DEPOSITION_SCHEME, chirp=0.0, angle_energy_corr=0.0,
                           seed=P.DEFAULT_SEED, chunk_particles=P.STREAM_CHUNK_PARTICLES,
                           grid=None, device="gpu", gamma0=P.GAMMA0, sigma_gamma0=P.SIGMA_GAMMA0,
                           quiet=True):
    """Stage 0+1 for n_particles*n_steps samples, without ever materialising
    them all at once: draws and pushes `chunk_particles` at a time, deriving
    the grid from the first chunk if not supplied (so every chunk deposits
    into the same fixed grid), and accumulates.
    """
    rng = np.random.default_rng(seed)
    n_done = 0
    H_total = None
    occ_total = None
    n_discarded_total = 0
    n_samples_total = 0

    t0 = time.time()
    while n_done < n_particles:
        n_chunk = min(chunk_particles, n_particles - n_done)
        bunch = particles.sample_bunch(compton, n_chunk, gamma0, sigma_gamma0,
                                        chirp=chirp, angle_energy_corr=angle_energy_corr, rng=rng)
        # sample_bunch sets bunch.weight = N_e / n_chunk, i.e. as if this chunk
        # alone were the whole population; rescale to N_e / n_particles (the
        # correct per-macroparticle weight for a fraction of a streamed bunch)
        # or the total deposited weight inflates by n_particles/chunk_particles.
        bunch.weight *= n_chunk / n_particles
        gamma, tx, ty, a0, w = particles.push_and_sample(compton, bunch, n_steps=n_steps)

        if grid is None:
            grid = deposition.Grid4D.from_samples(gamma, tx, ty, a0, n_bins=n_bins)

        deposit_fn = deposition._DEPOSIT_FUNCS[scheme]
        gamma_g, tx_g, ty_g, a0_g, w_g = (cp.asarray(x) for x in (gamma, tx, ty, a0, w))
        H_chunk, occ_chunk, n_disc = deposit_fn(grid, gamma_g, tx_g, ty_g, a0_g, w_g,
                                                 accumulate_dtype=cp.float64, xp=cp)
        if H_total is None:
            H_total = H_chunk
            occ_total = occ_chunk
        else:
            H_total += H_chunk
            occ_total += occ_chunk
        n_discarded_total += n_disc
        n_samples_total += gamma.shape[0]
        n_done += n_chunk
        # H_chunk/occ_chunk and the per-chunk cupy inputs are now dead; drop the
        # pool's hold on them explicitly rather than trusting GC timing -- a 6 GB
        # card has no headroom for a large accumulator plus several stale chunks.
        del gamma_g, tx_g, ty_g, a0_g, w_g, H_chunk, occ_chunk
        cp.get_default_memory_pool().free_all_blocks()
        if not quiet:
            print(f"  ... {n_done}/{n_particles} particles ({time.time() - t0:.1f}s)", flush=True)

    H_raw = H_total.get()
    occupancy = occ_total.get()
    H_density = H_raw / grid.bin_volume
    bracket = deposition.gamma_bracket(H_density, grid, q=1e-4)

    return deposition.Table(
        H=H_density, grid=grid, scheme=scheme, n_particle_samples=n_samples_total,
        total_weight=float(H_raw.sum()), n_discarded=n_discarded_total, occupancy=occupancy,
        gamma_bracket=bracket,
    )


def deposit_in_chunks(grid, gamma, tx, ty, a0, w, scheme=P.DEFAULT_DEPOSITION_SCHEME,
                       chunk_size=P.STREAM_CHUNK_PARTICLES, accumulate_dtype=cp.float64):
    """Deposit an already-materialised (host numpy) sample set into a fixed
    `grid`, chunked over the sample axis so peak GPU memory is bounded by
    (grid size + one chunk), not by len(gamma). Used by fig_gridres.py/
    fig_deposition.py, which redeposit the *same* large sample set into
    several grid variants (so the memory cost cannot be sidestepped by
    build_table_streaming's usual approach of never materialising the full
    sample set at all -- here it already exists as one array by construction,
    only the *deposition* needs chunking). A single un-chunked
    `deposition.deposit_cic` call over a large sample array creates several
    same-sized float64 temporaries per corner internally and OOM'd on this
    project's 6 GB dev GPU well before the grid accumulator itself was the
    limiting factor -- this is the general fix, not just a smaller N_p.
    """
    deposit_fn = deposition._DEPOSIT_FUNCS[scheme]
    n = gamma.shape[0]
    H_total = None
    occ_total = None
    n_discarded_total = 0
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        args = [cp.asarray(x[start:end]) for x in (gamma, tx, ty, a0, w)]
        H_chunk, occ_chunk, n_disc = deposit_fn(grid, *args, accumulate_dtype=accumulate_dtype, xp=cp)
        if H_total is None:
            H_total, occ_total = H_chunk, occ_chunk
        else:
            H_total += H_chunk
            occ_total += occ_chunk
        n_discarded_total += n_disc
        del args, H_chunk, occ_chunk
        cp.get_default_memory_pool().free_all_blocks()
    return H_total, occ_total, n_discarded_total


def _rebuild_forced():
    return os.environ.get("VALIDATION_REBUILD_CACHE", "").strip().lower() in ("1", "true", "yes")


def cached_table(compton, n_particles, n_steps, force=False, **kwargs):
    """force=True (or the VALIDATION_REBUILD_CACHE env var, which every
    script's --rebuild-cache flag sets) skips the cache-hit check and always
    rebuilds, then overwrites the cache file with the fresh result.

    Needed when running across multiple machines kept in sync by a plain
    file-sync tool rather than git: the cache key is a hash of *parameters*
    (n_particles, n_steps, n_bins, scheme, ...), not of the code that
    builds the table, so a cache entry from before a bug fix in
    build_table_streaming/deposition.py is still a "hit" by that key and
    would otherwise be silently reused unchanged. A half-synced/partially
    written npz is a different failure mode (Table.load raises on it) and
    is not what this flag is for -- delete the file (or the whole
    data/cache/ directory) if that happens instead.
    """
    key_kwargs = dict(n_particles=n_particles, n_steps=n_steps, **kwargs)
    path = _cache_key("table", **key_kwargs)
    if path.exists() and not (force or _rebuild_forced()):
        return deposition.Table.load(path)
    table = build_table_streaming(compton, n_particles, n_steps, **kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.save(path)
    return table


def get_fine_table(compton, **overrides):
    """The ground-truth table for Figs. 1-3 -- see module docstring."""
    kwargs = dict(n_bins=P.FINE_N_BINS, scheme="cic", chunk_particles=P.STREAM_CHUNK_PARTICLES)
    kwargs.update(overrides)
    return cached_table(compton, P.FINE_N_PARTICLES, P.FINE_N_STEPS, **kwargs)


def get_default_table(compton, **overrides):
    """The "production" table Figs. 1-3 hold fixed while scanning one axis
    (M for Fig. 1; the scanned axis's own resolution/N_p/dt for Figs. 2-3
    is passed as an override, not this function, so callers should not
    just call this with defaults inside a sweep loop for the *scanned*
    parameter).
    """
    kwargs = dict(n_bins=P.DEFAULT_N_BINS, scheme=P.DEFAULT_DEPOSITION_SCHEME,
                   chunk_particles=P.STREAM_CHUNK_PARTICLES)
    kwargs.update(overrides)
    return cached_table(compton, P.DEFAULT_N_PARTICLES, P.DEFAULT_N_STEPS, **kwargs)


def ref_quadrature(table, compton, x0, y0, s):
    """reference.spectrum_from_table on `table` -- an exact grid quadrature,
    no sampling noise. See module docstring for which table to pass.
    """
    return reference.spectrum_from_table(table, compton, x0, y0, s, P.PHI_POL)


def ref_direct_binning(gamma, theta_x, theta_y, weight, compton, x0, y0, s_edges):
    """Cost-baseline only -- see module docstring; do not trust absolute
    values (reference.py: unresolved ~3000-4000x normalisation gap).
    """
    return reference.direct_binning_spectrum(gamma, theta_x, theta_y, weight, compton,
                                              x0, y0, s_edges, P.PHI_POL)
