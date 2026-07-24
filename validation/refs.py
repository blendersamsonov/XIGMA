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
    only as a cost baseline (bench.py) -- its normalisation was root-caused
    and fixed (git log "reference.py: root-cause and fix direct_binning_
    spectrum's normalization"), but a real, still-open ~2*pi-adjacent
    residual remains in its angle-integrated total (see reference.py's
    module docstring), so it is still not used as a correctness reference
    in any figure here.

A third, `ref_spectral_integration` (full per-electron resonance function,
no delta substitution), is Fig. 4 only and lives in fig_validation.py itself
since it is not reused elsewhere and needs the paper's Eq. (Rdef)/(wR)/
(ahattraj), not anything already in xigma_i.

Tables are cached to `data/cache/` keyed by every parameter that affects
them, since they are the expensive, reused artifact (plan.md Sec. 0: "Cache
all reference runs to disk").

GPU usage: table-building (build_table_streaming) and the quadrature
reference (ref_quadrature) run backend='cupy' -- the underlying
xigma_i functions gained numpy/cupy(/numba, for push_and_sample) backends
this session (git log "particles.py: add numba/cupy backends to
push_and_sample", "reference.py: add cupy backend to the spectrum
calculation functions"), default stays backend='numpy' upstream so this
module opts in explicitly where it's safe. `make_samples` is the one
exception -- see its docstring for why it deliberately stays on CPU.

Batching: this module's own chunked table-building was promoted to core
xigma_i this session (git log "deposition.py: add build_table_streaming, a
Stage 0+1 combined chunked pipeline") -- `build_table_streaming` here is
now a thin wrapper around `deposition.build_table_streaming`. The other
half of this module's batching (redepositing an *already-materialised*
sample set into several grid variants, needed by fig_gridres.py/
fig_deposition.py) turned out to already exist in core as
`deposition.build_table`'s own `batch_size` parameter -- that was never
duplicated here to begin with, it was just unused; callers now use it
directly instead of a validation-only `deposit_in_chunks` that no longer
exists.
"""
import hashlib
import json
import os
import sys
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

    Returns one row per particle (particles.push_and_sample's contract since
    the a0/H fix -- see git log "Fix a0/H..."; n_steps sets the trajectory-
    integration resolution for the per-particle ahat/L it returns, not the
    output array length). The returned arrays are therefore always
    O(n_particles), not O(n_particles*n_steps) -- but push_and_sample's own
    *internal* computation is still O(n_particles*n_steps) transiently (the
    per-timestep arrays it integrates over before returning), so a very large
    product can still be memory-heavy to call this with directly even though
    what comes back is small; build_table_streaming's chunking exists for
    that internal cost, not because the returned arrays themselves are large.
    """
    bunch = make_bunch(compton, n_particles, gamma0=gamma0, sigma_gamma0=sigma_gamma0,
                        chirp=chirp, angle_energy_corr=angle_energy_corr, seed=seed)
    # Deliberately backend='numpy' (the default), not 'cupy', despite this whole
    # module otherwise preferring GPU where it's safe: this call is a single,
    # unchunked push_and_sample of the *entire* n_particles set (that's the
    # point -- callers need one shared sample set to redeposit into several
    # grids), and push_and_sample's internal trajectory-integration arrays are
    # O(n_particles*n_steps) regardless of backend (see this function's own
    # docstring above) -- for the particle counts this is called with (Fig. 2's
    # GRIDRES_N_PARTICLES, up to millions), that product can exceed GPU memory
    # even though it fits comfortably in system RAM. build_table_streaming
    # below runs the same function on GPU precisely because it chunks by
    # STREAM_CHUNK_PARTICLES first, sized to bound exactly this cost.
    return particles.push_and_sample(compton, bunch, n_steps=n_steps)


def build_table_streaming(compton, n_particles, n_steps, *, n_bins=P.DEFAULT_N_BINS,
                           scheme=P.DEFAULT_DEPOSITION_SCHEME, chirp=0.0, angle_energy_corr=0.0,
                           seed=P.DEFAULT_SEED, chunk_particles=P.STREAM_CHUNK_PARTICLES,
                           grid=None, device="gpu", gamma0=P.GAMMA0, sigma_gamma0=P.SIGMA_GAMMA0,
                           quiet=True):
    """Thin wrapper around deposition.build_table_streaming -- promoted to
    core this session (git log "deposition.py: add build_table_streaming,
    a Stage 0+1 combined chunked pipeline"; this validation-only copy is
    what that core function was extracted from, see its own docstring for
    the full Stage 0+1 chunking rationale). Applies this module's own
    parameter defaults and always runs push_backend='cupy': chunk_particles
    is exactly the size params.py's STREAM_CHUNK_PARTICLES tier setting
    picks so one chunk's internal arrays fit in GPU memory, so running the
    push itself on-device is safe by the same reasoning that already
    justified chunking (see params.py's hardware-tier comment).
    """
    rng = np.random.default_rng(seed)
    return deposition.build_table_streaming(
        compton, n_particles, n_steps, chunk_particles=chunk_particles,
        gamma0=gamma0, sigma_gamma0=sigma_gamma0, chirp=chirp, angle_energy_corr=angle_energy_corr,
        rng=rng, push_backend='cupy', grid=grid, scheme=scheme, device=device,
        n_bins=n_bins, quiet=quiet,
    )


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

    backend='cupy': this is called repeatedly, often inside the M/N_p/bin-
    count sweep loops in fig_sampling.py/fig_gridres.py/fig_deposition.py,
    against tables as large as FINE_N_BINS -- table.H is transferred to the
    GPU once inside spectrum_from_table and reused across its (small) loop
    over s, rather than re-transferred per call the way a naive per-call
    cp.asarray would (see reference.py's module docstring on this function's
    backend support).
    """
    return reference.spectrum_from_table(table, compton, x0, y0, s, P.PHI_POL, backend='cupy')


def ref_direct_binning(gamma, theta_x, theta_y, weight, a0, x0, y0, s_edges, backend='cupy'):
    """Cost-baseline only -- see module docstring. Its normalisation was
    fixed (git log "reference.py: root-cause and fix direct_binning_
    spectrum's normalization"), but a real, still-open ~2*pi-adjacent
    residual remains in its angle-integrated total (see reference.py's
    module docstring) -- do not use this as a correctness reference in any
    figure here; only bench.py's timing baseline relies on it.

    a0 is now a required argument (one value per particle, e.g.
    push_and_sample's ahat output) -- there is no longer a `compton`
    parameter (dropped upstream, see the commit above).
    """
    return reference.direct_binning_spectrum(gamma, theta_x, theta_y, weight, a0,
                                              x0, y0, s_edges, P.PHI_POL, backend=backend)
