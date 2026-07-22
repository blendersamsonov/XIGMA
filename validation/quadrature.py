"""CPU box-integral Monte-Carlo/QMC harness, used only by fig_sampling.py.

Design note (read before reusing this elsewhere): `spectrum_kernel_4d`'s
actual sample-count knob (`samples_per_point`) only spans about 2-3 decades
before cost becomes impractical (`SAMPLES_TOTAL=256` is a compile-time
constant; `samples_per_point` multiplies it via a subsampling loop), and the
kernel hard-codes Fibonacci/regular-CDF sampling -- there is no
pseudorandom variant to compare against without a second rawkernel, which
CLAUDE.md flags as fragile (shared-memory aliasing) and not something to
edit for a convergence experiment. Fig. 1 asks for "~5 decades-ish" of M and
an equal-budget Fibonacci-vs-pseudorandom comparison, which this module gets
instead by integrating the *same physical integrand* (H-table quadrilinear
interpolation, Eq. xsec's angle/polarisation prefactor -- exactly
`reference.spectrum_from_table`'s integrand) as a plain 2D Monte-Carlo/QMC
average over the table's (theta_x, theta_y) rectangle, so swapping the point
generator is a one-line change and M is a free parameter. This is not
`spectrum_kernel_4d`'s own annulus/arc/inverse-CDF importance-sampling
machinery -- it is a from-scratch estimator of the same integral, used
because it isolates the QMC-vs-random question cleanly. Flagged here so
nobody mistakes fig_sampling.py's curves for direct kernel instrumentation.

a0 quadrature (git log "spectrum_kernel_4d: add missing a0 dependence to the
resonance condition"): `_integrand` now sums over the table's own a0 bins
with each bin's *own* resonant gamma, `g_sq = (1+a0)/(1/s - r_sq)`, and its
own `1/(1+a0)` Jacobian factor -- matching `reference.spectrum_from_table`'s
fix (same commit). An earlier version of this module instead pre-summed
`table.H` over its entire a0 axis once (`_a0_marginal`) and interpolated that
single marginal at one shared g, ~30x cheaper than looping interp4d once per
a0 bin -- valid only because the old (buggy) resonance condition didn't
depend on a0 at all, so the gamma-axis interpolation weights were the same
for every a0 bin and commuted with the sum. That commutation no longer holds
now that g depends on a0 (each a0 bin's gamma-axis interpolation samples a
different gi0/gw), so this module now loops over a0 bins explicitly, same as
spectrum_kernel_4d's own nested a0 quadrature. This makes fig_sampling.py's
sweep ~n_a0 times slower than before the fix; no cheaper equivalent exists
until/unless someone finds one.

Importance sampling (`importance_quadrature_spectrum`, added alongside the
above): fig_sampling.py wants an *importance-sampled* Fibonacci-vs-hash
comparison too, not just the no-importance-sampling one `box_quadrature_
spectrum` gives. spectrum_kernel_4d's own coarse importance-sampling
proposal is, post the a0-dependence fix (spectrum4d.py's module docstring):
`H_marginal`, the (theta_x, theta_y) marginal of H summed over gamma and
a0, looked up nearest-cell (no interpolation) -- deliberately ignoring the
resonance condition/s entirely at this stage (unbiased regardless of what
density built the proposal, as long as the final evaluation reweights
against it exactly, which it does). `importance_quadrature_spectrum` reuses
that same H_marginal target density (see `_theta_marginal`) so the two
"importance sampling" curves in Fig. 1 are genuinely comparing the same
importance density -- only the point-generation scheme differs.

A first version of this sampler additionally restricted (theta_x, theta_y)
to the resonant annulus around (x0, y0) (per-s, matching spectrum_kernel_4d's
own rmin/rmax) before importance-sampling *within* it via a separately
constructed polar (r, phi) grid. That broke badly on this codebase's actual
tables: OBS_POINTS only ever probes theta_y=0, so the deposited theta_y
extent ends up ~100x narrower than theta_x, and a circle at the resonant
radius then intersects the box through a sliver of phi far thinner than any
practical phi-bin width -- every bin centre could miss the populated region
entirely, and did (caught via an M-independent ~40% low bias against
box_quadrature_spectrum/ref_quadrature, i.e. a real coverage bug, not
sampling noise -- increasing M did not fix it). `_theta_grid_cdf`'s
docstring has the full story. The fix: sample directly from the table's own
(theta_x, theta_y) grid cells (already shaped to the box's actual aspect
ratio, whatever it is) instead of a fresh polar grid, and drop the annulus
restriction entirely -- not needed for correctness (the a0-aware
`_integrand` still zeroes out non-resonant samples), and it exactly matches
what the kernel's own coarse proposal does (the ring/arc radial restriction
is a separate, kernel-only efficiency feature, not required for an unbiased
estimator).

Each of the M samples is drawn in two steps: a cell index via inverse-CDF,
then a sub-cell (theta_x, theta_y) position within that cell -- both driven
by either a golden-ratio Kronecker sequence ("fibonacci", matching what the
compiled kernel does for its own analogous two-step draw) or a stateless
GPU-friendly integer hash ("hash", imitating classical/white-noise Monte
Carlo -- see `_hash_u32`).
"""
import numpy as np

from xigma_i import reference

GOLDEN_PHI = 1.618033988749894848


def _interp3d(H3d, grid, gamma, theta_x, theta_y):
    """Trilinear interpolation of a precomputed (gamma, theta_x, theta_y)
    array -- reference.interp4d's algorithm with the a0 axis dropped.
    """
    axes_edges = (grid.gamma_edges, grid.theta_x_edges, grid.theta_y_edges)
    coords = (gamma, theta_x, theta_y)
    shape = H3d.shape

    out_shape = np.broadcast(*coords).shape
    coords = [np.broadcast_to(c, out_shape).astype(np.float64) for c in coords]

    f = np.zeros(out_shape, dtype=np.float64)
    in_range = np.ones(out_shape, dtype=bool)

    i0s, ws = [], []
    for edges, x, n in zip(axes_edges, coords, shape):
        width = edges[1] - edges[0]
        centers0 = edges[0] + 0.5 * width
        f_idx = (x - centers0) / width
        i0 = np.floor(f_idx).astype(np.int64)
        w = f_idx - i0
        in_range &= (i0 >= -1) & (i0 < n)
        i0s.append(i0)
        ws.append(w)

    for dg in (0, 1):
        for dtx in (0, 1):
            for dty in (0, 1):
                idxs = []
                corner_w = np.ones(out_shape, dtype=np.float64)
                valid = np.ones(out_shape, dtype=bool)
                for (i0, w, d, n) in zip(i0s, ws, (dg, dtx, dty), shape):
                    ci = i0 + d
                    valid &= (ci >= 0) & (ci < n)
                    corner_w = corner_w * (w if d else (1 - w))
                    idxs.append(np.clip(ci, 0, n - 1))
                take = valid & in_range
                if np.any(take):
                    vals = H3d[idxs[0][take], idxs[1][take], idxs[2][take]]
                    f[take] += vals * corner_w[take]

    f[~in_range] = 0.0
    return f


def _integrand(table, compton, x0, y0, s, theta_x, theta_y, phi_pol):
    """dN/(ds dOmega), quadrature-summed (not importance-sampled) over the
    table's own a0 bins, each with its own resonant gamma and Jacobian --
    same convention as reference.spectrum_from_table/spectrum_kernel_4d (see
    module docstring for why this can no longer be done via a precomputed
    a0-marginal). theta_x, theta_y, s broadcast against each other (e.g.
    shapes (M, 1) and (1, n_s) -> (M, n_s)).
    """
    theta_x, theta_y, s = np.broadcast_arrays(theta_x, theta_y, s)
    r_sq = (theta_x - x0) ** 2 + (theta_y - y0) ** 2
    cos_pol = np.cos(phi_pol - np.arctan2(theta_y - y0, theta_x - x0)) ** 2

    grid = table.grid
    a0_centers = grid.centers[3]
    da = grid.widths[3]

    with np.errstate(divide="ignore", invalid="ignore"):
        inv_base = 1.0 / np.where(s > 0, s, 1.0) - r_sq

    f_tot = np.zeros_like(r_sq, dtype=np.float64)
    for ai, a0_val in enumerate(a0_centers):
        with np.errstate(divide="ignore", invalid="ignore"):
            g_sq = np.where(inv_base > 0, (1.0 + a0_val) / np.where(inv_base > 0, inv_base, 1.0), -1.0)
        valid = g_sq >= 0
        g = np.sqrt(np.where(valid, g_sq, 0.0))
        gth_sq_inv = 1.0 / (1.0 + r_sq * g_sq) ** 2
        a_fac = 1.0 - 4.0 * cos_pol * r_sq * g_sq * gth_sq_inv
        prefac = a_fac * g ** 5 * gth_sq_inv / (1.0 + a0_val)

        H_val = _interp3d(table.H[:, :, :, ai], grid, g, theta_x, theta_y)
        f_tot += np.where(valid, H_val * prefac, 0.0) * da

    coef = 3.0 / (4.0 * np.pi ** 4 * compton.Wph * 4.0) * reference.PHI_CELLS
    return coef * f_tot / np.where(s > 0, s, 1.0) ** 2


def _unit_square_points(M, scheme, rng):
    if scheme == "fibonacci":
        i = np.arange(M)
        u = (i + 0.5) / M
        v = np.remainder(i * GOLDEN_PHI, 1.0)
    elif scheme == "pseudorandom":
        u = rng.uniform(size=M)
        v = rng.uniform(size=M)
    else:
        raise ValueError(f"scheme must be 'fibonacci' or 'pseudorandom', got {scheme!r}")
    return u, v


def box_quadrature_spectrum(table, compton, x0, y0, s, M, scheme, phi_pol, rng=None):
    """M-sample box quadrature of the (theta_x, theta_y)-integral at every
    frequency in `s` simultaneously (broadcast over (M, len(s))). Returns an
    array matching s's shape: the estimate of dN/ds dOmega... integrated,
    i.e. directly comparable to reference.spectrum_from_table's output.
    """
    grid = table.grid
    tx_lo, tx_hi = float(grid.theta_x_edges[0]), float(grid.theta_x_edges[-1])
    ty_lo, ty_hi = float(grid.theta_y_edges[0]), float(grid.theta_y_edges[-1])
    area = (tx_hi - tx_lo) * (ty_hi - ty_lo)

    u, v = _unit_square_points(M, scheme, rng)
    theta_x = (tx_lo + u * (tx_hi - tx_lo))[:, None]
    theta_y = (ty_lo + v * (ty_hi - ty_lo))[:, None]
    s_row = np.asarray(s, dtype=np.float64)[None, :]

    f = _integrand(table, compton, x0, y0, s_row, theta_x, theta_y, phi_pol)
    return f.mean(axis=0) * area


def _hash_u32(idx, salt):
    """Stateless 32-bit integer hash ("lowbias32", Chris Wellons, public
    domain -- https://nullprogram.com/blog/2018/07/31/): single pass, good
    avalanche, and needs no shared/mutable state -- hash(idx, salt) is fully
    determined by its two integer inputs, so independent samples can be
    generated in any order, or in parallel, with no synchronisation. That
    statelessness is the actual "GPU-friendly" property being demonstrated
    here: a stateful generator (numpy's Generator, cupy's curand) needs a
    per-thread sub-stream to parallelise safely, a hash function does not.
    This module itself runs on CPU/numpy -- the point is the algorithm
    shape, not an actual GPU launch.
    """
    idx = np.asarray(idx, dtype=np.uint64)
    # Wraparound on the uint64/uint32 multiplies below is the intended modular
    # arithmetic, not a bug -- silence numpy's overflow warning for it.
    with np.errstate(over="ignore"):
        x = (idx * np.uint64(0x9E3779B97F4A7C15) + np.uint64(salt) * np.uint64(0xBF58476D1CE4E5B9))
        x = (x & np.uint64(0xFFFFFFFF)).astype(np.uint32)
        x ^= x >> np.uint32(16)
        x = (x * np.uint32(0x7feb352d)) & np.uint32(0xFFFFFFFF)
        x ^= x >> np.uint32(15)
        x = (x * np.uint32(0x846ca68b)) & np.uint32(0xFFFFFFFF)
        x ^= x >> np.uint32(16)
    return x


def _hash_uniform(idx, salt):
    """idx -> a uniform float in [0, 1), via _hash_u32."""
    return _hash_u32(idx, salt).astype(np.float64) / 2.0 ** 32


_theta_marginal_cache = {}


def _theta_marginal(table):
    """(theta_x, theta_y) marginal of H, summed over gamma and a0 -- the
    same coarse proposal density spectrum_kernel_4d itself now uses for its
    phi-importance-sampling CDF (spectrum4d.py: `H_marginal =
    H.sum(axis=(0,3))`). Used here as the proposal density for
    `importance_quadrature_spectrum`'s own from-scratch importance sampler
    -- deliberately the same target the kernel uses, so the "fibonacci" vs
    "hash" comparison isolates the point-generation scheme, not a different
    importance density.
    """
    key = id(table.H)
    cached = _theta_marginal_cache.get(key)
    if cached is None:
        cached = table.H.sum(axis=(0, 3))
        _theta_marginal_cache.clear()  # one table at a time in practice; avoid unbounded growth
        _theta_marginal_cache[key] = cached
    return cached


_theta_cdf_cache = {}


def _theta_grid_cdf(table):
    """Flattened CDF (length n_theta_x*n_theta_y) over the table's *own*
    (theta_x, theta_y) grid cells, weighted by H_marginal*cell_area -- the
    proposal density for importance_quadrature_spectrum. Deliberately the
    table's existing axis-aligned cells, not a separately-constructed polar
    (r, phi) grid: an earlier version of this module built its own
    circular-annulus proposal (matching spectrum_kernel_4d's rmin/rmax
    restriction) discretised into a modest (r, phi) grid, but that broke
    badly on this codebase's actual tables -- OBS_POINTS only ever probes
    theta_y=0, so the deposited theta_y extent ends up ~100x narrower than
    theta_x, and a circle at the resonant radius then intersects the box
    only through a sliver of phi far thinner than any practical phi-bin
    width, so every bin centre can miss the populated region entirely (found
    by comparing against box_quadrature_spectrum/ref_quadrature: the polar
    version returned an M-independent ~40% low bias, i.e. a real coverage
    bug, not sampling noise). Sampling directly from the table's own cells
    sidesteps this: whatever the box's aspect ratio, the cells are already
    shaped to it. This also drops the resonant-annulus restriction
    entirely -- not needed for correctness (the a0-aware `_integrand` still
    zeroes out non-resonant samples), and it exactly matches what
    spectrum_kernel_4d's own coarse proposal does post the a0-dependence fix
    (spectrum4d.py: the coarse pass ignores the resonance condition
    completely, using only the plain H_marginal -- the ring/arc geometry
    that further restricts by radius is a separate, kernel-only, efficiency
    feature, not required for an unbiased estimator). Cached by id(table.H).
    Returns (cdf, dtx, dty), or None if H_marginal is all-zero.
    """
    key = id(table.H)
    cached = _theta_cdf_cache.get(key)
    if cached is None:
        H_marg = _theta_marginal(table)
        grid = table.grid
        dtx, dty = grid.widths[1], grid.widths[2]
        w = H_marg.reshape(-1) * (dtx * dty)
        total = w.sum()
        if total <= 0:
            cached = None
        else:
            cdf = np.cumsum(w)
            cdf /= cdf[-1]
            cached = (cdf, dtx, dty)
        _theta_cdf_cache.clear()  # one table at a time in practice; avoid unbounded growth
        _theta_cdf_cache[key] = cached
    return cached


def _sample_importance(table, x0, y0, M, scheme, salt0=0):
    """Draw M (theta_x, theta_y) samples plus their self-normalised
    importance weights, from the table's own (theta_x, theta_y) grid cells
    weighted by H_marginal (see _theta_grid_cdf). Does not depend on s or
    (x0, y0) at all -- same draw is reused for every frequency/observation
    point, like box_quadrature_spectrum's single draw. Returns None if
    H_marginal is all-zero (no population anywhere in the table).

    `salt0` (scheme="hash" only) offsets the three hash salts so repeated
    calls draw independent noise realisations -- the "hash" analogue of
    box_quadrature_spectrum's repeated pseudorandom draws with a fresh rng
    state each time. "fibonacci" ignores it: it's deterministic, so
    repeating it would just reproduce the same draw (same as fig_sampling.py
    never repeating box_quadrature_spectrum's "fibonacci" scheme either).
    """
    built = _theta_grid_cdf(table)
    if built is None:
        return None
    cdf, dtx, dty = built
    grid = table.grid
    n_ty = grid.shape[2]
    tx_lo = float(grid.theta_x_edges[0])
    ty_lo = float(grid.theta_y_edges[0])

    i = np.arange(M)
    if scheme == "fibonacci":
        # Three independent 1D Kronecker sequences (cell-selection quantile,
        # x-within-cell, y-within-cell) via large additive offsets on the
        # same irrational multiplier, so the three draws don't correlate.
        q_cell = np.remainder(i * GOLDEN_PHI, 1.0)
        q_x = np.remainder((i + 10_000_019) * GOLDEN_PHI, 1.0)
        q_y = np.remainder((i + 20_000_003) * GOLDEN_PHI, 1.0)
    elif scheme == "hash":
        q_cell = _hash_uniform(i, salt=salt0 + 0)
        q_x = _hash_uniform(i, salt=salt0 + 1)
        q_y = _hash_uniform(i, salt=salt0 + 2)
    else:
        raise ValueError(f"scheme must be 'fibonacci' or 'hash', got {scheme!r}")

    cell_idx = np.clip(np.searchsorted(cdf, q_cell, side="right"), 0, cdf.size - 1)
    xi, yj = cell_idx // n_ty, cell_idx % n_ty

    theta_x = tx_lo + dtx * (xi + q_x)
    theta_y = ty_lo + dty * (yj + q_y)

    cell_prob = cdf[cell_idx] - np.where(cell_idx > 0, cdf[cell_idx - 1], 0.0)
    weight = np.where(cell_prob > 0, (dtx * dty) / np.where(cell_prob > 0, cell_prob, 1.0), 0.0)
    return theta_x, theta_y, weight


def importance_quadrature_spectrum(table, compton, x0, y0, s, M, scheme, phi_pol, salt0=0):
    """M-sample importance-sampling estimate of the (theta_x, theta_y)
    integral, importance-sampled by the table's own H_marginal (see module
    docstring for how this compares to spectrum_kernel_4d's own importance
    sampling and to box_quadrature_spectrum's plain uniform sampling).
    `scheme` is "fibonacci" or "hash"; `salt0` is "hash"-only, see
    _sample_importance. Returns an array matching s's shape.
    """
    sampled = _sample_importance(table, x0, y0, M, scheme, salt0=salt0)
    s_arr = np.atleast_1d(np.asarray(s, dtype=np.float64))
    if sampled is None:
        return np.zeros_like(s_arr) if np.ndim(s) else 0.0
    theta_x, theta_y, is_weight = sampled
    s_row = s_arr[None, :]
    f = _integrand(table, compton, x0, y0, s_row, theta_x[:, None], theta_y[:, None], phi_pol)
    out = np.mean(f * is_weight[:, None], axis=0)
    return out if np.ndim(s) else out[0]
