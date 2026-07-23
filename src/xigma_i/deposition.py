"""Stage 1: 4D deposition of macroparticle samples into H[gamma, theta_x, theta_y, a0].

a0 here is particles.push_and_sample's trajectory-averaged effective intensity
(one value per particle), not an instantaneous local field amplitude -- see
that function's docstring and CLAUDE.md. Deposition itself is agnostic to
what a0 physically means; only Stage 0 (particles.py) needed to change when
this was corrected.

Consumes the per-timestep samples produced by particles.push_and_sample and
bins them into the tabulated overlap function described in plan.md /
CLAUDE.md "Planned work". Two deposition schemes share the grid/indexing
logic and differ only in the weight-distribution step:

  - nearest: single-cell binning.
  - cic:     cloud-in-cell, 16-neighbour trilinear... quadrilinear weights.

deposit_nearest/deposit_cic are array-module-agnostic: they use only
elementwise arithmetic plus `xp.ravel_multi_index` and a scatter-add
(np.add.at for numpy, cupyx.scatter_add for cupy -- an exact drop-in with
the same signature and, per testing, matching float64 output). There's no
CUDA-specific machinery here (no shared memory, no thread cooperation, no
custom kernel) because there's nothing for a kernel to do beyond a plain
scatter-add: deposits are independent per sample, so the vectorised
form *is* the parallel form. (An earlier version of this file hand-wrote
cupyx.jit rawkernels for this; they added ~150 lines and their own bugs
without doing anything a scatter-add doesn't already do, and their
float32-only accumulation was strictly worse than what scatter_add
supports for free. Removed.)

xp is auto-detected from the input arrays (cp.get_array_module) or forced
via build_table(..., device='cpu'|'gpu'). Batching (build_table(...,
batch_size=...)) converts+deposits host arrays in chunks so peak GPU
memory is bounded independent of how many samples Stage 0 produced ("table
stays resident; particles stream through" -- plan.md), by accumulating
into a running resident H/occupancy rather than materialising the whole
input on the GPU at once.

See spectrum4d.py's spectrum_kernel_4d for the GPU consumer of the
resulting table (Stage 2).
"""
import numpy as np
import cupy as cp
import cupyx
from dataclasses import dataclass


def _scatter_add(xp, out, idx, val):
    """Accumulate val into out at (possibly repeated) idx, in place.

    CPU: np.add.at has a long-standing reputation (well-earned on older
    numpy) for being far slower than a proper vectorised scatter; np.bincount
    -- using the same flat composite index this module already computes --
    is the dedicated, guaranteed-vectorised tool for exactly this pattern
    (accumulate-by-integer-index), so it's used regardless. Measured on this
    environment's numpy (2.5.1): the two are within ~10% of each other in
    isolation (add.at appears to have been optimised significantly since
    the pathological-slowness reports), so don't expect a dramatic win from
    this alone -- but bincount doesn't depend on that being true in whatever
    numpy version this runs on later, and is the more obviously-correct tool
    for the job either way. bincount always accumulates in float64 when
    weights are given; astype(..., copy=False) avoids a redundant full-array
    copy when out is already float64 (the default accumulate_dtype).
    GPU: cupyx.scatter_add is already a proper parallel scatter-add (not
    the numpy pitfall above), kept as-is. It's backed by cupy.add.at, which
    only supports a fixed dtype list (int32, float16, float32, float64,
    uint32, uint64 -- notably NOT int64); callers on this path must not pass
    an int64 `out` (this bit occupancy counting once -- see deposit_nearest's
    comment -- worked on some cupy/driver combinations and raised
    TypeError on others, purely from `out`'s dtype, nothing to do with
    problem size or GPU model).
    """
    if xp is np:
        if np.isscalar(val):
            out += np.bincount(idx, minlength=out.shape[0]).astype(out.dtype, copy=False)
        else:
            out += np.bincount(idx, weights=val, minlength=out.shape[0]).astype(out.dtype, copy=False)
    else:
        cupyx.scatter_add(out, idx, val)


@dataclass
class Grid4D:
    """Uniform 4D grid over (gamma, theta_x, theta_y, a0)."""
    gamma_edges: np.ndarray
    theta_x_edges: np.ndarray
    theta_y_edges: np.ndarray
    a0_edges: np.ndarray

    @property
    def shape(self):
        return (len(self.gamma_edges) - 1, len(self.theta_x_edges) - 1,
                len(self.theta_y_edges) - 1, len(self.a0_edges) - 1)

    @property
    def centers(self):
        return tuple(0.5 * (e[1:] + e[:-1]) for e in
                     (self.gamma_edges, self.theta_x_edges, self.theta_y_edges, self.a0_edges))

    @property
    def widths(self):
        return tuple(e[1] - e[0] for e in
                     (self.gamma_edges, self.theta_x_edges, self.theta_y_edges, self.a0_edges))

    @property
    def bin_volume(self):
        dg, dtx, dty, da = self.widths
        return dg * dtx * dty * da

    @classmethod
    def from_samples(cls, gamma, theta_x, theta_y, a0, n_bins=(128, 128, 128, 32), margin=0.05):
        """Derive grid extents from the populated data range, plus a margin
        expressed as a fraction of each axis's data span. a0's lower edge is
        always clamped to 0 (a0 >= 0 by construction). Accepts numpy or
        cupy input arrays.
        """
        def edges(values, n, lower_clamp=None):
            xp = cp.get_array_module(values)
            lo, hi = float(xp.min(values)), float(xp.max(values))
            span = hi - lo
            if span <= 0:
                span = max(abs(lo), 1.0)
            pad = margin * span
            lo, hi = lo - pad, hi + pad
            if lower_clamp is not None:
                lo = max(lo, lower_clamp)
            return np.linspace(lo, hi, n + 1)

        return cls(
            gamma_edges=edges(gamma, n_bins[0]),
            theta_x_edges=edges(theta_x, n_bins[1]),
            theta_y_edges=edges(theta_y, n_bins[2]),
            a0_edges=edges(a0, n_bins[3], lower_clamp=0.0),
        )


@dataclass
class Table:
    """A deposited H table plus everything Stage 2 needs to consume it
    without re-deriving anything from the original particle set. Always
    holds host (numpy) arrays, regardless of which device built it.
    """
    H: np.ndarray
    grid: Grid4D
    scheme: str
    n_particle_samples: int
    total_weight: float
    n_discarded: int
    occupancy: np.ndarray  # int32, same shape as H: raw (unweighted) deposit counts per cell
    gamma_bracket: tuple  # (gamma_lo, gamma_hi) at quantiles (q, 1-q) of the gamma marginal

    def save(self, path):
        np.savez(
            path,
            H=self.H,
            gamma_edges=self.grid.gamma_edges,
            theta_x_edges=self.grid.theta_x_edges,
            theta_y_edges=self.grid.theta_y_edges,
            a0_edges=self.grid.a0_edges,
            scheme=self.scheme,
            n_particle_samples=self.n_particle_samples,
            total_weight=self.total_weight,
            n_discarded=self.n_discarded,
            occupancy=self.occupancy,
            gamma_bracket=np.array(self.gamma_bracket),
        )

    @classmethod
    def load(cls, path):
        d = np.load(path)
        grid = Grid4D(d['gamma_edges'], d['theta_x_edges'], d['theta_y_edges'], d['a0_edges'])
        return cls(
            H=d['H'], grid=grid, scheme=str(d['scheme']),
            n_particle_samples=int(d['n_particle_samples']), total_weight=float(d['total_weight']),
            n_discarded=int(d['n_discarded']), occupancy=d['occupancy'],
            gamma_bracket=tuple(d['gamma_bracket'].tolist()),
        )


def _cell_indices(grid, gamma, theta_x, theta_y, a0):
    """Continuous (float) cell coordinates along each axis, i.e. how many
    bin-widths past the lower edge each sample falls. Shared by nearest and
    CIC so their edge handling can't silently diverge.
    """
    dg, dtx, dty, da = grid.widths
    fg = (gamma - grid.gamma_edges[0]) / dg
    ftx = (theta_x - grid.theta_x_edges[0]) / dtx
    fty = (theta_y - grid.theta_y_edges[0]) / dty
    fa = (a0 - grid.a0_edges[0]) / da
    return fg, ftx, fty, fa


def deposit_nearest(grid, gamma, theta_x, theta_y, a0, weight, accumulate_dtype=np.float64, xp=None):
    xp = xp or cp.get_array_module(gamma, theta_x, theta_y, a0, weight)
    shape = grid.shape
    fg, ftx, fty, fa = _cell_indices(grid, gamma, theta_x, theta_y, a0)

    ig = xp.floor(fg).astype(xp.int64)
    itx = xp.floor(ftx).astype(xp.int64)
    ity = xp.floor(fty).astype(xp.int64)
    ia = xp.floor(fa).astype(xp.int64)

    in_range = (
        (ig >= 0) & (ig < shape[0]) & (itx >= 0) & (itx < shape[1]) &
        (ity >= 0) & (ity < shape[2]) & (ia >= 0) & (ia < shape[3])
    )
    n_discarded = int((~in_range).sum())

    flat_idx = xp.ravel_multi_index(
        (ig[in_range], itx[in_range], ity[in_range], ia[in_range]), shape)

    H_flat = xp.zeros(int(np.prod(shape)), dtype=accumulate_dtype)
    _scatter_add(xp, H_flat, flat_idx, weight[in_range].astype(accumulate_dtype))

    # int32, not int64: cupy.add.at (which cupyx.scatter_add uses under the
    # hood on GPU) doesn't support int64 -- see _scatter_add's docstring.
    # int32 tops out at ~2.1e9, far beyond any realistic per-cell count.
    occ_flat = xp.zeros(int(np.prod(shape)), dtype=xp.int32)
    _scatter_add(xp, occ_flat, flat_idx, 1)

    return H_flat.reshape(shape), occ_flat.reshape(shape), n_discarded


def deposit_cic(grid, gamma, theta_x, theta_y, a0, weight, accumulate_dtype=np.float64, edge='clamp', xp=None):
    """Cloud-in-cell deposition: each sample is spread over the 16
    neighbouring cells of the 4D grid, weighted by the product of the 1D
    linear weight along each axis.

    edge='clamp': corner indices are clamped into [0, n-1]; a sample near a
        boundary deposits its full weight (folded onto the boundary cell)
        rather than losing the fraction that would fall outside the grid.
        Total weight is conserved; the deposit is not exactly the ideal CIC
        stencil right at the edge.
    edge='discard': any sample whose 16-cell stencil would touch an
        out-of-range index is dropped entirely (and counted). Exact CIC
        stencil everywhere it's applied, at the cost of losing weight near
        the boundary.
    """
    if edge not in ('clamp', 'discard'):
        raise ValueError(f"edge must be 'clamp' or 'discard', got {edge!r}")
    xp = xp or cp.get_array_module(gamma, theta_x, theta_y, a0, weight)

    shape = grid.shape
    fg, ftx, fty, fa = _cell_indices(grid, gamma, theta_x, theta_y, a0)

    # Cell-centred convention: sample at continuous coordinate f belongs
    # between the centres of cell floor(f-0.5) and floor(f-0.5)+1.
    fg, ftx, fty, fa = fg - 0.5, ftx - 0.5, fty - 0.5, fa - 0.5

    i0g, i0tx, i0ty, i0a = (xp.floor(f).astype(xp.int64) for f in (fg, ftx, fty, fa))
    wg, wtx, wty, wa = fg - i0g, ftx - i0tx, fty - i0ty, fa - i0a

    n = gamma.shape[0]
    H_flat = xp.zeros(int(np.prod(shape)), dtype=accumulate_dtype)
    occ_flat = xp.zeros(int(np.prod(shape)), dtype=xp.int32)  # see deposit_nearest's comment
    n_discarded = 0

    corner_bounds = [(i0g, shape[0]), (i0tx, shape[1]), (i0ty, shape[2]), (i0a, shape[3])]

    if edge == 'discard':
        valid = xp.ones(n, dtype=bool)
        for i0, dim in corner_bounds:
            valid &= (i0 >= 0) & (i0 + 1 < dim)
        n_discarded = int((~valid).sum())
        if not bool(xp.any(valid)):
            return H_flat.reshape(shape), occ_flat.reshape(shape), n_discarded
        i0g, i0tx, i0ty, i0a = i0g[valid], i0tx[valid], i0ty[valid], i0a[valid]
        wg, wtx, wty, wa = wg[valid], wtx[valid], wty[valid], wa[valid]
        weight = weight[valid]

    for dg_ in (0, 1):
        cg = xp.clip(i0g + dg_, 0, shape[0] - 1) if edge == 'clamp' else i0g + dg_
        wg_ = wg if dg_ else (1 - wg)
        for dtx_ in (0, 1):
            ctx = xp.clip(i0tx + dtx_, 0, shape[1] - 1) if edge == 'clamp' else i0tx + dtx_
            wtx_ = wtx if dtx_ else (1 - wtx)
            for dty_ in (0, 1):
                cty = xp.clip(i0ty + dty_, 0, shape[2] - 1) if edge == 'clamp' else i0ty + dty_
                wty_ = wty if dty_ else (1 - wty)
                for da_ in (0, 1):
                    ca = xp.clip(i0a + da_, 0, shape[3] - 1) if edge == 'clamp' else i0a + da_
                    wa_ = wa if da_ else (1 - wa)

                    w_corner = wg_ * wtx_ * wty_ * wa_
                    flat_idx = xp.ravel_multi_index((cg, ctx, cty, ca), shape)
                    _scatter_add(xp, H_flat, flat_idx, (weight * w_corner).astype(accumulate_dtype))
                    _scatter_add(xp, occ_flat, flat_idx, 1)

    return H_flat.reshape(shape), occ_flat.reshape(shape), n_discarded


_DEPOSIT_FUNCS = {'nearest': deposit_nearest, 'cic': deposit_cic}


def _deposit(scheme, grid, gamma, theta_x, theta_y, a0, weight, *, xp, accumulate_dtype,
             batch_size=None, **scheme_kwargs):
    """Dispatches to deposit_nearest/deposit_cic, optionally batching: if
    batch_size is given and the input arrays don't already belong to `xp`
    (e.g. host numpy samples being deposited with xp=cupy), converts and
    deposits `batch_size`-sized chunks at a time, accumulating into a
    resident `xp` H/occupancy rather than materialising the whole input on
    the target device at once.
    """
    deposit_fn = _DEPOSIT_FUNCS[scheme]
    input_xp = cp.get_array_module(gamma)

    if batch_size is None or input_xp is xp:
        if input_xp is not xp:
            gamma, theta_x, theta_y, a0, weight = (xp.asarray(a) for a in (gamma, theta_x, theta_y, a0, weight))
        return deposit_fn(grid, gamma, theta_x, theta_y, a0, weight,
                           accumulate_dtype=accumulate_dtype, xp=xp, **scheme_kwargs)

    shape = grid.shape
    H_total = xp.zeros(shape, dtype=accumulate_dtype)
    occ_total = xp.zeros(shape, dtype=xp.int32)  # see deposit_nearest's comment on scatter_add dtypes
    n_discarded_total = 0
    n = gamma.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        g_b, tx_b, ty_b, a0_b, w_b = (xp.asarray(a[start:end]) for a in (gamma, theta_x, theta_y, a0, weight))
        H_b, occ_b, n_disc_b = deposit_fn(grid, g_b, tx_b, ty_b, a0_b, w_b,
                                           accumulate_dtype=accumulate_dtype, xp=xp, **scheme_kwargs)
        H_total += H_b
        occ_total += occ_b
        n_discarded_total += n_disc_b
    return H_total, occ_total, n_discarded_total


def gamma_bracket(H, grid, q=1e-4):
    """Lowest/highest gamma at which the table has non-negligible content,
    as the q and 1-q quantiles of the gamma marginal (not raw min/max, so
    isolated stray particles don't inflate the domain). See plan.md Stage 2
    "Annulus brackets from the table". H must be a host (numpy) array.
    """
    gamma_centers = grid.centers[0]
    marginal = H.sum(axis=(1, 2, 3))
    total = marginal.sum()
    if total <= 0:
        return float(grid.gamma_edges[0]), float(grid.gamma_edges[-1])
    cdf = np.cumsum(marginal) / total
    lo = float(np.interp(q, cdf, gamma_centers))
    hi = float(np.interp(1 - q, cdf, gamma_centers))
    return lo, hi


def check_accumulation_precision(H_f64, H_f32, rtol=1e-3):
    """Compare float64 vs float32 accumulation of the same deposits.
    Returns (max_relative_difference, recommend_float64: bool).
    """
    denom = np.maximum(np.abs(H_f64), 1e-300)
    rel_diff = np.abs(H_f64 - H_f32.astype(np.float64)) / denom
    mask = H_f64 > 0
    max_rel = float(rel_diff[mask].max()) if np.any(mask) else 0.0
    return max_rel, max_rel > rtol


def build_table(gamma, theta_x, theta_y, a0, weight, *, grid=None, scheme='nearest', device=None,
                 n_bins=(128, 128, 128, 32), margin=0.05, accumulate_dtype=np.float64,
                 gamma_quantile=1e-4, batch_size=None, **scheme_kwargs):
    """Orchestrates Stage 1: grid derivation (if not supplied) + deposition +
    diagnostics, returning a ready-to-save Table (always host/numpy arrays,
    regardless of which device built it).

    device: None (default) auto-detects from `gamma`'s array type (numpy ->
        CPU, cupy -> GPU); 'cpu'/'gpu' forces numpy/cupy, converting the
        input arrays if needed.
    batch_size: if set and a device conversion is needed (e.g. host numpy
        samples with device='gpu'), streams the conversion+deposition in
        chunks instead of transferring everything at once -- see
        _deposit's docstring. Ignored if no conversion is needed.
    accumulate_dtype: np.float64 by default on both CPU and GPU (cupyx.
        scatter_add supports float64 natively); pass np.float32 for less
        memory/faster on very large depositions, and compare against the
        float64 result via check_accumulation_precision per plan.md's
        "Accumulation precision" guidance if in doubt.
    """
    if scheme not in _DEPOSIT_FUNCS:
        raise ValueError(f"scheme must be one of {list(_DEPOSIT_FUNCS)}, got {scheme!r}")
    xp = {'cpu': np, 'gpu': cp}.get(device)
    if device is not None and xp is None:
        raise ValueError(f"device must be 'cpu', 'gpu', or None, got {device!r}")
    if xp is None:
        xp = cp.get_array_module(gamma)

    if grid is None:
        grid = Grid4D.from_samples(gamma, theta_x, theta_y, a0, n_bins=n_bins, margin=margin)

    H_raw, occupancy, n_discarded = _deposit(
        scheme, grid, gamma, theta_x, theta_y, a0, weight,
        xp=xp, accumulate_dtype=accumulate_dtype, batch_size=batch_size, **scheme_kwargs)

    if xp is cp:
        H_raw = H_raw.get()
        occupancy = occupancy.get()

    H_density = H_raw / grid.bin_volume

    bracket = gamma_bracket(H_density, grid, q=gamma_quantile)

    return Table(
        H=H_density, grid=grid, scheme=scheme, n_particle_samples=int(gamma.shape[0]),
        total_weight=float(H_raw.sum()), n_discarded=n_discarded, occupancy=occupancy,
        gamma_bracket=bracket,
    )


def occupancy_diagnostics(table):
    """Fraction of empty cells, histogram of per-cell deposit counts, and
    per-axis marginals -- the direct measure of whether the table is
    adequately populated (see CLAUDE.md "Table too sparse").
    """
    occ = table.occupancy
    n_cells = occ.size
    n_empty = int((occ == 0).sum())
    max_count = int(occ.max()) if n_cells else 0
    hist, hist_edges = np.histogram(occ.ravel(), bins=min(50, max(1, max_count + 1)))
    marginals = {
        'gamma': table.H.sum(axis=(1, 2, 3)),
        'theta_x': table.H.sum(axis=(0, 2, 3)),
        'theta_y': table.H.sum(axis=(0, 1, 3)),
        'a0': table.H.sum(axis=(0, 1, 2)),
    }
    return {
        'n_cells': n_cells,
        'n_empty': n_empty,
        'empty_fraction': n_empty / n_cells if n_cells else 0.0,
        'occupancy_histogram': (hist, hist_edges),
        'max_count': max_count,
        'marginals': marginals,
    }
