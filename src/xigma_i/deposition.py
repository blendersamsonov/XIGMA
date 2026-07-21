"""Stage 1: 4D deposition of macroparticle samples into H[gamma, theta_x, theta_y, a0].

Consumes the per-timestep samples produced by particles.push_and_sample and
bins them into the tabulated overlap function described in plan.md /
CLAUDE.md "Planned work". Two deposition schemes share the grid/indexing
logic and differ only in the weight-distribution step:

  - nearest: single-cell binning.
  - cic:     cloud-in-cell, 16-neighbour trilinear... quadrilinear weights.

Two backends, selected via build_table(..., device='cpu'|'gpu'):

  - CPU (numpy): deposit_nearest/deposit_cic, vectorised across samples --
    "single-threaded" in the sense plan.md's build order means it (no
    explicit parallelisation beyond numpy's own vectorisation).
  - GPU (cupyx.jit rawkernel, atomic adds): deposit_nearest_gpu/
    deposit_cic_gpu. Deposits are scattered across the table so contention
    is low in practice (plan.md "Parallelisation"); no sorting/binning
    optimisation attempted here, matching the plan's "don't attempt until
    profiling shows they are needed". The table (H, occupancy, discard
    counter) stays resident on the GPU for the whole call; input sample
    arrays are streamed through in `batch_size`-sized chunks so peak GPU
    memory is bounded and configurable, independent of how many samples
    Stage 0 produced.

See spectrum4d.py's spectrum_kernel_4d for the GPU consumer of the
resulting table (Stage 2).
"""
import numpy as np
import cupy as cp
from cupyx import jit
from dataclasses import dataclass, field

from .core import CP_FLOAT, CP_INT


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
        always clamped to 0 (a0 >= 0 by construction).
        """
        def edges(values, n, lower_clamp=None):
            lo, hi = float(np.min(values)), float(np.max(values))
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
    without re-deriving anything from the original particle set.
    """
    H: np.ndarray
    grid: Grid4D
    scheme: str
    n_particle_samples: int
    total_weight: float
    n_discarded: int
    occupancy: np.ndarray  # int64, same shape as H: raw (unweighted) deposit counts per cell
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


def deposit_nearest(grid, gamma, theta_x, theta_y, a0, weight, accumulate_dtype=np.float64):
    shape = grid.shape
    fg, ftx, fty, fa = _cell_indices(grid, gamma, theta_x, theta_y, a0)

    ig = np.floor(fg).astype(np.int64)
    itx = np.floor(ftx).astype(np.int64)
    ity = np.floor(fty).astype(np.int64)
    ia = np.floor(fa).astype(np.int64)

    in_range = (
        (ig >= 0) & (ig < shape[0]) & (itx >= 0) & (itx < shape[1]) &
        (ity >= 0) & (ity < shape[2]) & (ia >= 0) & (ia < shape[3])
    )
    n_discarded = int((~in_range).sum())

    flat_idx = np.ravel_multi_index(
        (ig[in_range], itx[in_range], ity[in_range], ia[in_range]), shape)

    H_flat = np.zeros(int(np.prod(shape)), dtype=accumulate_dtype)
    np.add.at(H_flat, flat_idx, weight[in_range].astype(accumulate_dtype))

    occ_flat = np.zeros(int(np.prod(shape)), dtype=np.int64)
    np.add.at(occ_flat, flat_idx, 1)

    return H_flat.reshape(shape), occ_flat.reshape(shape), n_discarded


def deposit_cic(grid, gamma, theta_x, theta_y, a0, weight, accumulate_dtype=np.float64, edge='clamp'):
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

    shape = grid.shape
    fg, ftx, fty, fa = _cell_indices(grid, gamma, theta_x, theta_y, a0)

    # Cell-centred convention: sample at continuous coordinate f belongs
    # between the centres of cell floor(f-0.5) and floor(f-0.5)+1.
    fg, ftx, fty, fa = fg - 0.5, ftx - 0.5, fty - 0.5, fa - 0.5

    i0g, i0tx, i0ty, i0a = (np.floor(f).astype(np.int64) for f in (fg, ftx, fty, fa))
    wg, wtx, wty, wa = fg - i0g, ftx - i0tx, fty - i0ty, fa - i0a

    n = gamma.shape[0]
    H_flat = np.zeros(int(np.prod(shape)), dtype=accumulate_dtype)
    occ_flat = np.zeros(int(np.prod(shape)), dtype=np.int64)
    n_discarded = 0

    corner_bounds = [(i0g, shape[0]), (i0tx, shape[1]), (i0ty, shape[2]), (i0a, shape[3])]

    if edge == 'discard':
        valid = np.ones(n, dtype=bool)
        for i0, dim in corner_bounds:
            valid &= (i0 >= 0) & (i0 + 1 < dim)
        n_discarded = int((~valid).sum())
        if not np.any(valid):
            return H_flat.reshape(shape), occ_flat.reshape(shape), n_discarded
        i0g, i0tx, i0ty, i0a = i0g[valid], i0tx[valid], i0ty[valid], i0a[valid]
        wg, wtx, wty, wa = wg[valid], wtx[valid], wty[valid], wa[valid]
        weight = weight[valid]

    for dg_ in (0, 1):
        cg = np.clip(i0g + dg_, 0, shape[0] - 1) if edge == 'clamp' else i0g + dg_
        wg_ = wg if dg_ else (1 - wg)
        for dtx_ in (0, 1):
            ctx = np.clip(i0tx + dtx_, 0, shape[1] - 1) if edge == 'clamp' else i0tx + dtx_
            wtx_ = wtx if dtx_ else (1 - wtx)
            for dty_ in (0, 1):
                cty = np.clip(i0ty + dty_, 0, shape[2] - 1) if edge == 'clamp' else i0ty + dty_
                wty_ = wty if dty_ else (1 - wty)
                for da_ in (0, 1):
                    ca = np.clip(i0a + da_, 0, shape[3] - 1) if edge == 'clamp' else i0a + da_
                    wa_ = wa if da_ else (1 - wa)

                    w_corner = wg_ * wtx_ * wty_ * wa_
                    flat_idx = np.ravel_multi_index((cg, ctx, cty, ca), shape)
                    np.add.at(H_flat, flat_idx, (weight * w_corner).astype(accumulate_dtype))
                    np.add.at(occ_flat, flat_idx, 1)

    return H_flat.reshape(shape), occ_flat.reshape(shape), n_discarded


_GPU_THREADS = 256


@jit.rawkernel()
def _deposit_nearest_kernel(H, occupancy, n_discarded, gamma, theta_x, theta_y, a0, weight, n_samples,
                             gamma_min, gamma_width, n_gamma,
                             theta_x_min, theta_x_width, n_theta_x,
                             theta_y_min, theta_y_width, n_theta_y,
                             a0_min, a0_width, n_a0):
    idx = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    if idx < n_samples:
        ig = CP_INT(cp.floor((gamma[idx] - gamma_min) / gamma_width))
        ix = CP_INT(cp.floor((theta_x[idx] - theta_x_min) / theta_x_width))
        iy = CP_INT(cp.floor((theta_y[idx] - theta_y_min) / theta_y_width))
        ia = CP_INT(cp.floor((a0[idx] - a0_min) / a0_width))

        if ig >= 0 and ig < n_gamma and ix >= 0 and ix < n_theta_x and iy >= 0 and iy < n_theta_y and ia >= 0 and ia < n_a0:
            flat = ((ig * n_theta_x + ix) * n_theta_y + iy) * n_a0 + ia
            jit.atomic_add(H, flat, weight[idx])
            jit.atomic_add(occupancy, flat, CP_INT(1))
        else:
            jit.atomic_add(n_discarded, 0, CP_INT(1))


@jit.rawkernel()
def _deposit_cic_kernel(H, occupancy, n_discarded, gamma, theta_x, theta_y, a0, weight, n_samples,
                         gamma_min, gamma_width, n_gamma,
                         theta_x_min, theta_x_width, n_theta_x,
                         theta_y_min, theta_y_width, n_theta_y,
                         a0_min, a0_width, n_a0, clamp_edges):
    idx = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    if idx < n_samples:
        fg = (gamma[idx] - gamma_min) / gamma_width - CP_FLOAT(0.5)
        ftx = (theta_x[idx] - theta_x_min) / theta_x_width - CP_FLOAT(0.5)
        fty = (theta_y[idx] - theta_y_min) / theta_y_width - CP_FLOAT(0.5)
        fa = (a0[idx] - a0_min) / a0_width - CP_FLOAT(0.5)

        i0g = CP_INT(cp.floor(fg))
        i0x = CP_INT(cp.floor(ftx))
        i0y = CP_INT(cp.floor(fty))
        i0a = CP_INT(cp.floor(fa))
        wg = fg - CP_FLOAT(i0g)
        wx = ftx - CP_FLOAT(i0x)
        wy = fty - CP_FLOAT(i0y)
        wa = fa - CP_FLOAT(i0a)

        if not clamp_edges:
            if i0g < 0 or i0g + 1 >= n_gamma or i0x < 0 or i0x + 1 >= n_theta_x \
                    or i0y < 0 or i0y + 1 >= n_theta_y or i0a < 0 or i0a + 1 >= n_a0:
                jit.atomic_add(n_discarded, 0, CP_INT(1))
                return

        w = weight[idx]
        for dg in range(2):
            cg = i0g + dg
            if clamp_edges:
                cg = min(max(cg, CP_INT(0)), CP_INT(n_gamma - 1))
            wg_ = wg if dg else (CP_FLOAT(1) - wg)
            for dx_ in range(2):
                cx = i0x + dx_
                if clamp_edges:
                    cx = min(max(cx, CP_INT(0)), CP_INT(n_theta_x - 1))
                wx_ = wx if dx_ else (CP_FLOAT(1) - wx)
                for dy_ in range(2):
                    cy = i0y + dy_
                    if clamp_edges:
                        cy = min(max(cy, CP_INT(0)), CP_INT(n_theta_y - 1))
                    wy_ = wy if dy_ else (CP_FLOAT(1) - wy)
                    for da_ in range(2):
                        ca = i0a + da_
                        if clamp_edges:
                            ca = min(max(ca, CP_INT(0)), CP_INT(n_a0 - 1))
                        wa_ = wa if da_ else (CP_FLOAT(1) - wa)

                        flat = ((cg * n_theta_x + cx) * n_theta_y + cy) * n_a0 + ca
                        jit.atomic_add(H, flat, w * wg_ * wx_ * wy_ * wa_)
                        jit.atomic_add(occupancy, flat, CP_INT(1))


def _deposit_gpu(kernel, grid, gamma, theta_x, theta_y, a0, weight, *, batch_size=4_000_000,
                  accumulate_dtype=np.float32, extra_args=()):
    """Shared driver for the GPU nearest/CIC kernels: keeps H/occupancy/discard
    counter resident on the GPU and streams the (possibly host-side, possibly
    much larger than GPU memory) sample arrays through in batches.

    Cell-level agreement with the CPU path (deposit_nearest/deposit_cic on
    the same samples): total weight matches to <1e-6 relative, and per-axis
    marginals to <0.1%, but individual cells can differ by O(1) in relative
    terms for a small fraction of cells (~0.06% with rel diff > 1% in
    testing) -- samples whose bin coordinate falls extremely close to an
    edge get assigned to the cell on one side by float64 CPU arithmetic and
    the cell on the other side by float32 GPU arithmetic. This is a genuine
    boundary/rounding effect, not a bug: verified by comparing the fraction
    of disagreeing cells and confirming total weight and marginals are
    unaffected. Don't rely on any single cell of a GPU-deposited table
    matching the CPU path exactly; aggregate statistics do.
    """
    if accumulate_dtype != np.float32:
        raise ValueError("GPU deposition accumulates in float32 only (cupyx.jit atomic_add); "
                          "use device='cpu' with accumulate_dtype=np.float64 for a precision check")

    shape = grid.shape
    n_cells = int(np.prod(shape))
    H_gpu = cp.zeros(n_cells, dtype=cp.float32)
    occ_gpu = cp.zeros(n_cells, dtype=cp.int32)
    n_discarded_gpu = cp.zeros(1, dtype=cp.int32)

    gamma_min, theta_x_min, theta_y_min, a0_min = (CP_FLOAT(e[0]) for e in
        (grid.gamma_edges, grid.theta_x_edges, grid.theta_y_edges, grid.a0_edges))
    gamma_width, theta_x_width, theta_y_width, a0_width = (CP_FLOAT(w) for w in grid.widths)
    n_gamma, n_theta_x, n_theta_y, n_a0 = (CP_INT(n) for n in shape)

    n_total = gamma.shape[0]
    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)
        g_b = cp.asarray(gamma[start:end], dtype=CP_FLOAT)
        tx_b = cp.asarray(theta_x[start:end], dtype=CP_FLOAT)
        ty_b = cp.asarray(theta_y[start:end], dtype=CP_FLOAT)
        a0_b = cp.asarray(a0[start:end], dtype=CP_FLOAT)
        w_b = cp.asarray(weight[start:end], dtype=CP_FLOAT)
        n_b = end - start

        n_blocks = (n_b + _GPU_THREADS - 1) // _GPU_THREADS
        kernel[n_blocks, _GPU_THREADS](
            H_gpu, occ_gpu, n_discarded_gpu, g_b, tx_b, ty_b, a0_b, w_b, CP_INT(n_b),
            gamma_min, gamma_width, n_gamma,
            theta_x_min, theta_x_width, n_theta_x,
            theta_y_min, theta_y_width, n_theta_y,
            a0_min, a0_width, n_a0, *extra_args)

    cp.cuda.Stream.null.synchronize()
    H = H_gpu.get().astype(accumulate_dtype).reshape(shape)
    occ = occ_gpu.get().astype(np.int64).reshape(shape)
    n_discarded = int(n_discarded_gpu.get()[0])
    return H, occ, n_discarded


def deposit_nearest_gpu(grid, gamma, theta_x, theta_y, a0, weight, accumulate_dtype=np.float32,
                         batch_size=4_000_000):
    return _deposit_gpu(_deposit_nearest_kernel, grid, gamma, theta_x, theta_y, a0, weight,
                         batch_size=batch_size, accumulate_dtype=accumulate_dtype)


def deposit_cic_gpu(grid, gamma, theta_x, theta_y, a0, weight, accumulate_dtype=np.float32,
                     batch_size=4_000_000, edge='clamp'):
    if edge not in ('clamp', 'discard'):
        raise ValueError(f"edge must be 'clamp' or 'discard', got {edge!r}")
    return _deposit_gpu(_deposit_cic_kernel, grid, gamma, theta_x, theta_y, a0, weight,
                         batch_size=batch_size, accumulate_dtype=accumulate_dtype,
                         extra_args=(bool(edge == 'clamp'),))


_DEPOSIT_FUNCS = {'nearest': deposit_nearest, 'cic': deposit_cic}
_DEPOSIT_FUNCS_GPU = {'nearest': deposit_nearest_gpu, 'cic': deposit_cic_gpu}


def gamma_bracket(H, grid, q=1e-4):
    """Lowest/highest gamma at which the table has non-negligible content,
    as the q and 1-q quantiles of the gamma marginal (not raw min/max, so
    isolated stray particles don't inflate the domain). See plan.md Stage 2
    "Annulus brackets from the table".
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


def build_table(gamma, theta_x, theta_y, a0, weight, *, grid=None, scheme='nearest', device='cpu',
                 n_bins=(128, 128, 128, 32), margin=0.05, accumulate_dtype=None,
                 gamma_quantile=1e-4, **scheme_kwargs):
    """Orchestrates Stage 1: grid derivation (if not supplied) + deposition +
    diagnostics, returning a ready-to-save Table.

    device='cpu' (default): numpy, accumulate_dtype defaults to float64.
    device='gpu': cupyx.jit rawkernel with atomic adds, table resident on
        the GPU for the whole call, sample arrays streamed through in
        `batch_size`-sized chunks (see deposit_nearest_gpu/deposit_cic_gpu).
        Only float32 accumulation is supported (cupyx.jit atomic_add); for
        the float32-vs-float64 precision check plan.md asks for, run
        device='cpu' with both accumulate_dtype values and compare via
        check_accumulation_precision.
    """
    funcs = {'cpu': _DEPOSIT_FUNCS, 'gpu': _DEPOSIT_FUNCS_GPU}.get(device)
    if funcs is None:
        raise ValueError(f"device must be 'cpu' or 'gpu', got {device!r}")
    if scheme not in funcs:
        raise ValueError(f"scheme must be one of {list(funcs)}, got {scheme!r}")
    if accumulate_dtype is None:
        accumulate_dtype = np.float64 if device == 'cpu' else np.float32

    if grid is None:
        grid = Grid4D.from_samples(gamma, theta_x, theta_y, a0, n_bins=n_bins, margin=margin)

    H_raw, occupancy, n_discarded = funcs[scheme](
        grid, gamma, theta_x, theta_y, a0, weight, accumulate_dtype=accumulate_dtype, **scheme_kwargs)

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
