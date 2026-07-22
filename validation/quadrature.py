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
