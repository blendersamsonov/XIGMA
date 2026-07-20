"""Stage 2 numpy reference paths -- deliberately not the GPU quadrature.

Three independent ways to turn (Stage 0/1) particle samples or an H table
into a spectrum, used only for validation (plan.md "Validation" steps 1
and 2):

  - angle_integrated_spectrum: dN/ds integrated over all emission solid
    angle, computed directly from real Stage 0/1 macroparticles using only
    the standard (textbook, angle-independent) Compton edge shape -- no
    coef, no collision/H lookup, no area quadrature at all. This is the
    one that's actually validated (see below) and is the right first check
    to run: it isolates the gamma axis and the Stage 0/1 weight
    normalisation from everything angular.
  - spectrum_from_table: brute-force grid quadrature over the H table (no
    annulus/arc/inverse-CDF importance sampling), for the per-solid-angle
    (theta_x, theta_y, s) spectrum. Has an unresolved normalisation issue,
    see below -- don't trust its absolute values yet.
  - direct_binning_spectrum: iterates real macroparticles, computes each
    one's resonance frequency for a fixed observation direction, and bins
    with its weight. No table, no quadrature grid at all. Intended as the
    primary correctness test for correlated bunches (plan.md validation 3)
    and to be kept permanently as a debug tool, but shares the same
    unresolved issue as spectrum_from_table -- see below.

WHAT'S VALIDATED: angle_integrated_spectrum, run on real Stage 0/1 output,
matches calculate_spectrum(s, gamma0, sigma_gamma0) to within 0.5-3% across
the whole resonance peak (worse only in the low-statistics tail), *after*
correcting for a units mismatch: calculate_spectrum returns dN/dE (matches
its use in calculate-spec-ang.py, axis label "dN/dE, MeV^-1"), not dN/ds.
dE = 4*Wph*ds (see calculate-spec-ang.py's `s_scale = 4*compton.Wph`), so
compare as `dN_ds_mine` vs `calculate_spectrum(...) * 4 * compton.Wph`. This
is a real, clean validation of Stage 0/1's gamma axis and overall weight
normalisation, independent of collision/H lookup or any theta quadrature.

WHAT'S STILL OPEN: spectrum_from_table and direct_binning_spectrum (the
per-solid-angle, (theta_x, theta_y, s)-resolved paths) disagree with
calculate_angular_spectrum's value at the same (x0, y0, s) by a large,
consistent, resolution-independent factor (~20-30x in the cases tried).
Investigation so far:
  - Per-sample integrand (ffac/a_fac/g**5/gth_sq_inv times `collision` or a
    real macroparticle) matches spectrum_kernel's own values exactly --
    verified by dumping spectrum_kernel's debug arrays and comparing
    point-by-point, ratio 1.0000 +/- 3e-5 over thousands of samples.
  - It is *not* the dN/dE-vs-dN/ds units confusion that explained the
    angle-integrated case above: spectrum_from_table and
    calculate_angular_spectrum use the *identical* coef and /s**2, so
    whatever units calculate_angular_spectrum's output is in (dN/dOmega dE,
    per its use in calculate-spec-ang.py), spectrum_from_table's output is
    already in the same units by construction -- no conversion should be
    needed between them, yet they still disagree.
  - calculate_angular_spectrum itself is not the problem: integrating its
    output over a fine (theta_x, theta_y) grid reproduces calculate_spectrum
    to within 2-3% (both in dN/dE units, no conversion needed for *that*
    comparison), stably across grid resolutions from 21 to 121 points per
    axis.
  - Dividing spectrum_kernel's `sample_area` by PHI_CELLS (a natural-looking
    guess, since the observed factor is close to PHI_CELLS=31) was tried
    directly in the kernel and made agreement with calculate_spectrum
    *worse*, ruling that out.
  - So the gap is isolated to spectrum_kernel's per-observation-point
    (theta, phi) quadrature/importance-sampling machinery specifically, not
    to units, not to the deposition, and not to calculate_angular_spectrum's
    own normalisation constant.

Net: Stage 0/1 (particles.py, deposition.py) are validated two ways now --
table total weight/theta marginal/gamma marginal matching
calculate_total()/calculate_intersection to 1-3%, and angle_integrated_spectrum
matching calculate_spectrum to 0.5-3%. Do not trust absolute values out of
spectrum_from_table or direct_binning_spectrum until the remaining issue is
resolved; shapes/relative comparisons (e.g. nearest vs cic) are unaffected
since the missing factor is a point-independent constant in every case
observed so far.
"""
import numpy as np


def angle_integrated_spectrum(gamma, particle_weight, s):
    """dN/ds integrated over all emission solid angle, from real Stage 0/1
    macroparticles. A single electron's angle-integrated spectral shape
    depends only on its own gamma (not its transverse angle), via the
    standard Compton edge formula also used -- in a different
    parametrisation -- by calculate_spectrum. Compare against
    `calculate_spectrum(s, gamma0, sigma_gamma0) * 4 * compton.Wph` (unit
    conversion from dN/dE to dN/ds, see module docstring).

    gamma, particle_weight: 1D arrays, one entry per macroparticle (e.g.
    bunch.gamma and the per-particle sum of push_and_sample's weight over
    its own timesteps).
    s: scalar or 1D array of normalised photon energies.
    """
    s_arr = np.atleast_1d(np.asarray(s, dtype=np.float64))
    gamma = gamma[:, None]
    y = s_arr[None, :] / gamma**2
    shape = 1.5 * (1.0 - 2.0 * y * (1.0 - y))
    shape = np.where((y < 0) | (y > 1), 0.0, shape)
    out = np.sum(particle_weight[:, None] * shape / gamma**2, axis=0)
    return out if np.ndim(s) else out[0]


def interp4d(table, gamma, theta_x, theta_y, a0):
    """Quadrilinear interpolation of table.H at query points (arrays of equal
    shape). Points outside the tabulated extent return 0.
    """
    grid = table.grid
    axes_edges = (grid.gamma_edges, grid.theta_x_edges, grid.theta_y_edges, grid.a0_edges)
    coords = (gamma, theta_x, theta_y, a0)
    shape = table.H.shape

    out_shape = np.broadcast(*coords).shape
    coords = [np.broadcast_to(c, out_shape).astype(np.float64) for c in coords]

    f = np.zeros(out_shape, dtype=np.float64)
    in_range = np.ones(out_shape, dtype=bool)

    i0s, ws = [], []
    for edges, x, n in zip(axes_edges, coords, shape):
        width = edges[1] - edges[0]
        centers0 = edges[0] + 0.5 * width  # centre of first cell
        f_idx = (x - centers0) / width
        i0 = np.floor(f_idx).astype(np.int64)
        w = f_idx - i0
        in_range &= (i0 >= -1) & (i0 < n)  # allow i0==-1/n-1 edge cases below via clipping + zero-weight
        i0s.append(i0)
        ws.append(w)

    for dg in (0, 1):
        for dtx in (0, 1):
            for dty in (0, 1):
                for da in (0, 1):
                    idxs = []
                    corner_w = np.ones(out_shape, dtype=np.float64)
                    valid = np.ones(out_shape, dtype=bool)
                    for (i0, w, d, n) in zip(i0s, ws, (dg, dtx, dty, da), shape):
                        ci = i0 + d
                        valid &= (ci >= 0) & (ci < n)
                        corner_w = corner_w * (w if d else (1 - w))
                        idxs.append(np.clip(ci, 0, n - 1))
                    take = valid & in_range
                    if np.any(take):
                        vals = table.H[idxs[0][take], idxs[1][take], idxs[2][take], idxs[3][take]]
                        f[take] += vals * corner_w[take]

    f[~in_range] = 0.0
    return f


def spectrum_from_table(table, compton, x0, y0, s, phi_pol):
    """Brute-force quadrature of dN/(ds dOmega) at a single observation point
    (x0, y0) over a grid of frequencies s, integrating the table over its
    full (theta_x, theta_y, a0) extent at each s.

    x0, y0, s: floats / 1D array for s. Returns array matching s's shape.
    """
    theta_x_c, theta_y_c, a0_c = table.grid.centers[1], table.grid.centers[2], table.grid.centers[3]
    TX, TY, A0 = np.meshgrid(theta_x_c, theta_y_c, a0_c, indexing='ij')
    r_sq = (TX - x0)**2 + (TY - y0)**2

    dtx, dty, da = table.grid.widths[1], table.grid.widths[2], table.grid.widths[3]
    cell_vol = dtx * dty * da

    cos_pol = np.cos(phi_pol - np.arctan2(TY - y0, TX - x0))**2

    coef = 3.0 / (4.0 * np.pi**4 * compton.Wph * 4.0)

    s_arr = np.atleast_1d(np.asarray(s, dtype=np.float64))
    out = np.zeros_like(s_arr)
    for k, sk in enumerate(s_arr):
        g_sq = 1.0 / (1.0 / sk - r_sq)
        valid = g_sq >= 0
        if not np.any(valid):
            continue
        g = np.sqrt(np.where(valid, g_sq, 0.0))
        gth_sq_inv = 1.0 / (1.0 + r_sq * g_sq)**2
        a_fac = 1.0 - 4.0 * cos_pol * r_sq * g_sq * gth_sq_inv

        H_val = interp4d(table, g, TX, TY, A0)

        f = np.where(valid, H_val * a_fac * g**5 * gth_sq_inv, 0.0)
        out[k] = coef * f.sum() * cell_vol / sk**2

    return out if np.ndim(s) else out[0]


def direct_binning_spectrum(gamma, theta_x, theta_y, particle_weight, compton,
                             x0, y0, s_edges, phi_pol):
    """Reference path: for each real macroparticle, compute the photon
    energy it resonates at when viewed from (x0, y0), and bin its weight
    into the s_edges histogram with the same physical prefactors spectrum_kernel
    applies (g**5 * gth_sq_inv * polarisation factor). No table, no importance
    sampling -- assumption-free on both the deposition and the lookup.
    """
    r_sq = (theta_x - x0)**2 + (theta_y - y0)**2
    g = gamma
    s_res = g**2 / (1.0 + g**2 * r_sq)

    gth_sq_inv = 1.0 / (1.0 + r_sq * g**2)**2
    cos_pol = np.cos(phi_pol - np.arctan2(theta_y - y0, theta_x - x0))**2
    a_fac = 1.0 - 4.0 * cos_pol * r_sq * g**2 * gth_sq_inv

    prefactor = particle_weight * a_fac * g**5 * gth_sq_inv

    hist, _ = np.histogram(s_res, bins=s_edges, weights=prefactor)
    ds = np.diff(s_edges)
    s_centers = 0.5 * (s_edges[1:] + s_edges[:-1])

    coef = 3.0 / (4.0 * np.pi**4 * compton.Wph * 4.0)
    return coef * hist / ds / s_centers**2
