"""Stage 2: spectrum kernel modifications for the tabulated 4D overlap function.

`spectrum_kernel_4d` is a new kernel, not an edit of core.py's spectrum_kernel
-- see plan.md's build order and the discussion in reference.py's module
docstring for why the legacy kernel is left untouched on this branch. It
reuses spectrum_kernel's ring/rectangle annulus geometry, arc construction,
and inverse-CDF phi sampling unchanged in structure (per plan.md: "Everything
else ... is unchanged"), and changes three things:

1. The 2D bilinear `collision` lookup + analytic Gaussian `ffac` becomes a
   lookup into H[gamma, theta_x, theta_y, a0]. The coarse proposal (used to
   build the phi-importance-sampling CDF) uses `H_marginal`, a precomputed
   `(theta_x, theta_y)` marginal of H summed over *both* gamma and a0
   (`H.sum(axis=(0,3))`, computed once per calculate_angular_spectrum_4d call
   on the host/GPU, not per output point) -- the direct analogue of the
   legacy kernel's precomputed 2D `collision` array. This deliberately
   ignores the resonance condition (no dependence on s/gamma/a0 at all): it
   only decides *where* to place phi/theta samples, and importance sampling
   stays unbiased regardless of what density built the CDF as long as the
   final evaluation reweights against the density actually used -- which it
   does, via `sample_area`. An earlier version of this coarse pass tried to
   track the a0-dependent resonance per bin here too; that's unnecessary
   work for a proposal density and was reverted in favour of the plain
   marginal. The final evaluation uses quadrilinear interpolation of H in
   (gamma, theta_x, theta_y), with an a0 QUADRATURE (not importance-sampled:
   a0 has few bins by design -- see plan.md's "Memory" note -- so a plain
   midpoint sum over the table's own a0 bins is accurate and cheap) nested
   inside the existing phi/subsampling loop -- this is where the resonance
   condition (point 4 below) actually matters, and it's computed exactly
   there, per a0 bin.
2. rmin_g/rmax_g bracket from `gamma_lo`/`gamma_hi` (the table's populated
   gamma extent, e.g. from deposition.gamma_bracket) instead of
   `gamma0 +/- 3*sigma_g`, and widened over the table's full `a0` range (see
   point 4) rather than evaluated at a single a0 value.
3. No `emulate_nonlinearity`/f_a: a0 is a real axis now, so the ponderomotive
   shift is carried by the data instead of a phenomenological correction. Note
   `a0` here is the trajectory-averaged effective intensity (Paper/xigma.tex
   eq. "ahattraj"), not an instantaneous per-timestep value -- see
   particles.push_and_sample's docstring and CLAUDE.md.
4. The resonance condition itself includes a0 (Paper/xigma.tex eq. "Gamma"
   §"Reduction to three dimensions"): `g**2 = (1+a0) / (1/s - theta**2)`, not
   `g**2 = 1/(1/s - theta**2)` -- each a0 bin resonates at its own gamma for a
   given (s, theta), not the shared gamma the legacy kernel/collision model
   used. The evaluation prefactor also picks up a Jacobian factor `1/(1+a0)`
   from this (eq. "jacobian", "Fmatrix"): `prefac = a_fac * g**5 * gth_sq_inv
   / (1+a0)`. Both g and prefac are therefore computed *inside* the a0
   quadrature loop of the final evaluation, not once and shared across a0
   bins -- an earlier version of this kernel got this wrong (used a single
   a0-independent g throughout, missing the `1/(1+a0)` factor entirely); see
   CLAUDE.md "Known bugs"/"Traps". `reference.spectrum_from_table` had the
   same gap and was fixed alongside this kernel for the same reason. (The
   coarse proposal, point 1 above, does not need this at all -- it never
   computes a resonant g.)

Normalisation: this kernel's importance-sampling weights are derived
independently (dphi_cell = (phi_max-phi_min)/PHI_CELLS used consistently
throughout, not the legacy kernel's (phi_max-phi_min)) -- that PHI_CELLS
usage is purely internal to this kernel's own phi-cell importance sampling
and has nothing to do with the output-scale `coef` below.

`coef = 1.5`, a pure numerical constant from eq. "main"/"Fmatrix" (Paper/
xigma.tex), same as reference.spectrum_from_table's identical fix -- see
that module's docstring point 2 for the full derivation. An earlier version
of both this driver and spectrum_from_table used `coef =
3/(4*pi**4*Wph*4) * PHI_CELLS`, copied from the legacy core.py kernel's own
(separately, empirically tuned) constant; that was root-caused this session
as the cause of a severe (4+ orders of magnitude near the Compton edge)
shape divergence from angle_integrated_spectrum, documented in
direct_vs_table_discrepancy_report.md. `compton` is no longer needed by
calculate_angular_spectrum_4d (it was only ever used for compton.Wph) and
has been dropped from its signature accordingly.

Previously validated (uncorrelated bunch, see particles.sample_bunch with
chirp=0, angle_energy_corr=0) to 6-8% max error in the peak region against
reference.spectrum_from_table -- but that number predates the a0-dependent
resonance/Jacobian fix (point 4 above) and needs re-measuring, not assumed
to still hold. The two most likely sources of a *changed* (not necessarily
worse) match: the annulus now covers a wider r range (whole a0 axis, not a
single value), and each a0 bin's own gamma-bin lookup means the effective
gamma resolution sampled per output point differs from before.

Two guards added beyond a straight port of spectrum_kernel's structure,
both needed because H (finite-particle deposition) has exact-zero cells in
a way the legacy smooth analytic `collision` essentially never does:
  - inv_cdf: a run of zero-weight phi cells can make the CDF flat at a
    sample point (cdf_ip1 == cdf_i); falls back to the cell's left edge
    instead of computing 0/0.
  - sample_area: a sample can land in a zero-weight cell after
    interpolation; contributes zero instead of computing x/0.
Without these two, the kernel produces NaN output for realistic
(sparsely-populated) tables. See CLAUDE.md's "Table too sparse" trap.

MAX_ARCS overflow guard added (core.py flags this as unguarded -- CLAUDE.md
"Traps"): if ring/quadrant geometry produces more arcs than MAX_ARCS, the
excess are dropped (undercounts flux for that output point rather than
corrupting shared memory).
"""
import cupy as cp
from cupyx import jit

from .core import (
    CP_FLOAT, CP_UINT, CP_INT, CP_ONE, CP_ZERO, CP_TWO_PI, PHI as GOLDEN_PHI,
    X_THREADS, N_RINGS_MIN, MAX_RINGS, MAX_ARCS, ARC_STRIDE, RING_STRIDE,
    RINGS_SIZE, INVAL, PHI_EDGES, PHI_CELLS, CUM_WEIGHTS_SIZE,
    CDF_PHI_RESOLUTION, CDF_PHI_REPEAT, CDF_SIZE, SAMPLES_TOTAL,
    SAMPLES_REPEAT, THREAD_STRIDE, R_MAX_NUDGE,
)

N_A0_MAX = 32  # upper bound on table.grid.shape[3] this kernel supports; see calculate_angular_spectrum_4d


@jit.rawkernel()
def spectrum_kernel_4d(output, params_Arr, H, H_marginal,
                        gamma_min, gamma_width, n_gamma,
                        theta_x_min, theta_x_width, n_theta_x,
                        theta_y_min, theta_y_width, n_theta_y,
                        a0_min, a0_width, n_a0,
                        gamma_lo, gamma_hi, dx, dy,
                        phi_pol, subsampling, debug_arr, debug_idx, dbg_scalars):
    thread_idx = jit.threadIdx.x
    out_idx = jit.blockIdx.x

    inv_cdf = jit.shared_memory(CP_FLOAT, CDF_SIZE)
    TMP_FLOAT_ARRAY = inv_cdf

    n_arcs_shared = jit.shared_memory(CP_UINT, 1)
    arcs = jit.shared_memory(CP_FLOAT, ARC_STRIDE * MAX_ARCS)
    cum_cell_weights = jit.shared_memory(CP_FLOAT, CUM_WEIGHTS_SIZE)
    thread_samples = jit.shared_memory(CP_UINT, X_THREADS * THREAD_STRIDE)

    x0 = params_Arr[out_idx, 0]
    y0 = params_Arr[out_idx, 1]
    s = params_Arr[out_idx, 2]

    a0_max = a0_min + a0_width * CP_FLOAT(n_a0)
    rmin_g = cp.sqrt(cp.maximum(CP_ZERO, CP_ONE / s - (CP_ONE + a0_max) / gamma_lo**2))
    rmax_g = cp.sqrt(cp.maximum(CP_ZERO, CP_ONE / s - (CP_ONE + a0_min) / gamma_hi**2))

    rmin_r = cp.sqrt(max(cp.abs(x0) - dx, CP_ZERO)**2 + max(cp.abs(y0) - dy, CP_ZERO)**2)

    diam = 2 * cp.sqrt(dx**2 + dy**2)
    xm = dx + cp.abs(x0)
    ym = dy + cp.abs(y0)
    rmax_r = cp.sqrt(xm**2 + ym**2) - diam / R_MAX_NUDGE

    rmin = max(rmin_g, rmin_r)
    rmax = min(rmax_g, rmax_r)

    skip = rmin >= rmax

    if thread_idx == 0:
        dbg_scalars[out_idx, 0] = CP_FLOAT(1) if skip else CP_FLOAT(0)
        dbg_scalars[out_idx, 1] = rmin
        dbg_scalars[out_idx, 2] = rmax
        dbg_scalars[out_idx, 5] = rmin_g
        dbg_scalars[out_idx, 6] = rmax_g
        dbg_scalars[out_idx, 7] = rmin_r
        dbg_scalars[out_idx, 8] = rmax_r
    if not skip:
        r_inside = max(CP_ZERO, min(dx - cp.abs(x0), dy - cp.abs(y0)))

        n_rings = max(N_RINGS_MIN, CP_UINT(MAX_RINGS * (rmax - rmin) / diam))

        dr = (rmax - rmin) / n_rings

        rings = TMP_FLOAT_ARRAY
        phi_cur = TMP_FLOAT_ARRAY
        if thread_idx < n_rings:
            phi_cur[RINGS_SIZE + 2 * thread_idx + 0] = -INVAL
            phi_cur[RINGS_SIZE + 2 * thread_idx + 1] = INVAL

            r_idx = thread_idx
            r = rmin + dr * (CP_FLOAT(r_idx) + CP_FLOAT(0.5))

            n_arcs = CP_UINT(0)

            if r < r_inside:
                rings[r_idx * RING_STRIDE + 0] = CP_ONE
                rings[r_idx * RING_STRIDE + 1] = CP_ZERO
                rings[r_idx * RING_STRIDE + 2] = CP_TWO_PI

            else:
                for q_idx in jit.range(4):
                    sin_pos = CP_UINT((q_idx // 2))
                    cos_pos = CP_UINT(((q_idx + 1) // 2) % 2)

                    sin_sign = CP_INT(2 * sin_pos - 1)
                    cos_sign = CP_INT(2 * cos_pos - 1)

                    cos_0 = (dx - x0) / r
                    sin_0 = cp.sqrt(CP_ONE - cos_0**2)

                    cos_1 = (-dx - x0) / r
                    sin_1 = cp.sqrt(CP_ONE - cos_1**2)

                    sin_2 = (dy - y0) / r
                    cos_2 = cp.sqrt(CP_ONE - sin_2**2)

                    sin_3 = (-dy - y0) / r
                    cos_3 = cp.sqrt(CP_ONE - sin_3**2)

                    if cos_sign * cos_0 > 0 and cp.abs(y0 + r * sin_0 * sin_sign) < dy:
                        phi_cur[RINGS_SIZE + 2 * thread_idx + (1 - sin_pos)] = cp.arctan2(sin_0 * sin_sign, cos_0)

                    if cos_sign * cos_1 > 0 and cp.abs(y0 + r * sin_1 * sin_sign) < dy:
                        phi_cur[RINGS_SIZE + 2 * thread_idx + (sin_pos)] = cp.arctan2(sin_1 * sin_sign, cos_1)

                    if sin_sign * sin_2 > 0 and cp.abs(x0 + r * cos_2 * cos_sign) < dx:
                        phi_cur[RINGS_SIZE + 2 * thread_idx + (cos_pos)] = cp.arctan2(sin_2, cos_2 * cos_sign)

                    if sin_sign * sin_3 > 0 and cp.abs(x0 + r * cos_3 * cos_sign) < dx:
                        phi_cur[RINGS_SIZE + 2 * thread_idx + (1 - cos_pos)] = cp.arctan2(sin_3, cos_3 * cos_sign)

                    if phi_cur[RINGS_SIZE + 2 * thread_idx + 1] < 1000.:
                        rings[r_idx * RING_STRIDE + 1 + 2 * n_arcs + 0] = phi_cur[RINGS_SIZE + 2 * thread_idx + 0]
                        rings[r_idx * RING_STRIDE + 1 + 2 * n_arcs + 1] = phi_cur[RINGS_SIZE + 2 * thread_idx + 1]

                        phi_cur[RINGS_SIZE + 2 * thread_idx + 0] = -INVAL
                        phi_cur[RINGS_SIZE + 2 * thread_idx + 1] = INVAL

                        n_arcs += CP_UINT(1)

                if phi_cur[RINGS_SIZE + 2 * thread_idx + 0] > -1000. and rings[r_idx * RING_STRIDE + 1] < -1000.:
                    rings[r_idx * RING_STRIDE + 1] = phi_cur[RINGS_SIZE + 2 * thread_idx + 0] - CP_TWO_PI

                rings[r_idx * RING_STRIDE + 0] = CP_FLOAT(n_arcs)

    jit.syncthreads()

    if not skip:
        n_arcs = CP_UINT(0)
        if thread_idx == 0:
            for i in jit.range(CP_INT(n_rings)):
                n_ring_arcs = CP_INT(rings[i * RING_STRIDE + 0])
                for j in jit.range(n_ring_arcs):
                    if n_arcs < MAX_ARCS:  # guard: drop excess arcs instead of overflowing shared memory
                        arcs[n_arcs * ARC_STRIDE + 0] = rmin + dr * (CP_FLOAT(i) + CP_FLOAT(0.5))
                        arcs[n_arcs * ARC_STRIDE + 1] = rings[i * RING_STRIDE + 1 + 2 * j + 0]
                        arcs[n_arcs * ARC_STRIDE + 2] = rings[i * RING_STRIDE + 1 + 2 * j + 1]
                    n_arcs += CP_UINT(1)
            n_arcs_shared[0] = min(n_arcs, CP_UINT(MAX_ARCS))

    jit.syncthreads()

    if not skip:
        n_arcs = n_arcs_shared[0]
        if thread_idx == 0:
            dbg_scalars[out_idx, 3] = CP_FLOAT(n_arcs)

        cell_weights = cum_cell_weights

        weights_size = n_arcs * PHI_EDGES
        weights_repeat = CP_INT((weights_size + X_THREADS - 1) // X_THREADS)

        for i in jit.range(weights_repeat):
            sample_idx = CP_UINT(i * X_THREADS) + thread_idx
            phi_idx = sample_idx % PHI_EDGES
            arc_idx = sample_idx // PHI_EDGES

            if arc_idx < n_arcs and phi_idx < PHI_CELLS:
                r = arcs[arc_idx * ARC_STRIDE + 0]
                phi_min = arcs[arc_idx * ARC_STRIDE + 1]
                phi_max = arcs[arc_idx * ARC_STRIDE + 2]

                phi = phi_min + ((phi_idx + CP_ONE / 2) / PHI_CELLS) * (phi_max - phi_min)
                x = x0 + r * cp.cos(phi)
                y = y0 + r * cp.sin(phi)

                w = CP_ZERO
                if (x > theta_x_min and x < theta_x_min + theta_x_width * n_theta_x and
                        y > theta_y_min and y < theta_y_min + theta_y_width * n_theta_y):
                    xi = min(CP_UINT(n_theta_x - 1), CP_UINT(cp.floor((x - theta_x_min) / theta_x_width)))
                    yj = min(CP_UINT(n_theta_y - 1), CP_UINT(cp.floor((y - theta_y_min) / theta_y_width)))
                    # proposal density: precomputed (theta_x, theta_y) marginal of H, summed
                    # over gamma and a0 on the host -- see module docstring. No resonance
                    # condition here; the final evaluation reweights exactly regardless of
                    # what density placed the samples.
                    w = H_marginal[xi, yj]

                dphi_cell = (phi_max - phi_min) / PHI_CELLS
                cell_weights[sample_idx] = w * dphi_cell * r

        jit.syncthreads()

        if thread_idx < n_arcs:
            total = CP_ZERO
            tmp = CP_ZERO
            for i in jit.range(PHI_EDGES):
                tmp = cell_weights[thread_idx * PHI_EDGES + CP_UINT(i)]
                cum_cell_weights[thread_idx * PHI_EDGES + CP_UINT(i)] = total
                total += tmp
        jit.syncthreads()

        if thread_idx == 0:
            TMP_FLOAT_ARRAY[0] = CP_ZERO
            for i in jit.range(CP_INT(n_arcs)):
                TMP_FLOAT_ARRAY[0] += cum_cell_weights[CP_UINT(i * PHI_EDGES) + (PHI_EDGES - 1)]
        jit.syncthreads()

        total_weight = TMP_FLOAT_ARRAY[0]
        if thread_idx == 0:
            dbg_scalars[out_idx, 4] = total_weight
        thread_samples[thread_idx * THREAD_STRIDE] = CP_UINT(0)
        if thread_idx == 0:
            cur_thread = CP_UINT(0)
            for k in jit.range(CP_INT(n_arcs)):
                arc_weight = cum_cell_weights[k * PHI_EDGES + (PHI_EDGES - 1)]
                s_add = CP_UINT(cp.floor(SAMPLES_TOTAL * arc_weight / total_weight))
                for j in jit.range(CP_INT(s_add)):
                    n_samples = thread_samples[cur_thread * THREAD_STRIDE + 0]
                    thread_samples[cur_thread * THREAD_STRIDE + 1 + 3 * n_samples + 0] = CP_UINT(k)
                    thread_samples[cur_thread * THREAD_STRIDE + 1 + 3 * n_samples + 1] = CP_UINT(j)
                    thread_samples[cur_thread * THREAD_STRIDE + 1 + 3 * n_samples + 2] = CP_UINT(s_add)
                    thread_samples[cur_thread * THREAD_STRIDE + 0] += CP_UINT(1)
                    cur_thread = (cur_thread + CP_UINT(1)) % X_THREADS
        jit.syncthreads()

        for arc_idx in jit.range(n_arcs):
            phi_min = arcs[arc_idx * ARC_STRIDE + 1]
            phi_max = arcs[arc_idx * ARC_STRIDE + 2]
            dphi = (phi_max - phi_min) / PHI_CELLS
            for k in jit.range(CDF_PHI_REPEAT):
                r_idx = CP_UINT(k * X_THREADS) + thread_idx
                if r_idx < CDF_PHI_RESOLUTION:
                    r = cum_cell_weights[arc_idx * PHI_EDGES + (PHI_EDGES - 1)] * r_idx / (CDF_PHI_RESOLUTION - 1)
                    left = CP_UINT(0)
                    right = CP_UINT(PHI_EDGES - 1)
                    while right - left > 1:
                        mid = (left + right) // 2
                        if cum_cell_weights[arc_idx * PHI_EDGES + mid] <= r:
                            left = mid
                        else:
                            right = mid

                    cdf_i = cum_cell_weights[arc_idx * PHI_EDGES + (left + 0)]
                    cdf_ip1 = cum_cell_weights[arc_idx * PHI_EDGES + (left + 1)]
                    cdf_span = cdf_ip1 - cdf_i
                    # H can have exact-zero cells (finite-particle deposition), unlike the
                    # legacy smooth analytic `collision`; a run of zero-weight cells makes
                    # cdf_span 0 at the boundary sample. Fall back to the cell's left edge
                    # rather than dividing by zero -- this sample carries no real density
                    # here anyway, and gets zero weight below via the cell_weight guard.
                    fac = CP_ZERO
                    if cdf_span > CP_ZERO:
                        fac = (r - cdf_i) / cdf_span
                    inv_cdf[arc_idx * CDF_PHI_RESOLUTION + r_idx] = phi_min + (CP_FLOAT(left) + fac) * dphi
        jit.syncthreads()

        f_tot = CP_ZERO
        n_thread_samples = thread_samples[thread_idx * THREAD_STRIDE + 0]
        for thread_sample_idx in jit.range(CP_UINT(SAMPLES_REPEAT)):
            if thread_sample_idx < n_thread_samples:
                arc_idx = thread_samples[thread_idx * THREAD_STRIDE + 1 + 3 * thread_sample_idx + 0]
                arc_sample_idx = thread_samples[thread_idx * THREAD_STRIDE + 1 + 3 * thread_sample_idx + 1]
                n_arc_samples = thread_samples[thread_idx * THREAD_STRIDE + 1 + 3 * thread_sample_idx + 2]

                arc_r = arcs[arc_idx * ARC_STRIDE + 0]
                phi_min = arcs[arc_idx * ARC_STRIDE + 1]
                phi_max = arcs[arc_idx * ARC_STRIDE + 2]
                arc_total_weight = cum_cell_weights[arc_idx * PHI_EDGES + (PHI_EDGES - 1)]
                dphi_cell = (phi_max - phi_min) / PHI_CELLS
                arc_area = dphi_cell * arc_r * dr  # note: dphi_cell, not (phi_max-phi_min) -- see module docstring

                for di in jit.range(subsampling):
                    subsample_idx = arc_sample_idx * subsampling + di

                    reg = (subsample_idx + 0.5) / n_arc_samples / subsampling
                    fib = cp.remainder(subsample_idx * GOLDEN_PHI, 1.0)

                    theta_min = arc_r - dr / 2
                    theta_max = theta_min + dr

                    theta_sq = theta_min**2 + fib * (theta_max**2 - theta_min**2)
                    theta = cp.sqrt(theta_sq)

                    il = CP_UINT(cp.floor(reg * (CDF_PHI_RESOLUTION - 1)))
                    fac = reg * (CDF_PHI_RESOLUTION - 1) - CP_FLOAT(il)
                    phi = inv_cdf[arc_idx * CDF_PHI_RESOLUTION + il] * (CP_ONE - fac) + inv_cdf[arc_idx * CDF_PHI_RESOLUTION + (il + 1)] * fac

                    phi_idx = min(PHI_CELLS - 1, CP_UINT(PHI_CELLS * (phi - phi_min) / (phi_max - phi_min)))
                    cell_weight = cum_cell_weights[arc_idx * PHI_EDGES + phi_idx + 1] - cum_cell_weights[arc_idx * PHI_EDGES + phi_idx]
                    # A sample can land in a zero-weight cell (see inv_cdf comment above);
                    # it carries no density there, so contribute nothing rather than 1/0.
                    sample_area = CP_ZERO
                    if cell_weight > CP_ZERO:
                        sample_area = arc_area / n_arc_samples / subsampling * arc_total_weight / cell_weight

                    x = x0 + theta * cp.cos(phi)
                    y = y0 + theta * cp.sin(phi)

                    if (x > theta_x_min and x < theta_x_min + theta_x_width * n_theta_x and
                            y > theta_y_min and y < theta_y_min + theta_y_width * n_theta_y):

                        cos_pol = cp.cos(phi_pol - phi)**2

                        Xf = (x - theta_x_min) / theta_x_width - CP_FLOAT(0.5)
                        Yf = (y - theta_y_min) / theta_y_width - CP_FLOAT(0.5)
                        xi2 = CP_INT(cp.floor(Xf))
                        yj2 = CP_INT(cp.floor(Yf))
                        xw = Xf - CP_FLOAT(xi2)
                        yw = Yf - CP_FLOAT(yj2)
                        xi2 = min(max(xi2, CP_INT(0)), CP_INT(n_theta_x - 2))
                        yj2 = min(max(yj2, CP_INT(0)), CP_INT(n_theta_y - 2))

                        # each a0 bin resonates at its own gamma (eq. "Gamma"), with its own
                        # Jacobian factor 1/(1+a0) in the prefactor (eq. "Fmatrix") -- both g
                        # and the gamma-axis interpolation must be recomputed per a0 bin, not
                        # shared across the quadrature the way x/y/theta_sq are.
                        inv_base = CP_ONE / s - theta_sq
                        h_sum = CP_ZERO
                        if inv_base > CP_ZERO:
                            for ai2 in jit.range(CP_INT(n_a0)):
                                a0_val = a0_min + (CP_FLOAT(ai2) + CP_FLOAT(0.5)) * a0_width
                                g_sq = (CP_ONE + a0_val) / inv_base
                                g = cp.sqrt(g_sq)

                                if g > gamma_min and g < gamma_min + gamma_width * n_gamma:
                                    gth_sq_inv = CP_ONE / (CP_ONE + theta_sq * g_sq)**2
                                    a_fac = CP_ONE - 4 * cos_pol * theta_sq * g_sq * gth_sq_inv
                                    prefac = a_fac * g**5 * gth_sq_inv / (CP_ONE + a0_val)

                                    Gf = (g - gamma_min) / gamma_width - CP_FLOAT(0.5)
                                    gi2 = CP_INT(cp.floor(Gf))
                                    gw = Gf - CP_FLOAT(gi2)
                                    gi2 = min(max(gi2, CP_INT(0)), CP_INT(n_gamma - 2))

                                    h000 = H[gi2, xi2, yj2, ai2]
                                    h100 = H[gi2 + 1, xi2, yj2, ai2]
                                    h010 = H[gi2, xi2 + 1, yj2, ai2]
                                    h110 = H[gi2 + 1, xi2 + 1, yj2, ai2]
                                    h001 = H[gi2, xi2, yj2 + 1, ai2]
                                    h101 = H[gi2 + 1, xi2, yj2 + 1, ai2]
                                    h011 = H[gi2, xi2 + 1, yj2 + 1, ai2]
                                    h111 = H[gi2 + 1, xi2 + 1, yj2 + 1, ai2]

                                    h_yj = (h000 * (CP_ONE - xw) + h010 * xw) * (CP_ONE - yw) \
                                         + (h001 * (CP_ONE - xw) + h011 * xw) * yw
                                    h_yj1 = (h100 * (CP_ONE - xw) + h110 * xw) * (CP_ONE - yw) \
                                          + (h101 * (CP_ONE - xw) + h111 * xw) * yw
                                    h_val = h_yj * (CP_ONE - gw) + h_yj1 * gw

                                    h_sum += h_val * a0_width * prefac

                        f = h_sum
                        f_tot += f * sample_area

                        if debug_idx == out_idx:
                            debug_arr[arc_idx, subsample_idx, 0] = x
                            debug_arr[arc_idx, subsample_idx, 1] = y
                            debug_arr[arc_idx, subsample_idx, 2] = f

        jit.atomic_add(output, out_idx, f_tot / s**2)


def calculate_angular_spectrum_4d(table, s, theta_x, theta_y, phi_pol,
                                   samples_per_point=32, debug_idx=0):
    """Host-side driver for spectrum_kernel_4d, the Stage-2 analogue of
    Compton.calculate_angular_spectrum. `table` is a deposition.Table (H
    plus grid metadata and gamma_bracket) built by particles.py/deposition.py.

    Returns (spectrum, elapsed_seconds, debug), matching
    calculate_angular_spectrum's return shape.
    """
    if table.grid.shape[3] > N_A0_MAX:
        raise ValueError(f"table has {table.grid.shape[3]} a0 bins; this kernel loops over "
                          f"all of them per sample and is only sized/tested up to {N_A0_MAX}")

    coef = 1.5  # pure numerical constant, eq. "main"/"Fmatrix" -- see module docstring

    params = cp.stack(cp.meshgrid(theta_x, theta_y, s, indexing='ij'), 3).reshape(-1, 3).astype(CP_FLOAT)
    grid_x = theta_x.size * theta_y.size * s.size

    grid = table.grid
    gamma_min, theta_x_min, theta_y_min, a0_min = (CP_FLOAT(e[0]) for e in
        (grid.gamma_edges, grid.theta_x_edges, grid.theta_y_edges, grid.a0_edges))
    gamma_width, theta_x_width, theta_y_width, a0_width = (CP_FLOAT(w) for w in grid.widths)
    n_gamma, n_theta_x, n_theta_y, n_a0 = (CP_UINT(n) for n in grid.shape)

    gamma_lo = CP_FLOAT(table.gamma_bracket[0])
    gamma_hi = CP_FLOAT(table.gamma_bracket[1])

    dx = CP_FLOAT(max(abs(float(grid.theta_x_edges[0])), abs(float(grid.theta_x_edges[-1]))))
    dy = CP_FLOAT(max(abs(float(grid.theta_y_edges[0])), abs(float(grid.theta_y_edges[-1]))))

    H_gpu = cp.asarray(table.H, dtype=CP_FLOAT)
    H_marginal_gpu = H_gpu.sum(axis=(0, 3))  # (n_theta_x, n_theta_y), summed over gamma and a0 -- coarse proposal

    debug = cp.zeros((MAX_ARCS, SAMPLES_TOTAL * samples_per_point, 3), dtype=CP_FLOAT) * cp.nan
    spec = cp.zeros((grid_x,), dtype=CP_FLOAT)
    dbg_scalars = cp.zeros((grid_x, 9), dtype=CP_FLOAT)  # per-output-point skip/rmin/rmax/n_arcs/total_weight, diagnostic only

    start = cp.cuda.Event()
    finish = cp.cuda.Event()
    start.record()
    spectrum_kernel_4d[grid_x, X_THREADS](
        spec, params, H_gpu, H_marginal_gpu,
        gamma_min, gamma_width, n_gamma,
        theta_x_min, theta_x_width, n_theta_x,
        theta_y_min, theta_y_width, n_theta_y,
        a0_min, a0_width, n_a0,
        gamma_lo, gamma_hi, dx, dy,
        CP_FLOAT(phi_pol), CP_UINT(samples_per_point), debug, CP_UINT(debug_idx), dbg_scalars)
    finish.record()
    finish.synchronize()
    dt = cp.cuda.get_elapsed_time(start, finish) * 1e-3

    return (coef * spec).reshape((theta_x.size, theta_y.size, s.size)).get(), dt, debug
