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
    (theta_x, theta_y, s) spectrum. RESOLVED, see below.
  - direct_binning_spectrum: iterates real macroparticles, computes each
    one's resonance frequency for a fixed observation direction, and bins
    with its weight. No table, no quadrature grid at all. Intended as the
    primary correctness test for correlated bunches (plan.md validation 3)
    and to be kept permanently as a debug tool. Normalisation root-caused
    this session, see below -- a real, systematic ~2*pi residual remains,
    deliberately not chased yet.

WHAT'S VALIDATED:

1. angle_integrated_spectrum, run on real Stage 0/1 output, matches
   calculate_spectrum(s, gamma0, sigma_gamma0) to within 0.5-3% across the
   whole resonance peak (worse only in the low-statistics tail), *after*
   correcting for a units mismatch: calculate_spectrum returns dN/dE
   (matches its use in calculate-spec-ang.py, axis label "dN/dE, MeV^-1"),
   not dN/ds. dE = 4*Wph*ds (see calculate-spec-ang.py's
   `s_scale = 4*compton.Wph`), so compare as `dN_ds_mine` vs
   `calculate_spectrum(...) * 4 * compton.Wph`. Validates Stage 0/1's gamma
   axis and overall weight normalisation, independent of collision/H lookup
   or any theta quadrature.

2. spectrum_from_table's per-solid-angle normalisation. Root cause: its
   grid quadrature (Sum f(theta) * dtheta_x*dtheta_y, a plain Riemann sum)
   is missing a factor of PHI_CELLS (=PHI_EDGES-1=31) relative to what
   spectrum_kernel's own sample_area convention effectively computes.
   Traced algebraically: spectrum_kernel builds a per-phi-cell coarse
   weight `cell_weights[j] = w_j * (phi_max-phi_min) * r` -- using the
   *full* arc angular width in every cell's weight, not that cell's own
   width `dphi_cell = (phi_max-phi_min)/PHI_CELLS`. For self-normalised
   importance sampling with a continuous inverse-CDF draw, the correct
   per-sample correction is `dphi_cell * arc_total_weight / cell_weight_j`;
   the code computes `(phi_max-phi_min) * arc_total_weight / cell_weight_j`,
   i.e. exactly PHI_CELLS times that, *unconditionally* (the ratio
   (phi_max-phi_min)/dphi_cell = PHI_CELLS regardless of the weight
   distribution). Confirmed empirically: multiplying spectrum_from_table's
   coef by PHI_CELLS brings it to within 1-5% of calculate_angular_spectrum
   across 7 configurations spanning aspect ratio 1:1 to 100:1, on-axis and
   off-axis observation points, and on- and off-peak s -- consistent with
   ordinary discretisation/interpolation differences between a plain grid
   sum and the kernel's QMC quadrature, not a remaining scale error.
   (A very close alternative constant, pi**3 = 31.006, fits the same data
   equally well within the ~1% residual and can't be distinguished from
   PHI_CELLS=31 at this precision -- but PHI_CELLS has an actual derivation
   above, and pi**3 doesn't, so that's the constant used here.)

   NOTE this is a bug in spectrum_kernel itself (`sample_area` in the final
   evaluation loop, core.py, needs `* dphi_cell` where dphi_cell =
   (phi_max-phi_min)/PHI_CELLS, or equivalently `/ PHI_CELLS` if the
   `(phi_max-phi_min)` in `cell_weights`'s definition is left as-is). It was
   *not* fixed there for this branch: patching it directly was tried and
   broke calculate_angular_spectrum's agreement with calculate_spectrum
   (which is unaffected by this bug -- see point 1 -- because it doesn't
   use spectrum_kernel at all), consistent with `coef =
   3/(4*pi**4*Wph*4)` having been empirically tuned against the buggy
   sample_area rather than derived independently; touching one without the
   other breaks calibration. Fixing both together, if desired, is future
   work -- flag it to the user rather than doing it unprompted.

3. direct_binning_spectrum's normalisation. Root cause (found this session,
   via a from-scratch derivation from Paper/xigma.tex "eq:xsec" rather than
   guesswork): it was using the *ensemble-collapsed* prefactor (`g**5`,
   Paper's "eq:Fmatrix" -- which bakes in a `|dGamma/domega|` Jacobian
   meant for looking up a smooth, already-binned H) directly on raw,
   un-binned macroparticles -- a Jacobian that doesn't apply there, since no
   ensemble collapse has happened. The correct single-electron form is
   "eq:xsec" itself: `g**2 * gth_sq_inv`, prefactor 3 (not the `Wph`/`pi**4`
   `coef` used by the other two functions -- that belongs to their
   H-density/omega-unit conversion, not to a raw-particle histogram), and
   the `domega -> ds` Jacobian (`domega = 4*omega_L*ds`, a *constant*) cancels
   exactly against the same factor converting the histogram's bin width, so
   the result needs no further `/ s**2` division at all (the previous code's
   `/ s_centers**2` was carried over from spectrum_from_table's convention by
   mistake and was the dominant part of the reported gap -- `s` is O(gamma0^2)
   in typical configurations, so a spurious `1/s**2` alone accounts for the
   bulk of "~3000-4000x", though the precise historical number came from a
   different (single-point vs angle-integrated) comparison methodology and
   wasn't reproduced exactly).

   REMAINING, DELIBERATELY DEFERRED: even with the fix above,
   direct_binning_spectrum's angle-integrated total (Riemann-summed over a
   grid of (x0, y0) weighted by cell area) is consistently ~6.3x
   angle_integrated_spectrum's output, suspiciously close to 2*pi, small
   spread across configurations (systematic, not noise). Not yet explained;
   flagged to the user, not being chased in this pass.

   a0/ahat resonance term: NOW INCLUDED (s_res = g**2/(1+a0+g**2*r_sq)),
   with no extra Jacobian in the prefactor -- see the function's own
   docstring for why that differs from spectrum_from_table/
   spectrum_kernel_4d. Verified empirically: this was the actual cause of
   the s-dependence seen when comparing against spectrum_from_table at
   a0~0.3 (a ~100x swing across one spectral peak); with a0 included the
   ratio is flat (~6% spread) at any a0, leaving only the still-open ~2*pi
   offset above.

Stage 0/1 (particles.py, deposition.py) are validated independently of both
issues above -- table total weight/theta marginal/gamma marginal matching
calculate_total()/calculate_intersection to 1-3%, and angle_integrated_spectrum
matching calculate_spectrum to 0.5-3%.
"""
import numpy as np

PHI_CELLS = 31  # PHI_EDGES - 1 in core.py; see module docstring for derivation


def _xp_for(backend):
    """Resolves backend='numpy'|'cupy' to its array module. cupy is imported
    lazily so this module still imports fine without it installed unless
    backend='cupy' is actually requested -- same convention as
    particles.push_and_sample.
    """
    if backend == 'numpy':
        return np
    if backend == 'cupy':
        import cupy as cp
        return cp
    raise ValueError(f"backend must be 'numpy' or 'cupy', got {backend!r}")


def angle_integrated_spectrum(gamma, particle_weight, s, backend='numpy'):
    """dN/ds integrated over all emission solid angle, from real Stage 0/1
    macroparticles. A single electron's angle-integrated spectral shape
    depends only on its own gamma (not its transverse angle), via the
    standard Compton edge formula also used -- in a different
    parametrisation -- by calculate_spectrum. Compare against
    `calculate_spectrum(s, gamma0, sigma_gamma0) * 4 * compton.Wph` (unit
    conversion from dN/dE to dN/ds, see module docstring).

    gamma, particle_weight: 1D arrays, one entry per macroparticle -- e.g.
    the gamma and weight arrays push_and_sample already returns (one row per
    particle; no external per-particle summing needed).
    s: scalar or 1D array of normalised photon energies.
    backend: 'numpy' (default) or 'cupy' -- array-module-agnostic, same
    pattern as deposition.py/particles.py. gamma/particle_weight/s are
    converted to the target module if not already; the whole computation is
    elementwise/reduction, so there's nothing GPU-specific to write.
    """
    xp = _xp_for(backend)
    gamma, particle_weight = xp.asarray(gamma), xp.asarray(particle_weight)
    s_arr = xp.atleast_1d(xp.asarray(s, dtype=xp.float64))
    gamma = gamma[:, None]
    y = s_arr[None, :] / gamma**2
    shape = 1.5 * (1.0 - 2.0 * y * (1.0 - y))
    shape = xp.where((y < 0) | (y > 1), 0.0, shape)
    out = xp.sum(particle_weight[:, None] * shape / gamma**2, axis=0)
    return out if np.ndim(s) else out[0]


def _interp4d(H, grid, gamma, theta_x, theta_y, a0, xp):
    """Array-module-agnostic core of interp4d: takes H already converted to
    xp's module (so callers looping over many query batches against the same
    table -- e.g. spectrum_from_table's loop over s -- transfer H once
    instead of once per call).
    """
    axes_edges = (grid.gamma_edges, grid.theta_x_edges, grid.theta_y_edges, grid.a0_edges)
    coords = [xp.asarray(c, dtype=xp.float64) for c in (gamma, theta_x, theta_y, a0)]
    shape = H.shape

    out_shape = np.broadcast_shapes(*(c.shape for c in coords))
    coords = [xp.broadcast_to(c, out_shape) for c in coords]

    f = xp.zeros(out_shape, dtype=xp.float64)
    in_range = xp.ones(out_shape, dtype=bool)

    i0s, ws = [], []
    for edges, x, n in zip(axes_edges, coords, shape):
        edges = xp.asarray(edges)
        width = edges[1] - edges[0]
        centers0 = edges[0] + 0.5 * width  # centre of first cell
        f_idx = (x - centers0) / width
        i0 = xp.floor(f_idx).astype(xp.int64)
        w = f_idx - i0
        in_range &= (i0 >= -1) & (i0 < n)  # allow i0==-1/n-1 edge cases below via clipping + zero-weight
        i0s.append(i0)
        ws.append(w)

    for dg in (0, 1):
        for dtx in (0, 1):
            for dty in (0, 1):
                for da in (0, 1):
                    idxs = []
                    corner_w = xp.ones(out_shape, dtype=xp.float64)
                    valid = xp.ones(out_shape, dtype=bool)
                    for (i0, w, d, n) in zip(i0s, ws, (dg, dtx, dty, da), shape):
                        ci = i0 + d
                        valid &= (ci >= 0) & (ci < n)
                        corner_w = corner_w * (w if d else (1 - w))
                        idxs.append(xp.clip(ci, 0, n - 1))
                    take = valid & in_range
                    if bool(xp.any(take)):
                        vals = H[idxs[0][take], idxs[1][take], idxs[2][take], idxs[3][take]]
                        f[take] += vals * corner_w[take]

    f[~in_range] = 0.0
    return f


def interp4d(table, gamma, theta_x, theta_y, a0, backend='numpy'):
    """Quadrilinear interpolation of table.H at query points (arrays of equal
    shape). Points outside the tabulated extent return 0.

    backend: 'numpy' (default) or 'cupy' -- table.H is always host (numpy),
    per deposition.Table's invariant, and is transferred to the target
    module here. For repeated calls against the same table (e.g. inside a
    loop), prefer calling _interp4d directly with a pre-transferred H
    instead of re-transferring on every call -- see spectrum_from_table.
    """
    xp = _xp_for(backend)
    H = xp.asarray(table.H)
    return _interp4d(H, table.grid, gamma, theta_x, theta_y, a0, xp)


def spectrum_from_table(table, compton, x0, y0, s, phi_pol, backend='numpy'):
    """Brute-force quadrature of dN/(ds dOmega) at a single observation point
    (x0, y0) over a grid of frequencies s, integrating the table over its
    full (theta_x, theta_y, a0) extent at each s.

    x0, y0, s: floats / 1D array for s. Returns array matching s's shape.

    Resonance condition includes a0 (Paper/xigma.tex eq. "Gamma", section
    "Reduction to three dimensions"): g**2 = (1+a0) / (1/s - r_sq), each a0
    bin resonating at its own gamma, with a Jacobian factor 1/(1+a0) in the
    prefactor (eq. "jacobian", "Fmatrix"). An earlier version of this
    function used g**2 = 1/(1/s - r_sq) (a0-independent) with no 1/(1+a0)
    factor -- the same gap spectrum_kernel_4d had; see CLAUDE.md "Known
    bugs"/"Traps". Fixed alongside that kernel.

    Previously validated against calculate_angular_spectrum to 1-5% across a
    range of aspect ratios / observation points / frequencies, but that
    predates the a0/Jacobian fix above and needs re-measuring -- see module
    docstring for the (unrelated, still valid) PHI_CELLS correction this
    also relies on.

    backend: 'numpy' (default) or 'cupy'. table.H is transferred to the
    target module once up front (not once per s, and not once per interp4d
    call inside the loop) -- see _interp4d's docstring for why that matters.
    The loop over s stays a host Python loop (s is typically a small grid);
    each iteration's (theta_x, theta_y, a0) quadrature -- the actual
    O(grid_size) work -- runs on the target device.
    """
    xp = _xp_for(backend)
    H = xp.asarray(table.H)

    theta_x_c, theta_y_c, a0_c = (xp.asarray(c) for c in
                                   (table.grid.centers[1], table.grid.centers[2], table.grid.centers[3]))
    TX, TY, A0 = xp.meshgrid(theta_x_c, theta_y_c, a0_c, indexing='ij')
    r_sq = (TX - x0)**2 + (TY - y0)**2

    dtx, dty, da = table.grid.widths[1], table.grid.widths[2], table.grid.widths[3]
    cell_vol = dtx * dty * da

    cos_pol = xp.cos(phi_pol - xp.arctan2(TY - y0, TX - x0))**2

    coef = 3.0 / (4.0 * np.pi**4 * compton.Wph * 4.0) * PHI_CELLS

    s_arr = np.atleast_1d(np.asarray(s, dtype=np.float64))
    out = np.zeros_like(s_arr)
    for k, sk in enumerate(s_arr):
        inv_base = 1.0 / sk - r_sq
        g_sq = xp.where(inv_base > 0, (1.0 + A0) / xp.where(inv_base > 0, inv_base, 1.0), -1.0)
        valid = g_sq >= 0
        if not bool(xp.any(valid)):
            continue
        g = xp.sqrt(xp.where(valid, g_sq, 0.0))
        gth_sq_inv = 1.0 / (1.0 + r_sq * g_sq)**2
        a_fac = 1.0 - 4.0 * cos_pol * r_sq * g_sq * gth_sq_inv
        prefac = a_fac * g**5 * gth_sq_inv / (1.0 + A0)

        H_val = _interp4d(H, table.grid, g, TX, TY, A0, xp)

        f = xp.where(valid, H_val * prefac, 0.0)
        out[k] = float(coef * f.sum() * cell_vol / sk**2)

    return out if np.ndim(s) else out[0]


def direct_binning_spectrum(gamma, theta_x, theta_y, particle_weight, a0,
                             x0, y0, s_edges, phi_pol, backend='numpy'):
    """Reference path: for each real macroparticle, compute the photon
    energy it resonates at when viewed from (x0, y0), and bin its weight
    into the s_edges histogram. No table, no importance sampling --
    assumption-free on both the deposition and the lookup.

    Normalisation, root-caused this session (was previously an unexplained
    ~3000-4000x gap): this is a *single-electron*, not-yet-ensemble-collapsed
    quantity, so it must use Paper/xigma.tex eq. "xsec" (the bare
    differential cross-section, g**2 * gth_sq_inv prefactor) rather than
    eq. "Fmatrix" (g**5, used by spectrum_from_table/spectrum_kernel_4d --
    that g**3 extra power is a |dGamma/domega| Jacobian for evaluating a
    *smooth, already-binned* H, and double-applying it here, on raw discrete
    macroparticles, was the original bug). Converting eq. "xsec" to a photon
    count via the incident flux uses v_rel (=2c for near-backscattering, the
    same V_REL particles.py already bakes into `particle_weight`), not bare
    c: d3N_i/(domega dOmega) = 3 * particle_weight_i * g_i**2 * gth_sq_inv *
    a_fac * delta(omega - omega_R,i). Histogrammed over s (domega = 4*omega_L
    * ds, a *constant* Jacobian that cancels exactly against the same factor
    converting the histogram's bin width to ds) gives d3N/(ds dOmega) =
    [sum of weights in bin] / ds -- no additional 1/s**2, unlike
    spectrum_from_table's coef (that division belongs to the H-density/coef
    convention of the other two reference functions, not to this one; carrying
    it over here was the second half of the original bug).

    Known residual, deliberately not chased in this pass: even with the
    fix above, this function's angle-integrated total (Riemann-summed over a
    grid of (x0, y0), weighted by cell area) is consistently ~6.3x
    angle_integrated_spectrum's output -- suspiciously close to 2*pi, not
    yet explained. Small, systematic spread across configurations (not
    noise). Flag to the user before spending more time on it.

    a0/ahat resonance term (Paper/xigma.tex eq. "wRgamma"): s_res =
    g**2 / (1 + a0 + g**2*r_sq), i.e. the resonance condition shifts with
    a0 same as spectrum_from_table/spectrum_kernel_4d's g**2 =
    (1+a0)/(1/s - r_sq) (same relation, solved for s instead of g here).
    Unlike those two, NO extra 1/(1+a0) Jacobian is needed in the prefactor:
    that factor comes specifically from the *ensemble* gamma-integral
    collapse (Paper eq. "jacobian") those two methods perform when inverting
    the resonance condition to look up a smooth, pre-binned H at an
    interpolated gamma. This function never does that inversion -- each
    particle contributes at its own exact gamma_i, no lookup, no second
    collapse -- so that Jacobian doesn't apply here. Verified empirically:
    adding a0 to s_res alone (prefactor untouched) flattens the ratio against
    spectrum_from_table from a ~100x s-dependent swing (at a0~0.3, without
    this term) down to a ~6% flat spread, at the same overall (still open,
    ~2*pi-adjacent) offset described above.

    backend: 'numpy' (default) or 'cupy' -- array-module-agnostic; the
    per-particle arrays and s_edges are converted to the target module,
    including the histogram+weights reduction (cupy.histogram supports
    weights the same as numpy.histogram).
    """
    xp = _xp_for(backend)
    gamma, theta_x, theta_y, particle_weight, a0 = (
        xp.asarray(a) for a in (gamma, theta_x, theta_y, particle_weight, a0))

    r_sq = (theta_x - x0)**2 + (theta_y - y0)**2
    g = gamma
    s_res = g**2 / (1.0 + a0 + g**2 * r_sq)

    gth_sq_inv = 1.0 / (1.0 + r_sq * g**2)**2
    cos_pol = xp.cos(phi_pol - xp.arctan2(theta_y - y0, theta_x - x0))**2
    a_fac = 1.0 - 4.0 * cos_pol * r_sq * g**2 * gth_sq_inv

    prefactor = 3.0 * particle_weight * a_fac * g**2 * gth_sq_inv

    s_edges = xp.asarray(s_edges)
    hist, _ = xp.histogram(s_res, bins=s_edges, weights=prefactor)
    ds = xp.diff(s_edges)

    return hist / ds
