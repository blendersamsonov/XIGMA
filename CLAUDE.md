# CLAUDE.md

## What this is

CuPy/CUDA code for computing inverse Compton scattering spectra from an
electron bunch colliding with a laser pulse, in `src/xigma_i/`. Two
coexisting implementations:

- **Legacy, analytic-energy path** (`core.py`, ~680 lines): two
  `@jit.rawkernel()` GPU kernels (`particle_kernel`, `spectrum_kernel`) plus
  a `Compton` host class. Assumes the electron energy distribution
  factorises out of phase space and is Gaussian.
- **New, tabulated-energy path** (`particles.py`, `deposition.py`,
  `spectrum4d.py`, `reference.py`): replaces the Gaussian-energy assumption
  with a 4D overlap table built by particle deposition, carrying arbitrary
  correlations (chirp, divergence-energy, focusing). Implemented and
  validated against the legacy path; **not merged into `Compton`** and not
  the default -- it's a parallel, opt-in pipeline you drive directly through
  its own modules.

Physics reference is the accompanying paper. Both paths replace the
single-electron resonance function by a delta function, collapsing the
energy integral analytically; the legacy path reduces this to a 2D overlap
table (`intersection`) plus a 2D quadrature per output point, the new path
to a 4D overlap table (`H`) plus a 3D quadrature (`theta_x, theta_y, a0`)
per output point.

## Current state

- **Legacy path**: `calculate_total()`, `calculate_spectrum()`,
  `calculate_angular_spectrum()` are internally self-consistent (matched to
  2-3% against each other). `spectrum_kernel` has an open, *unfixed*
  normalisation bug (see "Known bugs") -- don't treat its absolute output as
  ground truth without accounting for it.
- **New path**: Stage 0 (particle source + ballistic pusher) and Stage 1
  (nearest + CIC deposition, CPU and GPU) are done and validated to 1-3%
  against the legacy path's `calculate_total()`/`calculate_intersection`,
  by a construction that needed zero hand-tuning of the derived
  normalisation constant. Stage 2 (`spectrum_kernel_4d`) is done. Its
  previous "validated to 6-8% against an independent brute-force reference"
  figure predated a normalisation bug (both `spectrum_kernel_4d` and
  `reference.spectrum_from_table` were using a `coef` copied from the
  legacy kernel's own, unrelated, empirically-tuned constant -- see "Known
  bugs" for the fix); re-measured post-fix
  (`compare_direct_vs_table.py --grid-integrate`) at a typical bunch
  config, all three of `spectrum_from_table`/`direct_binning_spectrum`/
  `spectrum_kernel_4d` now agree within ~15% (clustered around the
  still-open, deliberately-deferred `~2*pi` residual against
  `angle_integrated_spectrum` -- see `direct_vs_table_discrepancy_report.md`).
  One caveat found during that same re-check: in narrow-angle/sparse-table
  configs, `spectrum_kernel_4d` alone shows large, unstable variance
  (overshoots the other two by anywhere from ~3x to >30x depending on
  particle count/table resolution, sometimes with huge run-to-run std),
  while `spectrum_from_table`/`direct_binning_spectrum` still agree tightly
  with each other in the same configs. Reads as heavy-tailed
  importance-sampling noise from sparse/zero `H` cells in the kernel's own
  quadrature (see "Table too sparse" trap), not a normalisation error --
  flagged, not chased further. The two other
  blockers on validating correlated bunches (chirp, divergence-energy
  correlation) -- `reference.direct_binning_spectrum`'s normalisation and
  this `coef` bug -- are both now fixed.
- **Not done**: systematic resolution/convergence scans (tooling exists,
  see "Convergence testing" below -- no results recorded yet); chasing the
  narrow-angle `spectrum_kernel_4d` sampling-noise caveat above further; the
  legacy `spectrum_kernel` bug is diagnosed but deliberately not fixed on
  this branch (see "Known bugs" for why).

## Architecture: legacy path (`core.py`)

Two stages, in this order.

### Stage 1 -- `particle_kernel` -> `self.intersection`

Computes the geometric/luminosity factor: the electron-laser overlap
resolved over transverse *angle*, written to a 2D array `intersection[nx,
ny]` indexed by `(theta_x, theta_y)`.

Launch geometry: `particle_kernel[(particles_amount, nx*ny), N_STEPS]`.
Grid x = particle index, grid y = flattened angle-grid cell, threads = time
steps along the trajectory. Each thread handles one particle at one angle
at one time step and does an `atomic_add` into `intersection[xy_idx]`.

Per-particle work: straight-line propagation (no pusher -- velocities are
constant), evaluation of the Gaussian laser envelope with Rayleigh
diffraction, and accumulation of `f_cur = dt * weight / n_particles ×
(local photon density)`.

Note `n_particles = ceil(weight * particles.shape[0])`: the number of
particles actually used at an angle cell scales with that cell's weight
`f_th`, so low-weight cells cost less. Also `reg`/`fib` spread velocities
within a cell -- a sub-cell stratification, regular in x and golden-ratio
in y.

`self.time_envelope` accumulates the same quantity binned in time, for
diagnostics.

Normalisation happens on the host afterwards, in `calculate_intersection`:
the `coef` line applying `sigma_T`, `k0_las**2`, `v_rel = 2.0`, `N_e`,
`N_l`, and the sampling-domain corrections `z_weight * dsx * dsy`. Any
change to what the kernel deposits must be checked against this line.

### Stage 2 -- `spectrum_kernel` -> spectrum

One CUDA block per output point `(theta_x, theta_y, s)`, `X_THREADS = 128`
threads per block, flattened into `params_Arr`. `s` is the normalised
frequency; `g_sq = 1/(1/s - theta_sq)` inverts the resonance condition.

Phases within the block, separated by `jit.syncthreads()`:

1. **Radial bracketing.** `rmin_g`/`rmax_g` from inverting the resonance
   condition at `gamma0 ± 3*sigma_g`; `rmin_r`/`rmax_r` from the rectangle
   over which `intersection` is tabulated. Intersection gives the annulus.
   `skip = rmin >= rmax` discards output points with no support -- a large
   fraction of any grid, and the single biggest saving in the kernel.
2. **Ring/arc geometry.** The annulus is divided into `n_rings` rings; each
   is intersected analytically with the tabulation rectangle quadrant by
   quadrant, including the 4th↔1st quadrant merge. One thread per ring.
   Results are packed into shared `rings[]`, then serialised by thread 0
   into `arcs[]` (`ARC_STRIDE = 3`: r, phi_min, phi_max).
3. **Coarse weighting.** Each arc is subdivided into `PHI_CELLS` azimuthal
   cells; the integrand is evaluated cheaply at cell centres (nearest-cell
   lookup of `collision`, Gaussian `fr`, no polarisation factor) times the
   geometric measure `(phi_max - phi_min) * r`. This is the
   importance-sampling proposal, and the source of the "Known bugs" entry
   below -- that measure should be `dphi_cell = (phi_max-phi_min)/PHI_CELLS`,
   not the full arc width.
4. **Cumulative weights and sample allocation.** `cum_cell_weights` is a
   per-arc prefix sum. Thread 0 distributes `SAMPLES_TOTAL` samples across
   arcs in proportion to arc weight, round-robining across threads so work
   is balanced regardless of geometry.
5. **Inverse-CDF tabulation.** Per arc, binary search inverts the
   cumulative weights onto a uniform grid of `CDF_PHI_RESOLUTION` points in
   the CDF variable.
6. **Evaluation.** Each sample takes `phi` from the inverse CDF at a
   *regular* position `reg` (stratified, not random) and `theta` from a
   golden-ratio sequence `fib` within the ring's radial extent. The full
   integrand is then evaluated: bilinear interpolation of `collision`, the
   `g**5 * gth_sq_inv` prefactor, the polarisation factor `a_fac`, and the
   Gaussian `ffac`. Weighted by `sample_area = arc_area / n_arc_samples /
   subsampling * arc_total_weight / cell_weight` -- geometric measure over
   proposal density. Final `atomic_add(output, out_idx, f_tot / s**2)`.

### Host class `Compton`

Setters (`set_electron_parameters`, `set_laser_parameters`,
`set_foci_displacement`) each reset `self.intersection = None` to
invalidate the cache. `calculate_intersection` runs Stage 1;
`calculate_angular_spectrum` runs Stage 2 and lazily triggers Stage 1 if
needed. `calculate_spectrum` is a separate angle-integrated path that does
not use `spectrum_kernel` at all -- it returns **dN/dE**, not dN/ds
(`E = 4 * Wph * s`; see `reference.angle_integrated_spectrum`'s docstring
for the conversion, easy to get backwards). `estimate_yield` and
`estimate_spectrum_width` are cheap analytic estimates for
sanity-checking.

## Architecture: new path (`particles.py`, `deposition.py`, `spectrum4d.py`, `reference.py`)

Same delta-function-resonance idea, four axes instead of two: `H[gamma,
theta_x, theta_y, a0]` replaces `intersection[theta_x, theta_y]`, carrying
the electron energy distribution and its correlations with angle/intensity
directly in the data instead of an analytic Gaussian applied at Stage 2.

**Stage 0 -- `particles.py`.** `sample_bunch(compton, n_particles, gamma0,
sigma_gamma0, chirp=, angle_energy_corr=, rng=)` draws a macroparticle bunch
with real per-particle `(x0, y0, z0, gamma, theta_x, theta_y)` from their
*true* (untruncated) distributions -- unlike `particle_kernel`'s
grid-supplied angles and importance-sampled truncated position domain, so
no `f_th`/`z_weight`/`dsx`/`dsy`-style correction factors are needed here.
`push_and_sample(compton, bunch, n_steps=)` ballistically pushes each
particle through the pulse (same trajectory equations as `particle_kernel`)
and emits **one** `(gamma, theta_x, theta_y, a0, weight)` sample **per
particle** (not per timestep -- see "a0 is a trajectory average, not an
instantaneous sample" below). `gamma`/`theta_x`/`theta_y` are constant per
particle (straight-line, no pusher acceleration); `weight` is the
luminosity functional `L(zeta) = sum over the trajectory of v_rel *
n_ph_shape * dt * weight_macro` (Paper/xigma.tex eq. "lumfun"), using the
same physical constants as `particle_kernel`'s `f_cur`/
`calculate_intersection`'s `coef` (minus the angular-grid and
position-truncation terms that don't apply here); `a0` is the
trajectory-averaged effective intensity, see below. `n_steps` sets the
resolution of the internal per-timestep integrals that produce `weight` and
`a0`, not the length of `push_and_sample`'s output (always `n_particles`).

**a0 is a trajectory average, not an instantaneous sample.** `a0` (`H`'s 4th
axis) is `ahat(zeta) = (TrXi/2) * integral[a0_local(t)^2]^2 dt /
integral[a0_local(t)^2] dt` (Paper/xigma.tex eq. "ahattraj"), where
`a0_local(t) = compton.a0 * sqrt(local intensity / peak intensity)` is the
*instantaneous* local field amplitude computed internally at each timestep,
and `TrXi/2 = (1 + compton.ellipticity**2) / 2` (eq. "Xi";
`ellipticity=0` linear, `+-1` circular -- `Compton.ellipticity`, set via
`set_laser_parameters`). This is a genuine, previously-made mistake, not a
hypothetical one: an earlier version of this code deposited `a0_local(t)`
itself into `H` once per timestep, i.e. treated `a0` the same way as
`gamma`/`theta_x`/`theta_y` -- one distribution smeared over each particle's
whole trajectory. That's only valid in the synchrotron/wiggler regime, where
the photon formation length is about one laser cycle and the trajectory can
be split into independently-radiating segments. This codebase is in the
opposite, weakly-nonlinear regime (`a0 <~ 1`; see the paper's regime-validity
discussion), where the formation length spans the *whole* trajectory: an
electron radiates one line, shaped by the single effective intensity value
it experienced over its entire passage, not a sequence of per-instant
emissions. **Do not reintroduce per-timestep a0 deposition.**

**Stage 1 -- `deposition.py`.** `Grid4D.from_samples` derives axis extents
from the sample data plus a margin (not hard-coded like the legacy
`dx`/`dy`). `deposit_nearest`/`deposit_cic` bin `(gamma, theta_x, theta_y,
a0, weight)` samples into `H`; both are **array-module-agnostic** (`xp =
numpy` or `cupy`, auto-detected via `cp.get_array_module` or forced) --
there is no CUDA-specific kernel here, because deposition is an
independent-per-sample scatter-add and there's nothing for a custom kernel
to do beyond `xp.ravel_multi_index` + scatter-add. Scatter-add is
`np.bincount` on CPU and `cupyx.scatter_add` on GPU (see `_scatter_add`'s
docstring -- including the honest measurement that `bincount` isn't
dramatically faster than `np.add.at` in this numpy version despite
`add.at`'s folklore reputation; it's used anyway as the more
version-independent tool for accumulate-by-integer-index).
`build_table(gamma, theta_x, theta_y, a0, weight, scheme='nearest'|'cic',
device=None|'cpu'|'gpu', n_bins=, batch_size=, ...)` orchestrates grid
derivation + deposition + diagnostics into a `Table`; `device=None`
auto-detects from the input arrays' type, `batch_size` streams host arrays
through GPU deposition in bounded chunks so a Stage-0 sample set larger
than GPU memory still works (table stays resident, particles stream
through). `Table.save`/`.load` round-trip through `.npz`, always as host
(numpy) arrays regardless of which device built them.
`occupancy_diagnostics(table)` gives the empty-cell fraction and per-cell
count histogram ("Table too sparse" in Traps); `gamma_bracket(H, grid, q=)`
computes the quantile-based `(gamma_lo, gamma_hi)` Stage 2 needs (also
computed automatically inside `build_table` and stored as
`table.gamma_bracket`).

**Stage 2 -- `spectrum4d.py`.** `spectrum_kernel_4d` is a *new* kernel, not
an edit of `spectrum_kernel` -- see "Known bugs" for why the legacy kernel
is untouched. It reuses `spectrum_kernel`'s ring/arc annulus geometry, arc
construction, and inverse-CDF phi sampling unchanged in structure (the
Conventions below still apply to it). Three differences:

1. Quadrilinear interpolation of `H[gamma, theta_x, theta_y, a0]` replaces
   `collision`'s bilinear lookup + analytic Gaussian `ffac`. Coarse
   proposal: nearest-cell in all four axes, marginalised (summed) over
   `a0`. Final evaluation: quadrilinear interpolation, with a plain
   midpoint quadrature over `a0`'s (few, by design) bins nested inside the
   existing phi/subsampling loop.
2. `rmin_g`/`rmax_g` bracket from `table.gamma_bracket` instead of
   `gamma0 ± 3*sigma_g`.
3. No `emulate_nonlinearity`/`f_a` -- `a0` is a real axis now, so applying
   the phenomenological ponderomotive shift too would double-count it.

Two guards not present in `spectrum_kernel`, needed because `H`
(finite-particle deposition) has exact-zero cells in a way the legacy
smooth analytic `collision` essentially never does -- without them the
kernel produces NaN on realistic (sparsely-populated) tables:
- `inv_cdf`: falls back to a cell's left edge instead of computing `0/0`
  when a run of zero-weight phi cells flattens the CDF at a sample point.
- `sample_area`: contributes zero instead of `x/0` when a sample lands in
  a zero-weight cell after interpolation.

Also has the `MAX_ARCS` overflow guard CLAUDE.md used to flag as missing
in the legacy kernel (drops excess arcs instead of corrupting shared
memory). `calculate_angular_spectrum_4d(table, s, theta_x, theta_y,
phi_pol, samples_per_point=)` is the host driver, mirroring
`Compton.calculate_angular_spectrum`'s signature and return shape
(`spectrum, elapsed_seconds, debug`) -- it no longer takes `compton` (it
was only ever used for `compton.Wph`, no longer needed now that `coef` is
a pure numerical constant, see "Known bugs").

**Validation tools -- `reference.py`.** Three independent, non-GPU-kernel
ways to compute a spectrum from Stage 0/1 output, used to validate Stage 2
without trusting Stage 2 itself:

- `angle_integrated_spectrum(gamma, particle_weight, s)`: dN/ds integrated
  over all emission solid angle, from the standard angle-independent
  Compton edge shape alone (no table, no coef, no theta quadrature).
  **Validated** to <5% against `calculate_spectrum(...) * 4 *
  compton.Wph` (note the unit conversion).
- `spectrum_from_table(table, x0, y0, s, phi_pol)`: brute-force grid
  quadrature over `H`, no importance sampling. `coef = 1.5`, a pure
  numerical constant derived directly from the paper -- see "Known bugs"
  for the fix (an earlier version used a `Wph`/`pi**4`/`PHI_CELLS`-based
  `coef` that was wrong). **Validated** post-fix by grid-integrated
  cross-check against `direct_binning_spectrum` (agrees to <5% for a
  typical bunch) rather than against the legacy `calculate_angular_spectrum`
  (which has its own separate, unrelated, unfixed normalisation bug -- see
  "Known bugs" -- and is not a trustworthy ground truth here).
- `direct_binning_spectrum(gamma, theta_x, theta_y, particle_weight, a0,
  x0, y0, s_edges, phi_pol)`: per-real-macroparticle resonance binning, no
  table, no quadrature at all. Intended as the assumption-free correctness
  test for correlated bunches and a permanent debug tool. **Fixed** this
  session (see "Known bugs") -- normalisation root-caused and corrected; a
  small, deliberately-deferred `~2*pi` residual remains against
  `angle_integrated_spectrum` in grid-integrated comparisons.

## Conventions

- **Units are CGS**, constants at the top of `core.py`. Lengths and times
  inside the kernels are normalised to the laser wavenumber `k0_las`:
  positions are `k0_las * x`, times are `k0_las * c * t`. Particle
  coordinates are already normalised when packed into the `particles`
  array (legacy) or produced by `particles.py` (new path -- same
  convention, reused deliberately).
- **`theta_x`, `theta_y` are momentum angles** `p_{x,y}/gamma`, never
  positions, in both paths. The tabulation half-widths `dx`, `dy` (legacy:
  `calculate_angular_spectrum`; new: derived from `table.grid` in
  `calculate_angular_spectrum_4d`) are angular half-widths, not to be
  confused with grid spacing (`dtheta_x`/`dtheta_y` legacy,
  `grid.widths` new).
- **Single precision by default** in both kernels' arithmetic.
  `SINGLE_PRECISION = True` (core.py) sets `CP_FLOAT` etc. Inside kernels,
  wrap literals as `CP_FLOAT(...)` and use `CP_ONE`/`CP_ZERO`/`CP_TWO_PI`
  rather than bare Python floats -- mixing promotes to float64 silently
  and costs performance. `deposition.py`'s CPU/GPU deposition defaults to
  **float64** accumulation on both devices instead (`cupyx.scatter_add`
  supports it) -- a deliberate difference from the kernels' convention,
  not an oversight.
- **Shared memory is aliased and reused** in both `spectrum_kernel` and
  `spectrum_kernel_4d`. `TMP_FLOAT_ARRAY = inv_cdf`, and `rings` and
  `phi_cur` are both views into it at different offsets (`RINGS_SIZE`
  separates them). This is deliberate but fragile. Any new shared
  allocation must respect the existing offsets, and any change to
  `MAX_RINGS`, `PHI_EDGES`, or `CDF_PHI_RESOLUTION` must be checked
  against total shared-memory use in *both* kernels (they share the sizing
  constants, imported from `core.py` into `spectrum4d.py`).
- **Sizing constants** at the top of `core.py` are interdependent:
  `MAX_ARCS = 4 * MAX_RINGS`, `WEIGHTS_SIZE`, `CDF_SIZE`, `THREAD_STRIDE`
  all derive from `MAX_RINGS`/`PHI_EDGES`/`SAMPLES_TOTAL`/`X_THREADS`.
  Change the primitives, not the derived values.
- **Radial sampling is uniform in `theta**2`, not `theta`** -- required by
  the polar measure. This is easy to "fix" incorrectly; leave it. Same in
  `spectrum_kernel_4d`.
- Debug arrays are threaded through `spectrum_kernel`'s signature and
  currently all bound to the same `debug` array on the host, with the
  writes commented out. Keep the parameters when editing the signature.
  `spectrum_kernel_4d` has an *active* debug path (`debug_arr`,
  `debug_idx`, writes `x, y, f` per sample) plus a separate
  `dbg_scalars` output (per-output-point `skip`/`rmin`/`rmax`/`n_arcs`/
  `total_weight`) used while diagnosing the NaN issue above -- cheap,
  left in, useful for future debugging of a specific output point.
- **`Table` (deposition.py) always holds host/numpy arrays**, regardless
  of which device built it (`build_table` calls `.get()` before
  returning). Don't assume `table.H` needs `.get()` again, and don't pass
  cupy arrays into `Table(...)` directly.

## Known bugs

- **`spectrum_kernel`'s `sample_area` is `PHI_CELLS` (31) too large,
  unconditionally**, for any weight distribution. Traced to `cell_weights`
  in the coarse-weighting pass: it uses the arc's *full* angular width
  `(phi_max - phi_min)` for every phi cell's weight instead of that cell's
  own width `dphi_cell = (phi_max-phi_min)/PHI_CELLS`, which for a
  continuous inverse-CDF draw makes the final importance-sampling
  correction exactly `PHI_CELLS` too large (`(phi_max-phi_min)/dphi_cell =
  PHI_CELLS`, independent of the weights). Verified both algebraically
  (exact identity in the uniform-weight limit) and empirically (an
  independently-derived, correctly-normalised grid quadrature needs
  `* PHI_CELLS` to match `spectrum_kernel`'s output, across 7
  configurations spanning aspect ratio 1:1-100:1, on/off-axis, on/off-peak
  s). **Deliberately not fixed on this branch**: `coef = 3 /
  (4*pi**4*Wph*4)` in `calculate_angular_spectrum` appears to have been
  empirically tuned against the buggy `sample_area` -- patching
  `sample_area` alone (tested) breaks `calculate_angular_spectrum`'s
  agreement with the independent `calculate_spectrum` path. The two need
  fixing and re-validating together, which is a real, separate task, not
  a drive-by fix. `spectrum_kernel_4d` derives its own weights
  independently (`dphi_cell` used correctly) and does not inherit this.
  Note this legacy bug is unrelated to the two bugs below, despite past
  confusion -- see next entry.
- **FIXED: `reference.spectrum_from_table` and `spectrum4d.
  calculate_angular_spectrum_4d`'s `coef` was copied from the *legacy*
  kernel's own, separately-tuned constant above (`3/(4*pi**4*Wph*4)`), then
  further multiplied by `PHI_CELLS`.** Neither move was justified for these
  two functions: `spectrum_from_table` is a plain grid quadrature with no
  phi cells at all, and `H`'s weights are already correctly CGS-normalised,
  so no `Wph`/`pi**4`-based unit conversion or `PHI_CELLS` factor belongs in
  their `coef`. This produced a severe (4+ orders of magnitude near the
  Compton edge) shape divergence from `angle_integrated_spectrum` in
  grid-integrated comparisons -- see `direct_vs_table_discrepancy_report.md`
  for the investigation (table sparsity, grid resolution, and a naive
  power-of-gamma swap were all ruled out before the actual cause was found).
  Root-caused and fixed: the correct `coef`, derived directly from eq.
  "main"/"Fmatrix" (Paper/xigma.tex) rather than by matching the legacy
  kernel, is the pure numerical constant `1.5` -- no pi, no `Wph`, no
  `PHI_CELLS` -- in both functions. `compton` was dropped from both
  functions' signatures as a result (it was only ever used for
  `compton.Wph`). Needs re-validating against `calculate_angular_spectrum`
  now that the constant has changed (previous 1-8%/6-8% figures predate
  this fix).
- **FIXED: `reference.direct_binning_spectrum` had an unexplained
  ~3000-4000x normalisation gap** against `calculate_angular_spectrum`. Root
  cause: it was using the *ensemble-collapsed* prefactor (`g**5`, eq.
  "Fmatrix" -- meant for looking up a smooth, already-binned `H`) directly
  on raw, un-binned macroparticles, plus a spurious extra `1/s**2` division
  carried over from `spectrum_from_table`'s convention. Fixed: single-electron
  prefactor from eq. "xsec" (`g**2`, not `g**5`), pure numerical coefficient
  `3` (no `Wph`/`pi**4`), no extra `1/s**2`. See `reference.py`'s module
  docstring for the full derivation. A small, deliberately-deferred `~2*pi`
  residual remains against `angle_integrated_spectrum` in grid-integrated
  comparisons -- flagged, not chased further.

## Convergence testing

Tooling for resolution/deposition-scheme scans exists; no scan has been run
and recorded yet. Pattern:

    import numpy as np
    from xigma_i import particles, deposition, spectrum4d

    bunch = particles.sample_bunch(compton, n_particles, gamma0, sigma_gamma0)
    gamma, tx, ty, a0, w = particles.push_and_sample(compton, bunch, n_steps=200)

    results = {}
    for n_bins in [(32, 32, 32, 8), (64, 64, 64, 16), (128, 128, 128, 32)]:
        table = deposition.build_table(gamma, tx, ty, a0, w, n_bins=n_bins,
                                        scheme='nearest', device='gpu')
        spec, _, _ = spectrum4d.calculate_angular_spectrum_4d(
            compton, table, s_array, theta_x_array, theta_y_array, phi_pol=0.0)
        results[n_bins] = spec

    # compare consecutive resolutions -- the difference should shrink as
    # resolution increases, not just change

What to vary, one at a time (hold the rest at a value already "clearly
enough"):

- **Grid resolution** (gamma/theta_x/theta_y/a0 bins): `n_bins=` on
  `build_table`. Gamma resolution is the physically constrained axis
  (`δω/ω ≈ 2δγ/γ` against the reporting resolution you actually need); the
  other three are a memory/accuracy tradeoff (a 128×128×128×32 float32
  table is ~270 MB).
- **Particle statistics**: `n_particles` (`sample_bunch`) sets how many
  `(gamma, theta_x, theta_y, a0, weight)` rows land in the table --
  `push_and_sample` emits exactly one per particle. `n_steps`
  (`push_and_sample`) does *not* affect that count; it's purely the
  trajectory-integration resolution feeding each particle's `weight`/`a0`
  (see "a0 is a trajectory average, not an instantaneous sample" above) --
  too coarse and those two integrals are inaccurate, but it never changes
  how many rows Stage 1 sees.
- **Quadrature resolution**: `samples_per_point=` on
  `calculate_angular_spectrum_4d`.
- **Deposition scheme**: rerun the same scan with `scheme='nearest'` vs
  `scheme='cic'` at fixed particle count. CIC should be smoother -- fewer
  near-zero-occupancy cells, check via `deposition.occupancy_diagnostics`
  -- without shifting peak position or integrated flux (integrate
  `spectrum_from_table` or `calculate_angular_spectrum_4d`'s output over
  theta/s).

Statistical and discretisation error don't disentangle automatically: at
fixed `n_bins`, increasing particle count converges the estimate toward
the true density; at fixed particle count, increasing `n_bins` past
roughly `sqrt(particles per cell)` starts amplifying per-cell shot noise
instead of reducing bias. Check `occupancy_diagnostics`' per-cell count
histogram, not just whether the spectrum shape looks smoother.

`deposition.check_accumulation_precision(H_f64, H_f32)` compares two
`build_table` runs with different `accumulate_dtype` directly, for the
"does float32 lose anything here" question specifically -- run both on
`device='cpu'` (float64 accumulation isn't optional on GPU here; both
devices default to float64, see Conventions).

## Traps

- **`cupyx.scatter_add`'s dtype restriction.** It's backed by `cupy.add.at`,
  which only supports `int32, float16, float32, float64, uint32, uint64` --
  notably not `int64`. `deposition.py`'s occupancy counting used to
  accumulate in `int64` and hit `TypeError: cupy.add.at only supports
  int32, float16, float32, float64, uint32, uint64` on some cupy/driver
  combinations but not others (nothing to do with problem size or GPU
  model) -- fixed by switching occupancy to `int32` (plenty for realistic
  per-cell counts). Any new GPU scatter-add target must stay off `int64`.
- **`coef` in `calculate_intersection`** is where all the physical
  normalisation lives. If the legacy-path spectrum comes out wrong by a
  clean factor, look here before anywhere else.
- **`emulate_nonlinearity` left active** alongside a real `a0` axis --
  silent double-count of the ponderomotive shift, will look like a
  calibration error. Not applicable to the new path (`spectrum_kernel_4d`
  never has this flag), but a trap if you ever compare the two paths'
  outputs directly.
- **Depositing `a0` per timestep instead of trajectory-averaged.** A real
  mistake made and fixed on this branch, not hypothetical -- see
  "Architecture: new path"'s "a0 is a trajectory average, not an
  instantaneous sample". If you're about to make `push_and_sample` emit
  more than one row per particle, or bin `a0_local(t)` directly into `H`,
  stop and re-read that section first.
- **Shared-memory aliasing.** Adding a shared array without accounting for
  `TMP_FLOAT_ARRAY`/`rings`/`phi_cur` overlap corrupts the arc geometry in
  ways that produce plausible-looking output. Applies to both
  `spectrum_kernel` and `spectrum_kernel_4d`.
- **`MAX_ARCS` overflow** is guarded in `spectrum_kernel_4d` (drops excess
  arcs) but still unguarded in the legacy `spectrum_kernel`.
- **`dr` scope.** In the final evaluation loop `dr` is read from a
  variable set inside the earlier `if not skip:` block, across
  `syncthreads()`. Verify it is still in scope and correct after any
  restructuring -- cupyx.jit scoping is easy to get subtly wrong. True in
  both kernels.
- **Table too sparse.** Check `deposition.occupancy_diagnostics`'
  per-cell occupancy histogram before blaming the quadrature for noise.
- **`theta_x`/`theta_y` as positions** rather than momentum angles.
- **Float32 vs float64 accumulation.** `deposition.py` defaults to
  float64 on both CPU and GPU now (see Conventions) -- if you override to
  float32 for memory, verify with
  `deposition.check_accumulation_precision` rather than assuming it's
  fine.
- **CPU vs GPU deposition, individual cells.** Total weight and marginals
  match tightly between `device='cpu'` and `device='gpu'` on the same
  samples, but a small fraction of *individual* cells can differ (a
  sample whose bin coordinate sits extremely close to an edge can round
  to different neighbouring cells under different floating-point paths).
  Benign, documented in `_deposit_gpu`'s docstring; don't expect
  bit-identical per-cell tables between devices.
- **`spectrum_kernel`'s absolute normalisation** -- see "Known bugs". Don't
  use it as an unqualified ground truth for the new path's validation;
  `spectrum_from_table`/`spectrum_kernel_4d` no longer share any
  normalisation convention with it (that coupling was itself a bug, now
  fixed -- see "Known bugs").

## Environment

CuPy with `cupyx.jit` rawkernels; requires a CUDA device. `scipy` is used
on the host for `erfcx`; `cupyx.scipy.special` for `erf`/`erfinv` on
device. No build system, no dependency manifest, no repo-tracked test
suite at present (validation lives in ad hoc scripts run against
`reference.py`/`deposition.py`'s functions, not a `pytest` tree).

A working GPU environment was set up as a conda env (`conda create -n
xigma python=3.12`, then `pip install numpy scipy pytest cupy-cuda12x
tomli`, matched to the local driver's CUDA version) -- there wasn't one in
the base environment this branch's work started from. Recreate similarly
if starting fresh; `cupyx.scatter_add`/`cupy.get_array_module` (used
throughout `deposition.py`) need a real cupy install, not just a CUDA
driver.
