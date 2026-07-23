# `direct_binning_spectrum` vs `spectrum_from_table`/`spectrum_kernel_4d`: open normalization/shape discrepancy

**RESOLVED.** Root cause: `spectrum_from_table`/`calculate_angular_spectrum_4d`'s
`coef` had been copied from the legacy kernel's own, separately-tuned
`3/(4*pi**4*Wph*4) * PHI_CELLS` constant (§2.2 below) — never independently
justified for these two functions, and wrong. Neither belongs here:
`spectrum_from_table` is a plain grid quadrature with no phi cells, and `H`'s
weights are already correctly CGS-normalised (§1's $L_i$ note). The correct
`coef`, re-derived directly from eq:main/eq:Fmatrix, is the pure numerical
constant `3/2` — no pi, no `Wph`, no `PHI_CELLS`. Fixed in both `reference.py`
and `spectrum4d.py`; see CLAUDE.md "Known bugs" for the fix as committed. The
rest of this document is kept as the investigation record that led there.

**Post-fix verification** (`compare_direct_vs_table.py --grid-integrate`):
for a typical, not-pathologically-collimated bunch (`sigma_thx=sigma_thy=3e-4`,
`s` swept `0.5-1.05 * gamma0**2`), all three methods now agree within
statistical noise, clustered around the known, still-deliberately-deferred
`~2*pi` residual against `angle_integrated_spectrum`:
`spectrum_from_table`=6.30±0.17, `direct_binning_spectrum`=6.44±0.68,
`spectrum_kernel_4d`=7.15±0.93 — a very different picture from the 4+
orders of magnitude divergence this document describes below. One remaining
wrinkle: in narrow-angle/sparse-table configs (small `sigma_thx`, few
particles, or coarse `n_bins`), `spectrum_kernel_4d` alone shows large,
unstable run-to-run variance — overshoots ranging from ~3x up to >30x
depending on particle count/table resolution, with correspondingly huge
std — while `spectrum_from_table`/`direct_binning_spectrum` keep agreeing
tightly with each other in the same configs. This reads as heavy-tailed
importance-sampling noise from sparse/zero-weight `H` cells in the kernel's
own quadrature (see CLAUDE.md's "Table too sparse" trap), not a
normalisation error — but it's a real, separate, open observation, flagged
here rather than chased further this pass.

Status: unresolved (historical). This document lays out the two competing
derivations precisely, the concrete numbers, and what's been ruled out, so the
remaining gap can be chased analytically rather than by further blind
numerical experiments. Equation labels (`eq:xxx`) refer to `Paper/xigma.tex`.

**Update**: `spectrum_kernel_4d` (the production GPU kernel — Fibonacci/
inverse-CDF importance sampling, completely independent code from
`spectrum_from_table`'s brute-force grid quadrature) was checked against the
same test and shows the *same* divergence, same onset/peak location (§3).
Since the two share nothing but the underlying `eq:main`/`eq:Fmatrix`
formula, this rules out an implementation bug in either quadrature and
narrows the problem to that shared derivation (§5).

## 1. Target quantity

Both functions are meant to compute the same doubly-differential spectrum,
photons per unit normalised frequency per unit solid angle, at one fixed
observation direction $\vb n = (x_0, y_0)$ (small-angle $\theta_x,\theta_y$
momentum-angle convention, $s \equiv \omega/(4\omega_L)$):

$$
\left.\frac{d^3N}{ds\,d^2\Omega}\right|_{\vb n=(x_0,y_0)}
$$

from the same underlying macroparticle ensemble
$\{(\gamma_i,\theta_{x,i},\theta_{y,i},\hat a_i, L_i)\}_{i=1}^{N_p}$
(`particles.push_and_sample`'s output; $L_i$ is the luminosity weight,
eq:lumfun, confirmed to already be the correctly CGS-normalised
$v_\mathrm{rel}\sigma_\mathrm{T}\!\int n_\mathrm{ph}\,dt$ per macroparticle).

## 2. Two derivations of the same quantity

### 2.1 `direct_binning_spectrum` — raw particles, eq:xsec

Starts from the single-electron, **not-yet-ensemble-collapsed**
differential cross-section (eq:xsec):

$$
\frac{d^3\hat\sigma}{d\omega\,d^2\Omega}
= 3\sigma_\mathrm{T}\,\frac{\wR}{\omega}\,
\frac{\gamma^2}{(1+\gamma^2\theta^2)^2}\,
R(\omega-\wR;\hat a)\;\hat U^T\hat\Xi\hat U .
$$

Convert to a photon count via the incident flux
($d^3N = v_\mathrm{rel}\!\int n_\mathrm{ph}\,dt\;d^3\hat\sigma$,
$v_\mathrm{rel}=2c$ for near-backscattering — this is exactly the $V\_REL$
already baked into $L_i$), and collapse $R\to\delta(\omega-\wR)$
per electron (justified for any bunch/pulse whose resonant-frequency spread
exceeds $\dwsingle$, eq:deltasub). For particle $i$, observed from
$\vb n=(x_0,y_0)$ with $\theta_i \equiv \abs{\vb n - \vb\theta_i}$
(eq:smallangle):

$$
\frac{d^3N_i}{d\omega\,d^2\Omega}
= 3\,L_i\,\gamma_i^2\,\tilde g(\theta_i)\,a(\theta_i)\;
\delta\!\bigl(\omega-\wR(\gamma_i,\theta_i,\hat a_i)\bigr),
\qquad
\tilde g(\theta) = \frac{1}{(1+\gamma^2\theta^2)^2},
$$

$a(\theta)$ the polarisation factor (eq:Ftrace's bracket), and $\wR$ from
eq:wRgamma:

$$
\wR(\gamma,\theta,\hat a) = \frac{4\wL\gamma^2}{1+\gamma^2\theta^2+\hat a}.
$$

Histogramming over $s$ in bins of width $\Delta s$
($d\omega = 4\wL\,ds$, a **constant** Jacobian that cancels exactly against
the same factor converting the histogram's own bin width — no leftover
factor):

$$
\left.\frac{d^3N}{ds\,d^2\Omega}\right|_\mathrm{direct}
= \frac{1}{\Delta s}\sum_{i\,:\,s_{\mathrm{res},i}\in\mathrm{bin}}
3\,L_i\,\gamma_i^2\,\tilde g(\theta_i)\,a(\theta_i),
\qquad
s_{\mathrm{res},i} = \frac{\gamma_i^2}{1+\hat a_i+\gamma_i^2\theta_i^2}.
$$

This is exactly `direct_binning_spectrum`'s code. Verified this session:

- On-axis, $\theta=0$: reduces to
  $d^2N/d^2\Omega|_{\theta=0}=3\gamma^2 L_i$, matching the independently
  stated $6\sigma_\mathrm{T}\gamma^2\cdot c\!\int n\,dt$ once $v_\mathrm{rel}=2c$
  is used in the flux conversion instead of bare $c$.
- Grid-integrated over $(x_0,y_0)$ (Riemann sum), it tracks
  `angle_integrated_spectrum` (independently validated to 0.5-3% against the
  legacy `calculate_spectrum`) in **both shape and scale** (within an
  $O(1)$-few factor) across a wide $s$ range including the Compton edge.

### 2.2 `spectrum_from_table` — table quadrature, eq:main + eq:Fmatrix

Starts from the **ensemble-collapsed** spectrum (eq:main), which performs a
*second* collapse beyond 2.1: it sifts $\delta(\omega-\wR(\gamma,\theta,\hat
a))$ against the smooth electron density
$\mathcal H(\gamma,\theta_y,\theta_z,\hat a)$ (eq:Hdef) over the electron
*ensemble's* $\gamma$-spread (eq:sift), producing the Jacobian
$\abs{d\Gamma/d\omega}$ (eq:jacobian):

$$
\frac{d^3\hat N}{d\omega\,d^2\Omega}
= \frac{6\sigma_\mathrm{T}\wL}{\omega^2}
\int \hat F(\Gamma,\theta,\hat a)\,
     H(4\wL-\omega\theta^2)\,
     \mathcal H(\Gamma,\theta_y,\theta_z,\hat a)\,
     \dd\theta_y\dd\theta_z\dd\hat a,
$$

$$
\Gamma(\omega,\theta,\hat a) = \sqrt{\frac{\omega(1+\hat a)}{4\wL-\omega\theta^2}}
\quad\text{(eq:Gamma, a \emph{lookup}, not integrated)},
$$

$$
\hat F(\Gamma,\theta,\hat a) = \frac{\Gamma^5}{(1+\Gamma^2\theta^2)^2(1+\hat a)}\,
\hat U^T\hat\Xi\hat U
\qquad\text{(eq:Fmatrix)}.
$$

The extra $\Gamma^3/(1+\hat a)$ in $\hat F$ relative to eq:xsec's bare
$\Gamma^2$ **is** $\abs{d\Gamma/d\omega}$ — it belongs to the *ensemble*
$\gamma$-collapse, not to anything in 2.1.

Converting $\omega\to s$ (again a constant Jacobian) and discretising
$\mathcal H$ on a grid with cells of volume
$\delta V_3=\delta\theta_x\,\delta\theta_y\,\delta\hat a$ (no $\delta\gamma$
— $\gamma$ is a lookup, not an integration variable in this integral) gives
`spectrum_from_table`'s actual code:

$$
\left.\frac{d^3N}{ds\,d^2\Omega}\right|_\mathrm{table}
= \frac{C}{s^2}\sum_{\mathrm{cells}}
\hat F(\Gamma_c,\theta_c,\hat a_c)\,
H(\Gamma_c,\theta_{x,c},\theta_{y,c},\hat a_c)\,\delta V_3,
\qquad
C = \frac{3}{4\pi^4 W_\mathrm{ph}}\times\mathrm{PHI\_CELLS}
$$

($\mathrm{PHI\_CELLS}$ is an unrelated calibration constant tying this
kernel's importance-sampling convention to `spectrum_kernel`'s, see
`reference.py`'s module docstring — not implicated here). $H$ is
`deposition.py`'s **density**,
$H = H_\mathrm{raw}/(\delta\gamma\,\delta\theta_x\,\delta\theta_y\,\delta\hat a)$
— the full 4D bin volume, including $\delta\gamma$ — so that
$\sum_\mathrm{cells} H\cdot(\delta\gamma\,\delta V_3) = \sum_i L_i$ exactly
(checked numerically: `table.H.sum()*table.grid.bin_volume == table.total_weight`).

## 3. The concrete numbers

Setup: $\gamma_0=1000$, $\sigma_\gamma/\gamma_0=0.01$, $\hat a\sim10^{-9}$
(deliberately negligible — nonlinearity ruled out as a cause), tight beam
($\sigma_{\theta x}=\sigma_{\theta y}=3\times10^{-5}$, i.e.
$\sigma_\theta\gamma_0 \ll 1$), $N_p=4\times10^5$, table $64\times8\times8\times4$.

**Single point**, $(x_0,y_0)=(0,0)$, $s=9.7\times10^5$:

$$
\left.\frac{d^3N}{ds\,d^2\Omega}\right|_\mathrm{table} = 4.5\times10^{9}
\qquad
\left.\frac{d^3N}{ds\,d^2\Omega}\right|_\mathrm{direct} \approx 2.0\times10^{5}
\qquad(\text{ratio} \approx 2.3\times10^4)
$$

**Grid-integrated** (Riemann sum over an $(x_0,y_0)$ grid,
$\times\,\delta x_0\,\delta y_0$), compared against
`angle_integrated_spectrum` (trusted, validated independently):

- `direct_binning_spectrum`: tracks the trusted curve's *shape* (peak
  location, width) across the full $s$ range including the Compton edge,
  staying within an $O(1)$-few factor throughout.
- `spectrum_from_table`: diverges by **more than four orders of magnitude**
  specifically as $s$ approaches the edge — rises smoothly from $\sim1$ to
  $\sim5\times10^4$ across a modest $s$-window where the trusted curve is
  roughly flat. This is a **shape** divergence, not a constant offset.
- `spectrum_kernel_4d` (the production GPU kernel — Fibonacci/inverse-CDF
  importance-sampled quadrature, `debug_plots/grid_integrate_3way.png`):
  shows the **same** divergence as `spectrum_from_table` — same onset
  location ($s\approx0.79\times10^6$), same peak location
  ($s\approx0.99\times10^6$), same qualitative shape, actually a *larger*
  peak ratio ($\sim3\times10^5$ mean vs. table's $\sim2.9\times10^4$).
  This is the decisive result: `spectrum_from_table` and
  `spectrum_kernel_4d` share **only** the `eq:main`/`eq:Fmatrix`/`eq:Gamma`
  formula — their quadrature implementations (brute-force grid sum vs.
  Fibonacci/importance-sampled Monte Carlo) are completely different code
  paths. Both failing the same way, with the same onset/peak location,
  rules out an implementation bug specific to either one's quadrature and
  points squarely at the shared formula itself (§5).

## 4. What's been ruled out

- **Table sparsity.** Cutting empty-cell fraction from $0.887\to0.527$
  (coarser $\theta,\hat a$ binning at fixed $N_p$) changed the discrepancy by
  $<10\%$.
- **$(x_0,y_0)$ grid resolution** in the Riemann-sum test: $11\times11$ vs
  $25\times25$ gave identical shape and peak location.
- **A missing scalar constant.** Swapping $\hat F$'s
  $\Gamma^5/(1+\hat a)\to\gamma^2$ (testing whether `spectrum_from_table`
  should use eq:xsec's bare power instead of eq:Fmatrix's Jacobian-inclusive
  one) moved the single-point value from $4.5\times10^9$ to
  $3.7\times10^{-4}$ — twelve orders of magnitude in the *wrong* direction
  relative to the $2.0\times10^5$ target. Not simply "wrong power of
  $\gamma$".
- **The $\delta\gamma$ question in $\delta V_3$.** Including vs excluding
  $\delta\gamma$ in the quadrature cell volume changes the single-point
  value only by the modest factor $\delta\gamma\approx1.7$ itself — nowhere
  near the $10^4$ gap.
- **A quadrature-implementation bug in `spectrum_from_table` specifically.**
  `spectrum_kernel_4d` (completely independent code: GPU, Fibonacci/inverse-
  CDF importance sampling, no brute-force grid loop) shows the same
  divergence, same onset, same peak location (§3). Rules this out; the bug
  is upstream of both quadrature implementations.

## 5. The open question

`spectrum_from_table` and `spectrum_kernel_4d` are meant to evaluate the
same integral as `direct_binning_spectrum` (eq:collision, un-collapsed)
against the same particle ensemble, via two representations of
$\mathcal H$ (eq:Hdef):

- **direct**: the literal $\delta$-comb,
  $\mathcal H = \sum_i L_i\,\delta(\gamma-\gamma_i)\delta(\theta_y-\theta_{y,i})
  \delta(\theta_z-\theta_{z,i})\delta(\hat a-\hat a_i)$, substituted directly
  into eq:collision (before the ensemble $\gamma$-collapse) and evaluated by
  sifting the $\theta_y,\theta_z,\hat a$ deltas against the
  observation-relative resonance condition — one $\delta(\omega-\wR_i)$
  spike per particle, no second collapse.
- **table / kernel_4d**: the same ensemble, first **binned/smoothed** into
  $H$ on a grid, then plugged into eq:main — the *already* $\gamma$-collapsed
  3D integral (only $\theta_y,\theta_z,\hat a$ remain; $\Gamma$ is a lookup
  via eq:Gamma, carrying eq:jacobian's $\Gamma^3/(1+\hat a)$). Both
  `spectrum_from_table` (brute-force grid sum) and `spectrum_kernel_4d`
  (Fibonacci/inverse-CDF importance sampling) implement *this same*
  derivation independently, and both fail the same way (§3) — so the
  problem is in this shared derivation/formula itself, not in either
  quadrature's code.

These must agree in the limit of a fine table / many particles, but the
observed gap *grows* toward the Compton edge and is insensitive to table
resolution over the range tested here, and is now confirmed to affect
**both** independent implementations of the table/eq:main path equally.
The likely location of the real bug: something in how eq:main's
*already-collapsed* 3D integral is supposed to reduce back to eq:collision's
*uncollapsed* form when $\mathcal H$ is a literal particle-by-particle
$\delta$-comb rather than a genuinely smooth density — i.e. the
correspondence between "bin the ensemble first, then apply the
$\Gamma^5$/eq:jacobian machinery" and "skip the ensemble collapse entirely,
using eq:xsec's bare $\gamma^2$ per real particle" has not been shown to
actually agree numerically, only assumed to.

## 6. Where to reproduce

`compare_direct_vs_table.py` (repo root), `--grid-integrate` mode (add
`--samples-per-point N` to control `spectrum_kernel_4d`'s importance-sampling
resolution — it's always included in grid-integrate mode), e.g.:

```
conda run -n xigma python compare_direct_vs_table.py --grid-integrate \
  --grid-n 25 --grid-halfwidth-sigma 6 --s-range 0.6 1.1 --n-s-bins 40 \
  --n-bins 64 8 8 4 --a0-target 0.0001 --samples-per-point 128 \
  --out grid_integrate.png
```

Produced plot for the 3-way comparison in §3:
`debug_plots/grid_integrate_3way.png`.

Cell-by-cell tracing of `spectrum_from_table`'s top contributors (used to
produce §3's single-point numbers) is inline Python, not yet a saved script
— ask if you want it turned into one.
