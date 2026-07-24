"""The single error metric used by every figure (plan.md Sec. 0: "Do not
switch metrics between figures").

Convention: bin the spectrum into windows of width `mu` (the reporting
resolution), integrate flux within each window (trapezoidal in `s`), and
report two numbers relative to a reference spectrum computed on the same
`s` grid:

  - the reference-flux-weighted L1 relative deviation (the headline number,
    representative of the whole spectrum), and
  - the max relative deviation over windows (sensitive to localized failures,
    e.g. near the Compton edge).

Windows with negligible reference flux are down-weighted rather than
excluded, so a handful of empty-reference bins at the frequency-grid edges
cannot produce spurious inf/nan.
"""
import numpy as np


def window_integrated_relative_error(s, spec, spec_ref, mu, floor_fraction=1e-6):
    """s, spec, spec_ref: 1D arrays of equal shape (same frequency grid for
    both spectra -- resample spec onto spec_ref's grid before calling if
    they differ). mu: window width in the same units as s.

    Returns (weighted_l1, max_rel, n_windows).
    """
    s = np.asarray(s, dtype=np.float64)
    spec = np.asarray(spec, dtype=np.float64)
    spec_ref = np.asarray(spec_ref, dtype=np.float64)
    order = np.argsort(s)
    s, spec, spec_ref = s[order], spec[order], spec_ref[order]

    lo, hi = s[0], s[-1]
    n_windows = max(1, int(round((hi - lo) / mu)))
    edges = np.linspace(lo, hi, n_windows + 1)
    idx = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, n_windows - 1)

    ds = np.gradient(s)
    flux = spec * ds
    flux_ref = spec_ref * ds

    win_flux = np.bincount(idx, weights=flux, minlength=n_windows)
    win_flux_ref = np.bincount(idx, weights=flux_ref, minlength=n_windows)

    total_ref = win_flux_ref.sum()
    floor = floor_fraction * total_ref / n_windows if total_ref > 0 else 0.0
    denom = np.maximum(np.abs(win_flux_ref), max(floor, 1e-300))
    rel = np.abs(win_flux - win_flux_ref) / denom

    significant = win_flux_ref > floor
    if not np.any(significant):
        return 0.0, 0.0, n_windows

    weights = np.where(significant, win_flux_ref, 0.0)
    weights = weights / weights.sum()

    l1 = float(np.sum(weights * rel))
    mx = float(rel[significant].max())
    return l1, mx, n_windows


def resample_to(s_ref, s_src, spec_src):
    """Linear-interpolate spec_src(s_src) onto s_ref, zero outside s_src's range."""
    return np.interp(s_ref, s_src, spec_src, left=0.0, right=0.0)
