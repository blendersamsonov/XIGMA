"""Debug/validation script: compare direct_binning_spectrum against
spectrum_from_table at a single observation point, for a configurable
electron/laser setup.

Both are supposed to converge to the same dN/(ds dOmega) at a fixed
observation direction (x0, y0), from the same bunch -- one via the raw
macroparticles directly (direct_binning_spectrum), one via the binned/
interpolated 4D table H (spectrum_from_table). See CLAUDE.md and
reference.py's module docstring for the derivation and the currently-open
~20000-40000x single-point discrepancy this script is built to investigate.

Usage:
    conda run -n xigma python compare_direct_vs_table.py [options]
    conda run -n xigma python compare_direct_vs_table.py --gamma0 2000 \
        --sigma-thx 5e-5 --sigma-thy 5e-5 --a0 0.5 --x0 0.0 --y0 0.0

All electron/laser parameters are exposed as CLI flags; run with --help for
the full list. Output: prints summary ratio statistics and saves a PNG
(default compare_direct_vs_table.png) with two panels -- the two spectra
overlaid (log scale) and their ratio -- plus a printed diagnostic block
(bunch a0/theta stats, table occupancy).
"""
import argparse
import os
import sys

import numpy as np
import cupy as cp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from xigma_i.core import Compton
from xigma_i import particles, deposition, reference, spectrum4d


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    pe = p.add_argument_group("electron bunch")
    pe.add_argument("--charge-nc", type=float, default=1.0, help="bunch charge, nC")
    pe.add_argument("--gamma0", type=float, default=1000.0, help="mean Lorentz factor")
    pe.add_argument("--sigma-gamma-rel", type=float, default=0.01, help="fractional energy spread, sigma_gamma/gamma0")
    pe.add_argument("--sigma-ex", type=float, default=5e-4, help="electron RMS transverse size x, cm")
    pe.add_argument("--sigma-ey", type=float, default=5e-4, help="electron RMS transverse size y, cm")
    pe.add_argument("--sigma-ez", type=float, default=1e-2, help="electron RMS bunch length, cm")
    pe.add_argument("--norm-emit-x", type=float, default=None,
                     help="normalised emittance x, cm*rad (overrides --sigma-thx if given)")
    pe.add_argument("--norm-emit-y", type=float, default=None,
                     help="normalised emittance y, cm*rad (overrides --sigma-thy if given)")
    pe.add_argument("--sigma-thx", type=float, default=3e-5,
                     help="angular divergence x (theta_x RMS); sets emit_x = sigma_thx*sigma_ex "
                          "unless --norm-emit-x is given")
    pe.add_argument("--sigma-thy", type=float, default=3e-5, help="angular divergence y, see --sigma-thx")
    pe.add_argument("--chirp", type=float, default=0.0, help="energy-position chirp, see particles.sample_bunch")
    pe.add_argument("--angle-energy-corr", type=float, default=0.0, help="divergence-energy correlation")

    pl = p.add_argument_group("laser")
    pl.add_argument("--wl", type=float, default=None,
                     help="laser pulse energy, J (overrides --a0-target if given)")
    pl.add_argument("--a0-target", type=float, default=0.0001,
                     help="approximate desired compton.a0 (peak); WL is solved for unless --wl is given. "
                          "Kept deliberately tiny by default so nonlinearity/a0 effects are ruled out "
                          "while debugging other issues -- raise explicitly when actually testing a0.")
    pl.add_argument("--lambda-l", type=float, default=0.8e-4, help="laser wavelength, cm")
    pl.add_argument("--sigma-lr0", type=float, default=3e-4, help="laser RMS focal radius, cm")
    pl.add_argument("--sigma-lz", type=float, default=3e-4, help="laser RMS pulse length, cm")
    pl.add_argument("--ellipticity", type=float, default=0.0, help="laser polarisation ellipticity, 0=linear, +-1=circular")
    pl.add_argument("--beta-ff", type=float, default=0.0)

    pp = p.add_argument_group("particles / table / observation")
    pp.add_argument("--n-particles", type=int, default=400_000)
    pp.add_argument("--n-steps", type=int, default=200)
    pp.add_argument("--n-bins", type=int, nargs=4, default=(64, 64, 64, 16),
                     metavar=("N_GAMMA", "N_THETA_X", "N_THETA_Y", "N_A0"))
    pp.add_argument("--scheme", choices=["nearest", "cic"], default="cic")
    pp.add_argument("--x0", type=float, default=0.0, help="observation direction theta_x")
    pp.add_argument("--y0", type=float, default=0.0, help="observation direction theta_y")
    pp.add_argument("--phi-pol", type=float, default=0.0)
    pp.add_argument("--s-range", type=float, nargs=2, default=(0.9, 1.05),
                     metavar=("S_LO_FRAC", "S_HI_FRAC"),
                     help="s range as fractions of gamma0**2 (on-axis resonant peak); default is wide "
                          "enough to show the full rise-peak-cutoff shape, not just the declining tail")
    pp.add_argument("--n-s-bins", type=int, default=60)
    pp.add_argument("--seed", type=int, default=0)
    pp.add_argument("--also-kernel", action="store_true",
                     help="also compute spectrum_kernel_4d (GPU importance-sampled quadrature) for a 3-way comparison")
    pp.add_argument("--samples-per-point", type=int, default=64, help="only used with --also-kernel")

    pg = p.add_argument_group("grid-integrate mode (--grid-integrate)")
    pg.add_argument("--grid-integrate", action="store_true",
                     help="instead of a single-point comparison, Riemann-sum spectrum_from_table and "
                          "direct_binning_spectrum over a coarse (x0,y0) grid and compare the angle-"
                          "integrated result against the trusted angle_integrated_spectrum -- decisive "
                          "near the Compton edge, where the small-angle approximation both methods use "
                          "is actually valid (see CLAUDE.md / this session's discussion)")
    pg.add_argument("--grid-n", type=int, default=21, help="grid points per axis (coarse is fine near the edge)")
    pg.add_argument("--grid-halfwidth-sigma", type=float, default=6.0,
                     help="grid half-width in units of theta_x's RMS divergence")

    po = p.add_argument_group("output")
    po.add_argument("--out", default="compare_direct_vs_table.png")

    return p.parse_args()


def build_compton(args):
    compton = Compton()

    emit_x = args.norm_emit_x / args.gamma0 if args.norm_emit_x is not None else args.sigma_thx * args.sigma_ex
    emit_y = args.norm_emit_y / args.gamma0 if args.norm_emit_y is not None else args.sigma_thy * args.sigma_ey

    compton.set_electron_parameters(
        chargeNC=args.charge_nc, emit_x=emit_x, emit_y=emit_y,
        sigma_ex=args.sigma_ex, sigma_ey=args.sigma_ey, sigma_ez=args.sigma_ez,
    )

    wl = args.wl
    if wl is None:
        # a0 ~ WL linearly (see core.py set_laser_parameters); solve by a quick probe.
        compton.set_laser_parameters(WL=1.0, lambda_l=args.lambda_l, sigma_lr0=args.sigma_lr0,
                                      sigma_lz=args.sigma_lz, beta_ff=args.beta_ff, ellipticity=args.ellipticity)
        wl = args.a0_target / compton.a0 if compton.a0 > 0 else 1.0
    compton.set_laser_parameters(WL=wl, lambda_l=args.lambda_l, sigma_lr0=args.sigma_lr0,
                                  sigma_lz=args.sigma_lz, beta_ff=args.beta_ff, ellipticity=args.ellipticity)
    compton.set_foci_displacement(0.0, 0.0, 0.0)
    return compton


def grid_integrate(table, gamma, tx, ty, w, a0, s_edges, s_centers, phi_pol,
                    grid_n, half_width, samples_per_point=None):
    """Riemann-sum spectrum_from_table, direct_binning_spectrum, and (if
    samples_per_point is given) spectrum_kernel_4d over a (x0, y0) grid,
    using the actual (fixed) functions directly -- not a reimplementation --
    so this can't silently drift out of sync with the real formulas.
    Returns (total_table, total_direct, total_kernel or None), each matching
    s_centers's shape, in dN/ds (angle-integrated).

    spectrum_kernel_4d natively takes array (theta_x, theta_y, s) grids in a
    single GPU call (it returns a (theta_x.size, theta_y.size, s.size)
    array), so unlike the other two this doesn't need a Python-level loop
    over grid points -- just sum the returned array over its first two axes.
    """
    xs = np.linspace(-half_width, half_width, grid_n)
    ys = np.linspace(-half_width, half_width, grid_n)
    dA = (xs[1] - xs[0]) * (ys[1] - ys[0]) if grid_n > 1 else 1.0

    total_table = np.zeros_like(s_centers)
    total_direct = np.zeros_like(s_centers)
    for x0 in xs:
        for y0 in ys:
            total_table += reference.spectrum_from_table(table, x0, y0, s_centers, phi_pol=phi_pol)
            total_direct += reference.direct_binning_spectrum(gamma, tx, ty, w, a0, x0, y0, s_edges, phi_pol=phi_pol)
    total_table *= dA
    total_direct *= dA

    total_kernel = None
    if samples_per_point is not None:
        s_gpu = cp.asarray(s_centers, dtype=cp.float32)
        xs_gpu = cp.asarray(xs, dtype=cp.float32)
        ys_gpu = cp.asarray(ys, dtype=cp.float32)
        spec_grid, _, _ = spectrum4d.calculate_angular_spectrum_4d(
            table, s_gpu, xs_gpu, ys_gpu, phi_pol=phi_pol,
            samples_per_point=samples_per_point)  # (grid_n, grid_n, n_s)
        total_kernel = spec_grid.sum(axis=(0, 1)) * dA

    return total_table, total_direct, total_kernel


def run_grid_integrate(args, table, gamma, tx, ty, w, a0, s_edges, s_centers):
    ref = reference.angle_integrated_spectrum(gamma, w, s_centers)

    half_width = args.grid_halfwidth_sigma * tx.std()
    print(f"=== grid-integrate: {args.grid_n}x{args.grid_n} grid, half-width {half_width:.4g} "
          f"({args.grid_halfwidth_sigma:.3g} sigma_thx), spectrum_kernel_4d samples_per_point="
          f"{args.samples_per_point} ===")
    total_table, total_direct, total_kernel = grid_integrate(
        table, gamma, tx, ty, w, a0, s_edges, s_centers, args.phi_pol,
        args.grid_n, half_width, samples_per_point=args.samples_per_point)

    valid = ref > ref.max() * 0.02
    ratio_table = np.divide(total_table, ref, out=np.full_like(total_table, np.nan), where=valid)
    ratio_direct = np.divide(total_direct, ref, out=np.full_like(total_direct, np.nan), where=valid)
    ratio_kernel = np.divide(total_kernel, ref, out=np.full_like(total_kernel, np.nan), where=valid)
    print("grid-integrated spectrum_from_table / angle_integrated_spectrum:    "
          f"mean={np.nanmean(ratio_table):.4g}  std={np.nanstd(ratio_table):.4g}")
    print("grid-integrated direct_binning_spectrum / angle_integrated_spectrum: "
          f"mean={np.nanmean(ratio_direct):.4g}  std={np.nanstd(ratio_direct):.4g}")
    print("grid-integrated spectrum_kernel_4d / angle_integrated_spectrum:      "
          f"mean={np.nanmean(ratio_kernel):.4g}  std={np.nanstd(ratio_kernel):.4g}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

    ax1.plot(s_centers, ref, "o-", label="angle_integrated_spectrum (trusted)", ms=3, color="k")
    ax1.plot(s_centers, total_table, "s-", label="grid-integrated spectrum_from_table", ms=3, color="tab:blue")
    ax1.plot(s_centers, total_direct, "^-", label="grid-integrated direct_binning_spectrum", ms=3, color="tab:orange")
    ax1.plot(s_centers, total_kernel, "d-", label="grid-integrated spectrum_kernel_4d", ms=3, color="tab:green")
    ax1.set_yscale("log")
    ax1.set_ylabel("dN/ds (angle-integrated)")
    ax1.set_title(f"gamma0={args.gamma0:.4g}, sigma_thx={tx.std():.3g}, a0~{a0.mean():.3g}, "
                   f"grid {args.grid_n}x{args.grid_n} @ {args.grid_halfwidth_sigma:.3g}sigma")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.plot(s_centers, ratio_table, "s-", ms=3, color="tab:blue", label="table / trusted")
    ax2.plot(s_centers, ratio_direct, "^-", ms=3, color="tab:orange", label="direct / trusted")
    ax2.plot(s_centers, ratio_kernel, "d-", ms=3, color="tab:green", label="kernel_4d / trusted")
    ax2.axhline(1.0, color="k", lw=1, ls="--")
    ax2.set_ylabel("ratio to angle_integrated_spectrum")
    ax2.set_xlabel("s")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"saved plot to {args.out}")


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    compton = build_compton(args)
    sigma_gamma0 = args.sigma_gamma_rel * args.gamma0

    bunch = particles.sample_bunch(
        compton, args.n_particles, args.gamma0, sigma_gamma0,
        chirp=args.chirp, angle_energy_corr=args.angle_energy_corr, rng=rng,
    )
    gamma, tx, ty, a0, w = particles.push_and_sample(compton, bunch, n_steps=args.n_steps)

    print("=== bunch / laser diagnostics ===")
    print(f"compton.a0 (peak)   = {compton.a0:.4g}")
    print(f"theta_x std / 1/gamma0 = {tx.std():.4g} / {1.0/args.gamma0:.4g}  (ratio {tx.std()*args.gamma0:.3g})")
    print(f"a0 (ahat) mean/max  = {a0.mean():.4g} / {a0.max():.4g}")
    print(f"total weight (L)    = {w.sum():.6g}   estimate_yield() = {compton.estimate_yield():.6g}")

    table = deposition.build_table(gamma, tx, ty, a0, w, n_bins=tuple(args.n_bins),
                                    scheme=args.scheme, device="gpu")
    occ = deposition.occupancy_diagnostics(table)
    print(f"table empty_fraction = {occ['empty_fraction']:.3g}")
    print(f"gamma_bracket = {table.gamma_bracket}")

    s_center = args.gamma0**2
    s_lo, s_hi = args.s_range
    s_edges = np.linspace(s_lo * s_center, s_hi * s_center, args.n_s_bins + 1)
    s_centers = 0.5 * (s_edges[1:] + s_edges[:-1])

    if args.grid_integrate:
        run_grid_integrate(args, table, gamma, tx, ty, w, a0, s_edges, s_centers)
        return

    spec_table = reference.spectrum_from_table(table, args.x0, args.y0, s_centers, phi_pol=args.phi_pol)
    spec_direct = reference.direct_binning_spectrum(gamma, tx, ty, w, a0, args.x0, args.y0, s_edges, phi_pol=args.phi_pol)

    spec_kernel = None
    if args.also_kernel:
        s_gpu = cp.asarray(s_centers, dtype=cp.float32)
        theta_x_gpu = cp.asarray([args.x0], dtype=cp.float32)
        theta_y_gpu = cp.asarray([args.y0], dtype=cp.float32)
        spec_kernel_full, _, _ = spectrum4d.calculate_angular_spectrum_4d(
            table, s_gpu, theta_x_gpu, theta_y_gpu, phi_pol=args.phi_pol,
            samples_per_point=args.samples_per_point)
        spec_kernel = spec_kernel_full[0, 0, :]

    valid = spec_table > 0
    ratio = np.divide(spec_direct, spec_table, out=np.full_like(spec_direct, np.nan), where=valid)
    print("=== direct_binning_spectrum / spectrum_from_table ===")
    print(f"ratio: mean={np.nanmean(ratio):.4g}  std={np.nanstd(ratio):.4g}  "
          f"min={np.nanmin(ratio):.4g}  max={np.nanmax(ratio):.4g}")

    curves = [("spectrum_from_table", spec_table, "tab:blue"),
              ("direct_binning_spectrum", spec_direct, "tab:orange")]
    if spec_kernel is not None:
        curves.append(("spectrum_kernel_4d", cp.asnumpy(spec_kernel), "tab:green"))

    n_panels = len(curves) + 1  # +1 for the ratio panel
    fig, axes = plt.subplots(n_panels, 1, figsize=(7, 2.2 * n_panels), sharex=True)

    for ax, (label, data, color) in zip(axes, curves):
        ax.plot(s_centers, data, "o-", ms=3, color=color)
        ax.set_ylabel("dN/(ds dOmega)")
        ax.set_title(label, fontsize=10, loc="left")
        ax.grid(alpha=0.3)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    axes[0].set_title(f"x0={args.x0:.4g}, y0={args.y0:.4g}, gamma0={args.gamma0:.4g}, "
                       f"sigma_thx={tx.std():.3g}, a0~{a0.mean():.3g}\nspectrum_from_table",
                       fontsize=10, loc="left")

    ax_ratio = axes[-1]
    ax_ratio.plot(s_centers, ratio, "o-", ms=3, color="tab:red")
    ax_ratio.axhline(1.0, color="k", lw=1, ls="--")
    ax_ratio.set_yscale("log")
    ax_ratio.set_ylabel("direct / table")
    ax_ratio.set_title("direct / table (log scale)", fontsize=10, loc="left")
    ax_ratio.set_xlabel("s")

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"saved plot to {args.out}")


if __name__ == "__main__":
    main()
