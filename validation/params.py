"""Shared representative Compton-source parameter set (plan.md's Sec. 0).

Every figure/bench script imports from here so they all describe the same
physical case. The numbers are the ones already shipped as
`example-config.toml`/`calculate-spec-ang.py` (gamma0=2000, 10nC, 20J
1030nm laser) -- a sensible, already-exercised operating point, not a new
choice invented for this study.

`build_compton()` returns a freshly configured `xigma_i.Compton` instance
(cheap; `set_*` calls just store scalars and invalidate `.intersection`).
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from xigma_i import Compton

c = 2.99792458e10  # cm/s

ELECTRON = dict(
    chargeNC=10.0,          # nC
    gamma0=2000.0,
    sigma_gamma_rel=0.005,  # sigma_gamma0 / gamma0
    sigma_x=1e-3,           # cm
    sigma_y=1e-3,           # cm
    duration=10e-12,        # s (-> sigma_ez = duration * c)
    norm_emit_x=1e-4,       # cm*rad
    norm_emit_y=0.01e-4,    # cm*rad
    focus=(0.0, 0.0, 0.0),
)

LASER = dict(
    energy=20.0,            # J
    lambda_l=1030e-7,       # cm
    sigma_r=10e-4,          # cm (RMS radius of the photon-density distribution)
    duration=30e-12,        # s (-> sigma_lz = duration * c)
    polarization_angle=0.0, # deg
    beta_ff=0.0,
)

GAMMA0 = ELECTRON["gamma0"]
SIGMA_GAMMA0 = ELECTRON["sigma_gamma_rel"] * GAMMA0
PHI_POL = np.deg2rad(LASER["polarization_angle"])

# Peak of the on-axis resonance, s = gamma^2/(1+gamma^2 theta^2+ahat); used to
# scale observation grids and frequency windows consistently across scripts.
S_PEAK = GAMMA0**2

# Representative observation points (theta_x, theta_y), in units of 1/gamma0,
# spanning on-axis / mid-spectrum / near the Compton edge (large angle ->
# strongly red-shifted s). Shared by fig_sampling.py and fig_validation.py.
OBS_POINTS = {
    "on_axis": (0.0, 0.0),
    "mid": (0.5 / GAMMA0, 0.0),
    "edge": (0.95 / GAMMA0, 0.0),
}

# --------------------------------------------------------------------------
# Hardware-scaled settings tiers.
#
# All the numbers below (table sizes, particle counts, chunk sizes) trade
# statistical/resolution quality against GPU memory and wall-clock. Rather
# than hand-tuning one fixed set that happens to fit whatever machine this
# was written on, tier selection is auto-detected from the GPU's total
# memory (and, as a secondary cap, system RAM -- particles.push_and_sample's
# *internal* trajectory-integration arrays are O(n_particles*n_steps), even
# though what it returns is only O(n_particles); see refs.py), so the *same* code runs
# a smoke-sized sweep on a 6 GB laptop GPU and a substantially larger,
# statistically solid sweep on a data-center card with no manual editing.
#
# Override with the VALIDATION_SCALE env var ("small"/"medium"/"large") to
# force a tier -- e.g. if auto-detection picks the wrong device, or you want
# a smaller run on a big machine to iterate faster. VALIDATION_GPU_GB /
# VALIDATION_RAM_GB override the detected memory sizes feeding auto-selection.
#
# These tier numbers are a first calibration, sized with a comfortable
# safety margin (roughly half the tier's memory ceiling actually used at
# peak) rather than pushed to the edge -- if you have room to spare on a
# large card, VALIDATION_SCALE=large plus manually bumping FINE_N_PARTICLES/
# DEFAULT_N_PARTICLES afterwards is reasonable; going *past* what's tested
# here has not been.
# --------------------------------------------------------------------------


def _gpu_mem_gb():
    try:
        import cupy as cp
        _, total = cp.cuda.Device(0).mem_info
        return total / 1e9
    except Exception:
        return None


def _sys_ram_gb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / 1e6  # kB -> GB
    except Exception:
        pass
    return None


_GPU_GB = float(os.environ.get("VALIDATION_GPU_GB", "") or (_gpu_mem_gb() or 6.0))
_RAM_GB = float(os.environ.get("VALIDATION_RAM_GB", "") or (_sys_ram_gb() or 8.0))
RAM_GB = _RAM_GB  # public alias -- fig_validation.py is CPU/RAM-bound (no table, no GPU), not GPU-tier-scaled

_TIERS = {
    # This repo's dev box: GTX 1660 Ti, 6 GB.
    "small": dict(
        DEFAULT_N_BINS=(128, 128, 128, 32), DEFAULT_N_PARTICLES=200_000, DEFAULT_N_STEPS=64,
        FINE_N_BINS=(192, 192, 192, 8), FINE_N_PARTICLES=1_000_000, FINE_N_STEPS=96,
        STREAM_CHUNK_PARTICLES=20_000,
        GRIDRES_N_PARTICLES=300_000, GRIDRES_N_STEPS=64,
        DEPOSITION_NP_VALUES=[1_000, 4_000, 16_000, 64_000, 150_000, 300_000],
        DEPOSITION_NP_FIXED_FOR_DT=150_000,
        SAMPLING_M_VALUES=[4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144],
        BENCH_NP_VALUES=[1_000, 10_000, 100_000, 300_000],
    ),
    # e.g. RTX 3080/4070-class, 10-20 GB.
    "medium": dict(
        DEFAULT_N_BINS=(128, 128, 128, 32), DEFAULT_N_PARTICLES=500_000, DEFAULT_N_STEPS=64,
        FINE_N_BINS=(224, 224, 224, 16), FINE_N_PARTICLES=3_000_000, FINE_N_STEPS=96,
        STREAM_CHUNK_PARTICLES=50_000,
        GRIDRES_N_PARTICLES=1_500_000, GRIDRES_N_STEPS=64,
        DEPOSITION_NP_VALUES=[2_000, 10_000, 50_000, 250_000, 1_000_000, 2_000_000],
        DEPOSITION_NP_FIXED_FOR_DT=500_000,
        SAMPLING_M_VALUES=[4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1_048_576],
        BENCH_NP_VALUES=[1_000, 10_000, 100_000, 1_000_000],
    ),
    # RTX 4090/A100/H100-class, >=24 GB. A single conservative preset for
    # every card in this range rather than trying to size specifically for
    # e.g. an 80 GB H100 -- there is a lot of headroom left on this tier on
    # the largest cards; see the docstring above about pushing further.
    "large": dict(
        DEFAULT_N_BINS=(160, 160, 160, 32), DEFAULT_N_PARTICLES=1_500_000, DEFAULT_N_STEPS=96,
        FINE_N_BINS=(256, 256, 256, 24), FINE_N_PARTICLES=8_000_000, FINE_N_STEPS=128,
        STREAM_CHUNK_PARTICLES=200_000,
        GRIDRES_N_PARTICLES=5_000_000, GRIDRES_N_STEPS=96,
        DEPOSITION_NP_VALUES=[5_000, 25_000, 125_000, 625_000, 3_000_000, 6_000_000],
        DEPOSITION_NP_FIXED_FOR_DT=2_000_000,
        SAMPLING_M_VALUES=[4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1_048_576, 4_194_304],
        # bench_deposition times a *single* un-chunked deposit_cic call (the
        # point is raw kernel throughput, not correctness at arbitrary N_p --
        # unlike the deposition-scheme scan above, which streams/chunks and so
        # has no such ceiling); kept well short of what would need chunking.
        BENCH_NP_VALUES=[1_000, 10_000, 100_000, 1_000_000, 2_000_000],
    ),
}


def _pick_tier():
    forced = os.environ.get("VALIDATION_SCALE", "").strip().lower()
    if forced:
        if forced not in _TIERS:
            raise ValueError(f"VALIDATION_SCALE must be one of {list(_TIERS)}, got {forced!r}")
        return forced
    if _GPU_GB >= 24:
        return "large"
    if _GPU_GB >= 10:
        return "medium"
    return "small"


SCALE_TIER = _pick_tier()
_settings = _TIERS[SCALE_TIER]

# Default table grid + particle-count/sample budget "production" settings
# (Sec. 0: "the default table grid + sample budget"). Individual figures vary
# one of these at a time; everything else is held at these defaults.
DEFAULT_N_BINS = _settings["DEFAULT_N_BINS"]      # (gamma, theta_x, theta_y, a0)
DEFAULT_N_PARTICLES = _settings["DEFAULT_N_PARTICLES"]
DEFAULT_N_STEPS = _settings["DEFAULT_N_STEPS"]
DEFAULT_SAMPLES_PER_POINT = 32
DEFAULT_DEPOSITION_SCHEME = "cic"

# "Very high sample budget / fine grid" reference table (refs.ref_quadrature),
# reused as ground truth by Figs. 1-3. Finer than DEFAULT_* in gamma/theta_x/
# theta_y (the axes Figs. 1-3 actually probe); the a0 axis is deliberately
# left *coarse* here, not fine -- spectrum_from_table's resonance-condition
# inversion (g_sq = 1/(1/s - r_sq)) does not depend on a0 at all (see
# fig_gridres.py's module docstring/output), so extra a0 bins in the
# reference buy nothing for these figures and only cost GPU memory better
# spent on gamma/theta_x/theta_y.
FINE_N_BINS = _settings["FINE_N_BINS"]
FINE_N_PARTICLES = _settings["FINE_N_PARTICLES"]
FINE_N_STEPS = _settings["FINE_N_STEPS"]
STREAM_CHUNK_PARTICLES = _settings["STREAM_CHUNK_PARTICLES"]

# fig_gridres.py's fixed particle-sample-set size (reused across every grid
# variant in the scan, so its own count doesn't matter beyond "big enough
# that redeposition noise is small") and fig_deposition.py's N_p/n_steps
# sweep points and fixed-N_p-for-the-dt-scan value.
GRIDRES_N_PARTICLES = _settings["GRIDRES_N_PARTICLES"]
GRIDRES_N_STEPS = _settings["GRIDRES_N_STEPS"]
DEPOSITION_NP_VALUES = _settings["DEPOSITION_NP_VALUES"]
DEPOSITION_NP_FIXED_FOR_DT = _settings["DEPOSITION_NP_FIXED_FOR_DT"]

# fig_sampling.py's M sweep (samples per output point) and bench.py's N_p sweep.
SAMPLING_M_VALUES = _settings["SAMPLING_M_VALUES"]
BENCH_NP_VALUES = _settings["BENCH_NP_VALUES"]

# Reporting resolution mu (in s-units) used by metrics.window_integrated_relative_error
# unless a script overrides it -- see metrics.py docstring for the convention.
DEFAULT_MU = 2e-3 * S_PEAK

# Default random seed for every synthetic bunch -- deterministic sweeps.
DEFAULT_SEED = 20260721


def build_compton():
    """A freshly configured Compton instance for ELECTRON/LASER above."""
    e = ELECTRON
    emit_x = e["norm_emit_x"] / e["gamma0"]
    emit_y = e["norm_emit_y"] / e["gamma0"]
    sigma_ez = e["duration"] * c

    compton = Compton()
    compton.set_electron_parameters(
        chargeNC=e["chargeNC"], emit_x=emit_x, emit_y=emit_y,
        sigma_ex=e["sigma_x"], sigma_ey=e["sigma_y"], sigma_ez=sigma_ez,
    )

    l = LASER
    sigma_lz = l["duration"] * c
    compton.set_laser_parameters(
        WL=l["energy"], lambda_l=l["lambda_l"], sigma_lr0=l["sigma_r"],
        sigma_lz=sigma_lz, beta_ff=l["beta_ff"],
    )
    compton.set_foci_displacement(*e["focus"])
    return compton


def s_grid(n=512, lo=1e-3, hi=1.05):
    """Default frequency grid, in units of S_PEAK, covering the resonance peak
    and a margin past the Compton edge. lo > 0 strictly: s=0 is a removable
    singularity of the g_sq/output-scaling algebra (1/s, /s**2), not a
    physically meaningful point of the spectrum.
    """
    return (np.linspace(lo, hi, n) * S_PEAK).astype(np.float64)
