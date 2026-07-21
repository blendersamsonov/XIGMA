"""PRAB single-column sizing, consistent colours/markers, vector PDF output.

Import `apply()` once at the top of a figure script, then use `save(fig, name)`
to write `figs/<name>.pdf`.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIGS_DIR = HERE / "figs"
DATA_DIR = HERE / "data"

# APS/PRAB single-column width is 3.375 in; leave a little headroom for labels.
COL_WIDTH_IN = 3.4

COLORS = {
    "fibonacci": "#1b6ca8",
    "pseudorandom": "#c9622f",
    "nearest": "#c9622f",
    "cic": "#1b6ca8",
    "reference": "#444444",
    "gamma": "#1b6ca8",
    "a0": "#c9622f",
    "theta": "#4f9e5a",
    "delta_model": "#1b6ca8",
    "spectral_integration": "#c9622f",
}

MARKERS = {
    "fibonacci": "o",
    "pseudorandom": "s",
    "nearest": "s",
    "cic": "o",
    "gamma": "o",
    "a0": "s",
    "theta": "^",
}


def apply(width_in=COL_WIDTH_IN, aspect=0.8):
    plt.rcParams.update({
        "figure.figsize": (width_in, width_in * aspect),
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.3,
        "lines.markersize": 4.0,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    })


def save(fig, name, figs_dir=None):
    figs_dir = Path(figs_dir) if figs_dir is not None else FIGS_DIR
    figs_dir.mkdir(parents=True, exist_ok=True)
    path = figs_dir / f"{name}.pdf"
    fig.savefig(path)
    return path


def slope_guide(ax, x, y0, exponent, label, color="0.4"):
    """Draw a reference power-law slope through (x[0], y0) on a log-log axes."""
    import numpy as np
    x = np.asarray(x, dtype=float)
    y = y0 * (x / x[0]) ** exponent
    ax.plot(x, y, "--", color=color, linewidth=1.0, label=label)
