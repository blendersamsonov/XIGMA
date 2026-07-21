#!/usr/bin/env bash
# Run the full (non---quick) convergence/validation pipeline: Figs. 1-4 plus
# bench.py, in the order plan.md's build order implies (cheap/shared
# infrastructure first, Fig. 4 and bench last since neither is a dependency
# of the others). Meant for a machine with a substantially bigger GPU than
# this repo's 6 GB dev box -- params.py auto-detects GPU memory (and system
# RAM) at import time and picks a "small"/"medium"/"large" settings tier
# accordingly (table sizes, particle counts, sample budgets), so this same
# script produces a coarse smoke-sized sweep on a laptop GPU and a much
# larger, more statistically solid sweep on a data-center card with no
# editing required.
#
# Usage:
#   ./run_all.sh                    # auto-detect GPU/RAM, run everything
#   VALIDATION_SCALE=large ./run_all.sh   # force a tier instead of auto-detecting
#   VALIDATION_GPU_GB=40 ./run_all.sh     # force the *detected* GPU size (tier
#                                          # still chosen from this) -- use if
#                                          # auto-detection picks the wrong
#                                          # device in a multi-GPU box
#
# Tiers (see params.py's _TIERS for the exact numbers and the reasoning
# behind each): "small" <10 GB, "medium" 10-24 GB, "large" >=24 GB. "large" is
# one conservative preset shared by everything from a 24 GB card up to an
# 80 GB one -- see params.py's module docstring before pushing sizes further
# on the biggest cards.
#
# Each step's own --quick flag is intentionally *not* passed here: this
# script is the "full mode" runner. For a fast smoke test of the whole
# pipeline (a few minutes, any GPU) run the five scripts individually with
# --quick instead.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

# Activate the project's conda/mamba environment if one is named "xigma" and
# not already active; otherwise assume the caller has already activated
# whatever environment has cupy/numpy/matplotlib installed (see CLAUDE.md's
# "Environment" section -- no requirements.txt is shipped, this repo has been
# run out of a mamba env called "xigma").
if [[ "${CONDA_DEFAULT_ENV:-}" != "xigma" ]] && command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate xigma 2>/dev/null || echo "note: no 'xigma' conda env found; continuing with the current environment ($(command -v python))"
fi

python - <<'PY'
import cupy as cp
import params as P
free, total = cp.cuda.Device(0).mem_info
print(f"[run_all] GPU: {cp.cuda.Device(0).name if hasattr(cp.cuda.Device(0), 'name') else '(unknown)'} "
      f"-- {total/1e9:.1f} GB total, {free/1e9:.1f} GB free")
print(f"[run_all] settings tier: {P.SCALE_TIER}  (override with VALIDATION_SCALE=small|medium|large)")
print(f"[run_all] DEFAULT_N_BINS={P.DEFAULT_N_BINS} DEFAULT_N_PARTICLES={P.DEFAULT_N_PARTICLES}")
print(f"[run_all] FINE_N_BINS={P.FINE_N_BINS} FINE_N_PARTICLES={P.FINE_N_PARTICLES}")
PY

LOG=run_all.log
: > "$LOG"

run_step () {
    local name="$1"
    echo "=== $name ===" | tee -a "$LOG"
    local t0=$SECONDS
    if python "$name.py" >>"$LOG" 2>&1; then
        echo "    ok  ($((SECONDS - t0))s)" | tee -a "$LOG"
    else
        echo "    FAILED ($((SECONDS - t0))s) -- see $LOG" | tee -a "$LOG"
        return 1
    fi
}

status=0
run_step fig_sampling   || status=1
run_step fig_gridres    || status=1
run_step fig_deposition || status=1
run_step fig_validation || status=1
run_step bench          || status=1

echo "=== done (status=$status) ===" | tee -a "$LOG"
echo "figures: $(pwd)/figs/*.pdf"
echo "raw data: $(pwd)/data/*.npz"
echo "full log: $(pwd)/$LOG"
exit $status
