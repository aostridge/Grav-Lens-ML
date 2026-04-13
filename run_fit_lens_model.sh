#!/usr/bin/env bash
# =============================================================================
# run_fit_lens_model.sh — Lens model fitting runner with timing.
#
# Runs fit_lens_model.py, which applies the trained neural networks to real
# observational image data and iteratively optimises the lens model via
# lensmodel until χ² converges.
#
# Prerequisites:
#   • Trained model files must exist in data/models/:
#       model_mass_parameter.keras
#       model_ellipticity.keras
#   • Observational data file must exist at:
#       data/observations/quad_lens_observations.txt
#         Format (per system):
#           #<system_name>
#           RA  Dec  ErrorRA  ErrorDec  Flux  FluxError
#           <image A values>
#           <image B values>
#           <image C values>
#           <image D values>
#         Units: set INPUT_COORDINATE_UNITS in fit_lens_model.py
#           'degrees'    — absolute sky coordinates (converted to arcsec internally)
#           'arcseconds' — relative arcsecond offsets (no conversion applied)
#
# Usage (foreground):
#   bash run_fit_lens_model.sh
#
# Usage (background — survives terminal close):
#   nohup bash run_fit_lens_model.sh > fit_run.log 2>&1 &
#   tail -f fit_run.log      # monitor progress
# =============================================================================

set -euo pipefail

# =============================================================================
# Paths
# =============================================================================
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$REPO_DIR/.venv"
PUB_DIR="$REPO_DIR/pipeline"
LOG_DIR="$REPO_DIR/pipeline_logs"
LOG="$LOG_DIR/fit_lens_model.log"

# =============================================================================
# Setup
# =============================================================================
mkdir -p "$LOG_DIR"

if [[ ! -f "$VENV/bin/activate" ]]; then
    echo "ERROR: Virtual environment not found at $VENV" >&2
    exit 1
fi
# shellcheck source=/dev/null
source "$VENV/bin/activate"

PYTHON="$VENV/bin/python3"

# =============================================================================
# Timing helpers
# =============================================================================
fmt_duration() {
    local total=$1
    printf "%02dh %02dm %02ds" $((total/3600)) $(( (total%3600)/60 )) $((total%60))
}

# =============================================================================
# Hardware & environment info
# =============================================================================
RUN_START=$SECONDS
RUN_DATE=$(date '+%Y-%m-%d %H:%M:%S')

echo "============================================================"
echo "  GRAVITATIONAL LENS — FIT LENS MODEL"
echo "  Started: $RUN_DATE"
echo "============================================================"
echo ""
echo "── Hardware ─────────────────────────────────────────────"
CPU_MODEL=$(lscpu | grep "Model name" | sed 's/Model name:\s*//' | xargs)
CPU_CORES=$(lscpu | grep "^CPU(s):" | awk '{print $2}')
CPU_THREADS=$(lscpu | grep "^Thread(s) per core:" | awk '{print $4}')
RAM_TOTAL=$(free -h | awk '/^Mem:/{print $2}')
echo "  CPU:     $CPU_MODEL"
echo "  Cores:   $CPU_CORES logical ($CPU_THREADS thread(s) per core)"
echo "  RAM:     $RAM_TOTAL"

if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "nvidia-smi query failed")
    echo "  GPU:     $GPU_NAME (nvidia-smi)"
else
    echo "  GPU:     No NVIDIA GPU detected (nvidia-smi not found)"
fi

echo ""
echo "── Python & TensorFlow ──────────────────────────────────"
PY_VERSION=$("$PYTHON" --version 2>&1)
TF_VERSION=$("$PYTHON" -c "import tensorflow as tf; print(tf.__version__)" 2>/dev/null || echo "TensorFlow not importable")
echo "  $PY_VERSION"
echo "  TensorFlow $TF_VERSION"

TF_GPU_INFO=$("$PYTHON" - <<'PYEOF' 2>/dev/null
import os, logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    names = ', '.join(d.name for d in gpus)
    print(f"GPU AVAILABLE — TensorFlow will use: {names}")
else:
    print("NO GPU — TensorFlow will run on CPU only")
PYEOF
)
echo "  TF GPU:  $TF_GPU_INFO"

echo ""
echo "── Input ────────────────────────────────────────────────"
OBS_FILE="$REPO_DIR/data/observations/quad_lens_observations.txt"
MASS_MODEL="$REPO_DIR/data/models/model_mass_parameter.keras"
ELLIP_MODEL="$REPO_DIR/data/models/model_ellipticity.keras"

if [[ ! -f "$OBS_FILE" ]]; then
    echo "  ERROR: Observations file not found:" >&2
    echo "         $OBS_FILE" >&2
    exit 1
fi
if [[ ! -f "$MASS_MODEL" ]]; then
    echo "  ERROR: Mass model not found:" >&2
    echo "         $MASS_MODEL" >&2
    exit 1
fi
if [[ ! -f "$ELLIP_MODEL" ]]; then
    echo "  ERROR: Ellipticity model not found:" >&2
    echo "         $ELLIP_MODEL" >&2
    exit 1
fi

N_SYSTEMS=$(grep -c '^#' "$OBS_FILE" || true)
echo "  Observations: $OBS_FILE"
echo "  Systems:      $N_SYSTEMS"
echo "  Mass model:   $MASS_MODEL"
echo "  Ellip model:  $ELLIP_MODEL"
echo "  Log:          $LOG"

# =============================================================================
# Run
# =============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Stage: Fit Lens Model"
echo "  Script: fit_lens_model.py"
echo "  Start:  $(date '+%Y-%m-%d %H:%M:%S')"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

STAGE_START=$SECONDS
EXIT_CODE=0

(cd "$PUB_DIR" && "$PYTHON" fit_lens_model.py) > "$LOG" 2>&1 || EXIT_CODE=$?

ELAPSED=$(( SECONDS - STAGE_START ))
TOTAL_ELAPSED=$(( SECONDS - RUN_START ))

echo "  End:     $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Elapsed: $(fmt_duration $ELAPSED)"

if [[ $EXIT_CODE -ne 0 ]]; then
    echo "  STATUS:  FAILED (exit code $EXIT_CODE)"
    echo "  See log: $LOG"
    echo ""
    echo "Last 20 lines of log:"
    tail -20 "$LOG"
    exit $EXIT_CODE
fi

echo "  STATUS:  OK"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "============================================================"
echo "  FIT LENS MODEL COMPLETE"
echo "  Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""
echo "┌──────────────────────────────────────┬────────────────┬────────┐"
echo "│ Stage                                │ Elapsed        │ Status │"
echo "├──────────────────────────────────────┼────────────────┼────────┤"
printf "│ %-36s │ %-14s │ %-6s │\n" "Fit Lens Model" "$(fmt_duration $ELAPSED)" "OK"
echo "├──────────────────────────────────────┼────────────────┼────────┤"
printf "│ %-36s │ %-14s │ %-6s │\n" "TOTAL" "$(fmt_duration $TOTAL_ELAPSED)" ""
echo "└──────────────────────────────────────┴────────────────┴────────┘"
echo ""
echo "Hardware:     $CPU_MODEL ($CPU_CORES cores, $RAM_TOTAL RAM)"
echo "TF compute:   $TF_GPU_INFO"
echo "Results:      $REPO_DIR/data/observations/Results_quad_lens_observations/"
echo "Log file:     $LOG"
