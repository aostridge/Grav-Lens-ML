#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh — End-to-end pipeline runner with per-stage timing.
#
# Runs stages 1–4 (data generation → feature extraction → training →
# evaluation). fit_lens_model.py is excluded as it requires real
# observational data.
#
# Usage (foreground):
#   bash run_pipeline.sh
#
# Usage (background — survives terminal close):
#   nohup bash run_pipeline.sh > pipeline_run.log 2>&1 &
#   tail -f pipeline_run.log      # monitor progress
# =============================================================================

set -euo pipefail

# =============================================================================
# Paths
# =============================================================================
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$REPO_DIR/.venv"
PUB_DIR="$REPO_DIR/pipeline"
LOG_DIR="$REPO_DIR/pipeline_logs"

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

# Arrays to accumulate stage results for the final summary.
declare -a STAGE_NAMES
declare -a STAGE_ELAPSED_SECS
declare -a STAGE_STATUS

fmt_duration() {
    # Convert seconds to HH:MM:SS string.
    local total=$1
    printf "%02dh %02dm %02ds" $((total/3600)) $(( (total%3600)/60 )) $((total%60))
}

run_stage() {
    local label="$1"   # human-readable name
    local script="$2"  # filename inside pipeline/
    local log="$LOG_DIR/${script%.py}.log"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Stage: $label"
    echo "  Script: $script"
    echo "  Log:    $log"
    echo "  Start:  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    local stage_start=$SECONDS
    local exit_code=0

    # Run from pipeline/ so `from columns import` resolves.
    (cd "$PUB_DIR" && "$PYTHON" "$script") > "$log" 2>&1 || exit_code=$?

    local elapsed=$(( SECONDS - stage_start ))
    local end_ts
    end_ts=$(date '+%Y-%m-%d %H:%M:%S')

    STAGE_NAMES+=("$label")
    STAGE_ELAPSED_SECS+=("$elapsed")

    if [[ $exit_code -ne 0 ]]; then
        STAGE_STATUS+=("FAILED (exit $exit_code)")
        echo "  End:     $end_ts"
        echo "  Elapsed: $(fmt_duration $elapsed)"
        echo "  STATUS:  FAILED (exit code $exit_code)"
        echo "  See log: $log"
        echo ""
        echo "Last 20 lines of log:"
        tail -20 "$log"
        exit $exit_code
    else
        STAGE_STATUS+=("OK")
        echo "  End:     $end_ts"
        echo "  Elapsed: $(fmt_duration $elapsed)"
        echo "  STATUS:  OK"
    fi
}

# =============================================================================
# Hardware & environment info
# =============================================================================
PIPELINE_START=$SECONDS
RUN_DATE=$(date '+%Y-%m-%d %H:%M:%S')

echo "============================================================"
echo "  GRAVITATIONAL LENS PIPELINE RUN"
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

# GPU: check via nvidia-smi if present, then report TF's view
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

# Definitive GPU check: ask TensorFlow directly.
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

# =============================================================================
# Pipeline stages (fit_lens_model.py excluded — requires observational data)
# =============================================================================
run_stage "1 — Generate Training Data"  "generate_training_data.py"
run_stage "2 — Compute Features"        "compute_features.py"
run_stage "3 — Train Networks"          "train_networks.py"
run_stage "4 — Evaluate Predictions"    "evaluate_predictions.py"

# =============================================================================
# Summary
# =============================================================================
TOTAL_ELAPSED=$(( SECONDS - PIPELINE_START ))

echo ""
echo "============================================================"
echo "  PIPELINE COMPLETE"
echo "  Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""
echo "┌──────────────────────────────────────┬────────────────┬────────┐"
echo "│ Stage                                │ Elapsed        │ Status │"
echo "├──────────────────────────────────────┼────────────────┼────────┤"
for i in "${!STAGE_NAMES[@]}"; do
    printf "│ %-36s │ %-14s │ %-6s │\n" \
        "${STAGE_NAMES[$i]}" \
        "$(fmt_duration "${STAGE_ELAPSED_SECS[$i]}")" \
        "${STAGE_STATUS[$i]}"
done
echo "├──────────────────────────────────────┼────────────────┼────────┤"
printf "│ %-36s │ %-14s │ %-6s │\n" "TOTAL" "$(fmt_duration $TOTAL_ELAPSED)" ""
echo "└──────────────────────────────────────┴────────────────┴────────┘"
echo ""
echo "Hardware:     $CPU_MODEL ($CPU_CORES cores, $RAM_TOTAL RAM)"
echo "TF compute:   $TF_GPU_INFO"
echo "Log files:    $LOG_DIR/"
