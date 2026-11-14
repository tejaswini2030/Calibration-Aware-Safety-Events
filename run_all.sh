#!/usr/bin/env bash
# Reproduce the full Calibration-Aware Safety Events pipeline in order.
# Logs each step to outputs/logs/<step>.log and stops on the first failure.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

SKIP_PERCEPTION=0
DEVICE="mps"
# Predicted-track sweep checkpoints by default are discarded so a full run
# recomputes; pass --resume to keep the fast partial-run behaviour.
RESUME_SWEEPS=0

usage() {
  cat <<'EOF'
Usage: ./run_all.sh [--skip-perception] [--resume] [--device mps|cpu]

  --skip-perception   Reuse existing detection/track CSVs in results/
                      instead of rerunning YOLO and the trackers.
  --resume            Keep sweep checkpoints (e.g. sweep_C1_pred_tracks.csv)
                      and only run missing trials. Default is --force on
                      scripts/run_sweep_tracks.py so every trial is recomputed.
  --device mps|cpu    Passed to run_detection.py (default: mps).
  -h, --help          Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-perception) SKIP_PERCEPTION=1; shift ;;
    --resume) RESUME_SWEEPS=1; shift ;;
    --device)
      DEVICE="${2:-}"
      if [[ "$DEVICE" != "mps" && "$DEVICE" != "cpu" ]]; then
        echo "error: --device must be mps or cpu (got ${DEVICE:-empty})" >&2
        exit 2
      fi
      shift 2
      ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="$(command -v python3)"
else
  PYTHON="$(command -v python)"
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
mkdir -p outputs/logs "$MPLCONFIGDIR"

STEP_NAMES=()
STEP_STATUSES=()
STEP_SECONDS=()
TOTAL_START=$SECONDS
FAILED=0

format_duration() {
  local s=$1
  if (( s < 60 )); then
    printf "%ds" "$s"
  elif (( s < 3600 )); then
    printf "%dm %ds" "$((s / 60))" "$((s % 60))"
  else
    printf "%dh %dm %ds" "$((s / 3600))" "$(((s % 3600) / 60))" "$((s % 60))"
  fi
}

print_summary() {
  local i name status secs total
  echo
  echo "================ summary ================"
  printf "%-28s  %-6s  %s\n" "step" "status" "time"
  for i in "${!STEP_NAMES[@]}"; do
    printf "%-28s  %-6s  %s\n" \
      "${STEP_NAMES[$i]}" \
      "${STEP_STATUSES[$i]}" \
      "$(format_duration "${STEP_SECONDS[$i]}")"
  done
  total=$((SECONDS - TOTAL_START))
  echo "----------------------------------------"
  printf "%-28s  %-6s  %s\n" "TOTAL" "$([[ $FAILED -eq 0 ]] && echo PASS || echo FAIL)" "$(format_duration "$total")"
  echo "logs: outputs/logs/"
}

run_step() {
  local name="$1"
  shift
  local log="outputs/logs/${name}.log"
  local start=$SECONDS
  local rc=0
  local elapsed status

  echo "→ ${name} ..."
  set +e
  "$@" >"$log" 2>&1
  rc=$?
  set -e
  elapsed=$((SECONDS - start))

  if [[ $rc -eq 0 ]]; then
    status="PASS"
  else
    status="FAIL"
    FAILED=1
  fi

  STEP_NAMES+=("$name")
  STEP_STATUSES+=("$status")
  STEP_SECONDS+=("$elapsed")

  printf "  %-6s  %s  (%s)  log=%s\n" "$status" "$name" "$(format_duration "$elapsed")" "$log"

  if [[ $rc -ne 0 ]]; then
    echo "  last lines of ${log}:" >&2
    tail -n 20 "$log" >&2 || true
    print_summary
    exit "$rc"
  fi
}

SWEEP_TRACKS_ARGS=()
if [[ "$RESUME_SWEEPS" -eq 0 ]]; then
  SWEEP_TRACKS_ARGS+=(--force)
fi

echo "python: $PYTHON"
echo "device: $DEVICE"
echo "skip-perception: $SKIP_PERCEPTION"
echo "sweep-tracks: $([[ $RESUME_SWEEPS -eq 1 ]] && echo resume || echo force)"
echo

# 1. Tests
run_step pytest "$PYTHON" -m pytest

# 2. Dataset / calibration checks
run_step inspect_dataset "$PYTHON" scripts/inspect_dataset.py
run_step check_annotations "$PYTHON" scripts/check_annotations.py
run_step check_calibration "$PYTHON" scripts/check_calibration.py
run_step check_homography "$PYTHON" scripts/check_homography.py

# 3. Zones and answer keys (C1 + C6)
run_step show_zones_C1 "$PYTHON" scripts/show_zones.py --cam C1
run_step make_gt_events_C1 "$PYTHON" scripts/make_gt_events.py --cam C1
run_step show_zones_C6 "$PYTHON" scripts/show_zones.py --cam C6
run_step make_gt_events_C6 "$PYTHON" scripts/make_gt_events.py --cam C6
run_step check_scoring "$PYTHON" scripts/check_scoring.py

# 4. Zero-noise sanity
run_step check_zero_noise "$PYTHON" scripts/check_zero_noise.py

# 5. Calibration sweeps
run_step run_sweep "$PYTHON" scripts/run_sweep.py
run_step run_sweep_npoints_C1 "$PYTHON" scripts/run_sweep_npoints.py --cam C1
run_step run_sweep_npoints_C6 "$PYTHON" scripts/run_sweep_npoints.py --cam C6
run_step run_sweep_outlier "$PYTHON" scripts/run_sweep_outlier.py
run_step run_sweep_layouts "$PYTHON" scripts/run_sweep_layouts.py
run_step run_sweep_bump_C1 "$PYTHON" scripts/run_sweep_bump.py --cam C1
run_step run_sweep_bump_C6 "$PYTHON" scripts/run_sweep_bump.py --cam C6
run_step save_calib_points_C6 "$PYTHON" scripts/save_calib_points_C6.py
run_step compare_C1_C6 "$PYTHON" scripts/compare_C1_C6.py

# 6. Perception (optional)
if [[ "$SKIP_PERCEPTION" -eq 0 ]]; then
  run_step run_detection "$PYTHON" scripts/run_detection.py --device "$DEVICE"
  run_step run_tracking "$PYTHON" scripts/run_tracking.py
else
  echo "→ perception skipped (--skip-perception); reusing results/"
fi

# 7. Tracks → events → calibration on predicted tracks
run_step check_tracks "$PYTHON" scripts/check_tracks.py
run_step diagnose_foot_points "$PYTHON" scripts/diagnose_foot_points.py
run_step run_events_on_tracks "$PYTHON" scripts/run_events_on_tracks.py
run_step run_sweep_tracks "$PYTHON" scripts/run_sweep_tracks.py "${SWEEP_TRACKS_ARGS[@]}"

# 8. Report figures
run_step make_report_figures "$PYTHON" scripts/make_report_figures.py

print_summary
