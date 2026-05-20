#!/usr/bin/env bash
set -euo pipefail

# Experiment 1: Baseline Comparison (NPL-ready)
# Methods: LLM+Episodic, LLM_only, Bayesian, TPE, TPE+MV, TPE+Pruner, Random
# Default GPU split (3 GPUs):
#   stream A -> methods 1,4,7
#   stream B -> methods 2,5
#   stream C -> methods 3,6
# Usage:
#   bash scripts/run_exp1_baseline.sh --gpus 0,1,2 --python python

GPUS_CSV="0,1,2"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="configs/paper/exp1_baseline_comparison.json"
OUTPUT_DIR="results/runs/exp1_baseline"
N_TRIALS_OVERRIDE=""
N_SEEDS_OVERRIDE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -g|--gpu)
      GPUS_CSV="${2:-0}"
      shift 2
      ;;
    --gpus)
      GPUS_CSV="${2:-0,1,2}"
      shift 2
      ;;
    -p|--python)
      PYTHON_BIN="${2:-python}"
      shift 2
      ;;
    --config)
      CONFIG_PATH="${2}"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="${2}"
      shift 2
      ;;
    --n-trials)
      N_TRIALS_OVERRIDE="${2}"
      shift 2
      ;;
    --n-seeds)
      N_SEEDS_OVERRIDE="${2}"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN"
  exit 1
fi

cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
if [[ ${#GPUS[@]} -eq 0 ]]; then
  echo "No GPUs parsed from: $GPUS_CSV"
  exit 1
fi
for i in "${!GPUS[@]}"; do
  GPUS[$i]="${GPUS[$i]// /}"
done

gpu_for_slot() {
  local slot="$1"
  local n="${#GPUS[@]}"
  echo "${GPUS[$((slot % n))]}"
}

GPU_A="$(gpu_for_slot 0)"
GPU_B="$(gpu_for_slot 1)"
GPU_C="$(gpu_for_slot 2)"

mkdir -p "$OUTPUT_DIR"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH"
  exit 1
fi

NORMALIZED_CONFIG="$OUTPUT_DIR/_tmp_exp1_config_utf8.json"
"$PYTHON_BIN" - <<PY
import json
src = r"$CONFIG_PATH"
dst = r"$NORMALIZED_CONFIG"
with open(src, "r", encoding="utf-8-sig") as f:
    cfg = json.load(f)
with open(dst, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2, ensure_ascii=False)
PY
CONFIG_PATH="$NORMALIZED_CONFIG"
trap 'rm -f "$NORMALIZED_CONFIG"' EXIT

COMMON_ARGS=(--config "$CONFIG_PATH")
if [[ -n "$N_TRIALS_OVERRIDE" ]]; then
  COMMON_ARGS+=(--n-trials "$N_TRIALS_OVERRIDE")
fi
if [[ -n "$N_SEEDS_OVERRIDE" ]]; then
  COMMON_ARGS+=(--n-seeds "$N_SEEDS_OVERRIDE")
fi

echo "=========================================="
echo "Experiment 1: Baseline Comparison"
echo "GPUs: $GPUS_CSV | Python: $PYTHON_BIN"
echo "Config: $CONFIG_PATH"
echo "=========================================="

run_trial() {
  local gpu="$1"
  local label="$2"
  local method="$3"
  local output_file="$4"
  shift 4

  echo "[$label] GPU $gpu | method=$method"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" src/experiment_runner.py \
    "${COMMON_ARGS[@]}" \
    --method "$method" \
    --gpu "$gpu" \
    --output "$output_file" \
    "$@"
}

(
  run_trial "$GPU_A" "1/7" "LLM+Episodic" "$OUTPUT_DIR/llm_episodic.json"
  run_trial "$GPU_A" "4/7" "TPE" "$OUTPUT_DIR/tpe.json"
  run_trial "$GPU_A" "7/7" "Random" "$OUTPUT_DIR/random.json"
) &
PID_A=$!

(
  run_trial "$GPU_B" "2/7" "LLM_only" "$OUTPUT_DIR/llm_only.json" --use-episodic false --use-meta false
  run_trial "$GPU_B" "5/7" "TPE+MV" "$OUTPUT_DIR/tpe_mv.json"
) &
PID_B=$!

(
  run_trial "$GPU_C" "3/7" "Bayesian" "$OUTPUT_DIR/bayesian.json"
  run_trial "$GPU_C" "6/7" "TPE+Pruner" "$OUTPUT_DIR/tpe_pruner.json"
) &
PID_C=$!

wait "$PID_A" "$PID_B" "$PID_C"

echo "Done: $OUTPUT_DIR"
