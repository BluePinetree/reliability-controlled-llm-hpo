#!/usr/bin/env bash
set -euo pipefail

# Experiment 3: Ancillary Failure-Mode Stress Test
# Default methods: LLM+Episodic, LLM_only, TPE+MV
# Default domain->GPU mapping (3 GPUs):
#   GPU0: cifar100
#   GPU1: ag_news
#   GPU2: imdb
# Usage:
#   bash scripts/run_exp3_ancillary_stress_test.sh --gpus 0,1,2 --python python
#   bash scripts/run_exp3_ancillary_stress_test.sh --methods "LLM+Episodic,LLM_only,TPE+MV,Random"

GPUS_CSV="0,1,2"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="configs/paper/exp3_ancillary_stress_test.json"
OUTPUT_DIR="results/runs/exp3_ancillary_stress_test"
METHODS_CSV="LLM+Episodic,LLM_only,TPE+MV"
N_TRIALS_OVERRIDE=""
N_SEEDS_OVERRIDE=""
AUTO_INSTALL_ACCELERATE="false"

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
    --methods)
      METHODS_CSV="${2}"
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
    --auto-install-accelerate)
      AUTO_INSTALL_ACCELERATE="true"
      shift
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

ensure_python_module() {
  local module_name="$1"
  "$PYTHON_BIN" - <<PY >/dev/null 2>&1
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec("$module_name") else 1)
PY
}

IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
IFS=',' read -r -a METHODS <<< "$METHODS_CSV"
if [[ ${#GPUS[@]} -eq 0 ]]; then
  echo "No GPUs parsed from: $GPUS_CSV"
  exit 1
fi
if [[ ${#METHODS[@]} -eq 0 ]]; then
  echo "No methods parsed from: $METHODS_CSV"
  exit 1
fi
for i in "${!GPUS[@]}"; do
  GPUS[$i]="${GPUS[$i]// /}"
done
for i in "${!METHODS[@]}"; do
  METHODS[$i]="${METHODS[$i]#"${METHODS[$i]%%[![:space:]]*}"}"
  METHODS[$i]="${METHODS[$i]%"${METHODS[$i]##*[![:space:]]}"}"
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

NORMALIZED_CONFIG="$OUTPUT_DIR/_tmp_exp3_config_utf8.json"
"$PYTHON_BIN" - <<PY
import json

def infer_safe_lr(cfg):
    try:
        lr_high = float(cfg["search_space"]["learning_rate"]["bounds"][1])
        return min(0.01, lr_high)
    except Exception:
        return 0.01

src = r"$CONFIG_PATH"
dst = r"$NORMALIZED_CONFIG"
with open(src, "r", encoding="utf-8-sig") as f:
    cfg = json.load(f)

# Revised Exp3 profile. Keep proposal generation adaptive and guard NLP
# against the high-LR collapse region observed in preliminary runs.
cfg["llm_seed"] = None
cfg["use_meta_learning"] = False
cfg["use_variance_reduction"] = False
cfg["reset_calibration_state"] = True
cfg["reset_episodic_memory"] = True
cfg["enable_fail_safe"] = True
cfg["fail_safe_patience"] = 2
cfg["fail_safe_cooldown"] = 3
cfg["fail_safe_low_accuracy"] = 0.10
cfg["max_safe_learning_rate"] = infer_safe_lr(cfg)
cfg["conservative_lr_max"] = min(0.0075, cfg["max_safe_learning_rate"])
cfg["dataset_max_safe_learning_rates"] = {
    "ag_news": 2e-4,
    "imdb": 1.5e-4,
}
cfg["domain_max_safe_learning_rates"] = {
    "NLP": 2e-4,
}
cfg["dataset_risk_high_lr_thresholds"] = {
    "ag_news": 1.5e-4,
    "imdb": 1e-4,
}
cfg["domain_risk_high_lr_thresholds"] = {
    "NLP": 1.5e-4,
}
cfg["dataset_fail_safe_low_accuracy"] = {
    "ag_news": 0.55,
    "imdb": 0.55,
    "cifar100": 0.45,
}
cfg["domain_fail_safe_low_accuracy"] = {
    "NLP": 0.55,
    "CV": 0.45,
}
cfg["enable_confidence_gated_episodic"] = True
cfg["episodic_gate_min_episodes"] = 4
cfg["episodic_gate_min_calibration_records"] = 4
cfg["episodic_gate_min_coverage"] = 0.45
cfg["episodic_gate_max_coverage"] = 0.95
cfg["episodic_gate_max_calibration_error"] = 0.30
cfg["episodic_gate_max_nll"] = 2.5
cfg["enable_risk_aware_selection"] = True
cfg["risk_penalty_lambda"] = 0.50
cfg["param_risk_penalty_weight"] = 0.15

with open(dst, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2, ensure_ascii=False)
PY
CONFIG_PATH="$NORMALIZED_CONFIG"
trap 'rm -f "$NORMALIZED_CONFIG"' EXIT

if ! ensure_python_module "accelerate"; then
  if [[ "$AUTO_INSTALL_ACCELERATE" == "true" ]]; then
    echo "[Preflight] Installing accelerate>=0.26.0 ..."
    "$PYTHON_BIN" -m pip install "accelerate>=0.26.0"
  else
    echo "[Preflight] Missing module: accelerate (required for NLP domains in Exp3)."
    echo "Install with: $PYTHON_BIN -m pip install \"accelerate>=0.26.0\""
    echo "Or rerun with: --auto-install-accelerate"
    exit 1
  fi
fi

echo "=========================================="
echo "Experiment 3: Ancillary failure-mode stress test"
echo "GPUs: $GPUS_CSV | Python: $PYTHON_BIN"
echo "Config: $CONFIG_PATH"
echo "Methods: $METHODS_CSV"
echo "Revised core: adaptive loop, meta=off, variance=off, fail-safe=on, domain-aware LR caps"
echo "=========================================="

method_slug() {
  local method="$1"
  case "$method" in
    "LLM+Episodic") echo "llm_episodic" ;;
    "LLM_only") echo "llm_only" ;;
    "TPE+MV") echo "tpe_mv" ;;
    "TPE+Pruner") echo "tpe_pruner" ;;
    "TPE") echo "tpe" ;;
    "Bayesian") echo "bayesian" ;;
    "Random") echo "random" ;;
    *)
      echo "$method" | tr '[:upper:]' '[:lower:]' | tr ' +' '__'
      ;;
  esac
}

run_domain_method() {
  local gpu="$1"
  local dataset="$2"
  local domain="$3"
  local method="$4"

  local slug
  slug="$(method_slug "$method")"
  local out_dir="$OUTPUT_DIR/$slug"
  local out_file="$out_dir/${dataset}.json"

  mkdir -p "$out_dir"

  local extra_args=()
  if [[ "$method" == "LLM_only" ]]; then
    extra_args+=(--use-episodic false --use-meta false)
  fi
  if [[ -n "$N_TRIALS_OVERRIDE" ]]; then
    extra_args+=(--n-trials "$N_TRIALS_OVERRIDE")
  fi
  if [[ -n "$N_SEEDS_OVERRIDE" ]]; then
    extra_args+=(--n-seeds "$N_SEEDS_OVERRIDE")
  fi

  echo "[dataset=$dataset method=$method] GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" src/experiment_runner.py \
    --config "$CONFIG_PATH" \
    --dataset "$dataset" \
    --domain "$domain" \
    --method "$method" \
    --gpu "$gpu" \
    --output "$out_file" \
    "${extra_args[@]}"
}

run_dataset_group() {
  local gpu="$1"
  shift
  local pairs=("$@")

  for pair in "${pairs[@]}"; do
    local dataset="${pair%%:*}"
    local domain="${pair##*:}"

    for method in "${METHODS[@]}"; do
      run_domain_method "$gpu" "$dataset" "$domain" "$method"
    done
  done
}

# Preferred split for 3 GPUs. If fewer GPUs are provided, run sequentially.
if [[ ${#GPUS[@]} -ge 3 ]]; then
  (
    run_dataset_group "$GPU_A" "cifar100:CV"
  ) &
  PID_A=$!

  (
    run_dataset_group "$GPU_B" "ag_news:NLP"
  ) &
  PID_B=$!

  (
    run_dataset_group "$GPU_C" "imdb:NLP"
  ) &
  PID_C=$!

  wait "$PID_A" "$PID_B" "$PID_C"
else
  echo "Detected fewer than 3 GPUs in --gpus. Running sequential fallback."
  run_dataset_group "$GPU_A" "cifar100:CV" "ag_news:NLP" "imdb:NLP"
fi

echo "Done: $OUTPUT_DIR"
