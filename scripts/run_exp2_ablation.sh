#!/usr/bin/env bash
set -euo pipefail

# Experiment 2: CCRC Ablation Study
# Core comparisons:
#   Legacy-Full, CCRC-Full, CCRC-EpisodicOff, CCRC-NoCoupledRisk
# Appendix-style auxiliary runs:
#   CCRC-NoRerank, CCRC+Meta, CCRC+FailSafe
# Usage:
#   bash scripts/run_exp2_ablation.sh --gpus 0,1,2 --python python
#   bash scripts/run_exp2_ablation.sh --gpus 0,1,2 --skip-appendix
#   bash scripts/run_exp2_ablation.sh --gpus 0,1,2 --appendix-only

GPUS_CSV="0,1,2"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="configs/paper/exp2_ablation_study.json"
OUTPUT_DIR="results/runs/exp2_ablation"
RUN_CORE="true"
RUN_APPENDIX="true"
RESUME="false"
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
    --skip-appendix|--skip-no-failsafe)
      RUN_APPENDIX="false"
      shift
      ;;
    --resume)
      RESUME="true"
      shift
      ;;
    --force)
      RESUME="false"
      shift
      ;;
    --appendix-only)
      RUN_CORE="false"
      RUN_APPENDIX="true"
      shift
      ;;
    --include-appendix|--include-no-failsafe)
      RUN_APPENDIX="true"
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

BASE_CONFIG="$OUTPUT_DIR/_tmp_exp2_ccrc_base.json"
LEGACY_CONFIG="$OUTPUT_DIR/_tmp_exp2_legacy_full.json"
NO_COUPLED_RISK_CONFIG="$OUTPUT_DIR/_tmp_exp2_no_coupled_risk.json"
NO_RERANK_CONFIG="$OUTPUT_DIR/_tmp_exp2_no_rerank.json"
WITH_META_CONFIG="$OUTPUT_DIR/_tmp_exp2_with_meta.json"
WITH_FAILSAFE_CONFIG="$OUTPUT_DIR/_tmp_exp2_with_failsafe.json"

"$PYTHON_BIN" - <<PY
import json
from pathlib import Path

src = Path(r"$CONFIG_PATH")
base = Path(r"$BASE_CONFIG")
legacy = Path(r"$LEGACY_CONFIG")
no_coupled = Path(r"$NO_COUPLED_RISK_CONFIG")
no_rerank = Path(r"$NO_RERANK_CONFIG")
with_meta = Path(r"$WITH_META_CONFIG")
with_failsafe = Path(r"$WITH_FAILSAFE_CONFIG")

with open(src, "r", encoding="utf-8-sig") as f:
    cfg = json.load(f)

base_cfg = dict(cfg)
with open(base, "w", encoding="utf-8") as f:
    json.dump(base_cfg, f, indent=2, ensure_ascii=False)

legacy_cfg = dict(base_cfg)
legacy_cfg.update({
    "reliability_control_mode": "legacy",
    "episodic_gate_mode": "binary",
    "enable_near_optimal_rerank": False,
    "risk_sigma_penalty_mode": "legacy",
    "gate_coupled_soft_risk": False,
    "soft_risk_weight": legacy_cfg.get("param_risk_penalty_weight", 0.05),
})
with open(legacy, "w", encoding="utf-8") as f:
    json.dump(legacy_cfg, f, indent=2, ensure_ascii=False)

no_coupled_cfg = dict(base_cfg)
no_coupled_cfg["gate_coupled_soft_risk"] = False
with open(no_coupled, "w", encoding="utf-8") as f:
    json.dump(no_coupled_cfg, f, indent=2, ensure_ascii=False)

no_rerank_cfg = dict(base_cfg)
no_rerank_cfg["enable_near_optimal_rerank"] = False
with open(no_rerank, "w", encoding="utf-8") as f:
    json.dump(no_rerank_cfg, f, indent=2, ensure_ascii=False)

with_meta_cfg = dict(base_cfg)
with_meta_cfg["use_meta_learning"] = True
with open(with_meta, "w", encoding="utf-8") as f:
    json.dump(with_meta_cfg, f, indent=2, ensure_ascii=False)

with_failsafe_cfg = dict(base_cfg)
with_failsafe_cfg["enable_fail_safe"] = True
with open(with_failsafe, "w", encoding="utf-8") as f:
    json.dump(with_failsafe_cfg, f, indent=2, ensure_ascii=False)
PY

trap 'rm -f "$BASE_CONFIG" "$LEGACY_CONFIG" "$NO_COUPLED_RISK_CONFIG" "$NO_RERANK_CONFIG" "$WITH_META_CONFIG" "$WITH_FAILSAFE_CONFIG"' EXIT

COMMON_ARGS=()
if [[ -n "$N_TRIALS_OVERRIDE" ]]; then
  COMMON_ARGS+=(--n-trials "$N_TRIALS_OVERRIDE")
fi
if [[ -n "$N_SEEDS_OVERRIDE" ]]; then
  COMMON_ARGS+=(--n-seeds "$N_SEEDS_OVERRIDE")
fi

echo "=========================================="
echo "Experiment 2: CCRC Ablation"
echo "GPUs: $GPUS_CSV | Python: $PYTHON_BIN"
echo "Base config: $CONFIG_PATH"
echo "Run core: $RUN_CORE | Run appendix: $RUN_APPENDIX"
echo "Resume mode: $RESUME"
echo "Core: Legacy-Full, CCRC-Full, CCRC-EpisodicOff, CCRC-NoCoupledRisk"
echo "Appendix: CCRC-NoRerank, CCRC+Meta, CCRC+FailSafe"
echo "=========================================="

should_skip_output() {
  local output_file="$1"
  if [[ "$RESUME" != "true" ]]; then
    return 1
  fi
  if [[ -s "$output_file" ]]; then
    return 0
  fi
  return 1
}

run_trial() {
  local gpu="$1"
  local label="$2"
  local config_path="$3"
  local method="$4"
  local output_file="$5"
  shift 5

  if should_skip_output "$output_file"; then
    echo "[$label] Skip existing output: $output_file"
    return 0
  fi

  echo "[$label] GPU $gpu | method=$method"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" src/experiment_runner.py \
    --config "$config_path" \
    --method "$method" \
    --gpu "$gpu" \
    --output "$output_file" \
    "${COMMON_ARGS[@]}" \
    "$@"
}

if [[ "$RUN_CORE" == "true" ]]; then
  (
    run_trial "$GPU_A" "Legacy-Full" "$LEGACY_CONFIG" "LLM+Episodic" "$OUTPUT_DIR/legacy_full.json" \
      --use-episodic true --use-meta false --use-variance false --confidence-gated-episodic true --risk-aware-selection true

    run_trial "$GPU_A" "CCRC-NoCoupledRisk" "$NO_COUPLED_RISK_CONFIG" "LLM+Episodic" "$OUTPUT_DIR/ccrc_no_coupled_risk.json" \
      --use-episodic true --use-meta false --use-variance false --confidence-gated-episodic true --risk-aware-selection true
  ) &
  PID_A=$!

  (
    run_trial "$GPU_B" "CCRC-Full" "$BASE_CONFIG" "LLM+Episodic" "$OUTPUT_DIR/ccrc_full.json" \
      --use-episodic true --use-meta false --use-variance false --confidence-gated-episodic true --risk-aware-selection true

    run_trial "$GPU_B" "CCRC-EpisodicOff" "$BASE_CONFIG" "LLM_only" "$OUTPUT_DIR/ccrc_no_episodic.json" \
      --use-episodic false --use-meta false --use-variance false --confidence-gated-episodic false --risk-aware-selection true
  ) &
  PID_B=$!
fi

if [[ "$RUN_APPENDIX" == "true" ]]; then
  (
    run_trial "$GPU_C" "CCRC-NoRerank" "$NO_RERANK_CONFIG" "LLM+Episodic" "$OUTPUT_DIR/ccrc_no_rerank.json" \
      --use-episodic true --use-meta false --use-variance false --confidence-gated-episodic true --risk-aware-selection true

    run_trial "$GPU_C" "CCRC+Meta" "$WITH_META_CONFIG" "LLM+Episodic" "$OUTPUT_DIR/ccrc_with_meta.json" \
      --use-episodic true --use-meta true --use-variance false --confidence-gated-episodic true --risk-aware-selection true

    run_trial "$GPU_C" "CCRC+FailSafe" "$WITH_FAILSAFE_CONFIG" "LLM+Episodic" "$OUTPUT_DIR/ccrc_with_failsafe.json" \
      --use-episodic true --use-meta false --use-variance false --confidence-gated-episodic true --risk-aware-selection true
  ) &
  PID_C=$!
fi

if [[ "$RUN_CORE" == "true" && "$RUN_APPENDIX" == "true" ]]; then
  wait "$PID_A" "$PID_B" "$PID_C"
elif [[ "$RUN_CORE" == "true" ]]; then
  wait "$PID_A" "$PID_B"
elif [[ "$RUN_APPENDIX" == "true" ]]; then
  wait "$PID_C"
else
  echo "Nothing to run."
  exit 1
fi

echo "Done: $OUTPUT_DIR"
