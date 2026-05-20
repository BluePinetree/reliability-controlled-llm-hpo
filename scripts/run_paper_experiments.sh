#!/usr/bin/env bash
set -euo pipefail

EXP="${1:-all}"
shift || true

case "$EXP" in
  exp1)
    bash scripts/run_exp1_baseline.sh "$@"
    ;;
  exp2)
    bash scripts/run_exp2_ablation.sh "$@"
    ;;
  exp3)
    bash scripts/run_exp3_ancillary_stress_test.sh "$@"
    ;;
  all)
    bash scripts/run_exp1_baseline.sh "$@"
    bash scripts/run_exp2_ablation.sh "$@"
    bash scripts/run_exp3_ancillary_stress_test.sh "$@"
    ;;
  *)
    echo "Usage: bash scripts/run_paper_experiments.sh [exp1|exp2|exp3|all] [script args...]"
    exit 1
    ;;
esac
