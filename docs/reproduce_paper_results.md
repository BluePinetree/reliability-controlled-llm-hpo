# Reproducing Paper Results

## Hardware and Environment

Use the AutoDL environment snapshot in `environment/` as the reference package
stack. The original experiments used GPU execution; CPU smoke tests are only
for command and schema validation.

## Commands

```bash
bash scripts/run_paper_experiments.sh exp1 --gpus 0,1,2
bash scripts/run_paper_experiments.sh exp2 --gpus 0,1,2
bash scripts/run_paper_experiments.sh exp3 --gpus 0,1,2
```

Exp1 and Exp2 are the primary evidence. Exp3 is an ancillary failure-mode stress
test and should be interpreted as diagnostic.

## Figure and Table Artifacts

The released `results/paper/*` files are sufficient to validate paper table and
figure numbers:

- Figure 1: `figures/fig1_method_overview.png`
- Table 3 and Figure 2: `results/paper/exp1_baseline/`
- Table 4, Figure 3, and Figure 4: `results/paper/exp2_ablation/`
- Table 5: `results/paper/exp3_ancillary_stress_test/`
- Appendix C: `convergence_by_seed.csv`

Raw per-trial dumps and raw LLM logs are intentionally excluded.
