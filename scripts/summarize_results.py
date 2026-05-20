from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from result_io import ensure_dir, sha256_file, write_json  # noqa: E402


EXP1_METHODS = {
    "llm_episodic": "LLM+Episodic",
    "llm_only": "LLM-only",
    "bayesian": "Bayesian",
    "tpe": "TPE",
    "tpe_mv": "TPE+MV",
    "tpe_pruner": "TPE+Pruner",
    "random": "Random",
}

EXP2_METHODS = {
    "legacy_full": "Legacy-Full",
    "ccrc_full": "CCRC-Full",
    "ccrc_no_episodic": "CCRC-NoEpisodic",
    "ccrc_no_coupled_risk": "CCRC-NoCoupledRisk",
}

EXP3_METHODS = {
    "llm_episodic": "LLM+Episodic",
    "llm_only": "LLM-only",
    "tpe_mv": "TPE+MV",
}

DATASETS = {"cifar100": "CIFAR-100", "ag_news": "AG News", "imdb": "IMDb"}


def _safe_read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _aggregate_from_seed(experiment: str, dataset: str, method: str, seed_df: pd.DataFrame, raw_df: pd.DataFrame | None) -> dict:
    ok = seed_df[seed_df.get("status", "success").eq("success")] if "status" in seed_df.columns else seed_df
    n_seeds = int(ok["seed"].nunique()) if "seed" in ok.columns else 0
    n_trials = int(ok["trial"].max()) if "trial" in ok.columns and not ok.empty else 0
    final_by_seed = ok.sort_values("trial").groupby("seed")["best_so_far"].last() if "seed" in ok.columns else pd.Series(dtype=float)
    at10 = ok[ok["trial"].le(10)].sort_values("trial").groupby("seed")["best_so_far"].last() if "seed" in ok.columns else pd.Series(dtype=float)
    auc_by_seed = ok.groupby("seed")["best_so_far"].mean() if "seed" in ok.columns else pd.Series(dtype=float)
    acc_source = raw_df[raw_df["status"].eq("success")] if raw_df is not None and "status" in raw_df.columns else ok
    return {
        "experiment": experiment,
        "setting": "paper",
        "dataset": dataset,
        "method": method,
        "best_acc": float(acc_source["accuracy"].max()) if "accuracy" in acc_source and not acc_source.empty else "",
        "mean_acc": float(acc_source["accuracy"].mean()) if "accuracy" in acc_source and not acc_source.empty else "",
        "std": float(acc_source["accuracy"].std(ddof=0)) if "accuracy" in acc_source and len(acc_source) > 1 else 0.0,
        "best_at_10": float(at10.mean()) if not at10.empty else "",
        "final_bsf": float(final_by_seed.mean()) if not final_by_seed.empty else "",
        "auc_bsf": float(auc_by_seed.mean()) if not auc_by_seed.empty else "",
        "n_trials": n_trials,
        "n_seeds": n_seeds,
    }


def _normalize_seed_rows(experiment: str, dataset: str, method: str, seed_df: pd.DataFrame) -> pd.DataFrame:
    out = seed_df.copy()
    out.insert(0, "experiment", experiment)
    out.insert(1, "dataset", dataset)
    out.insert(2, "method", method)
    if "elapsed_sec" not in out.columns:
        out["elapsed_sec"] = ""
    return out[["experiment", "dataset", "method", "seed", "trial", "accuracy", "best_so_far", "elapsed_sec"]]


def summarize_flat(source_dir: Path, output_dir: Path, experiment: str, dataset: str, methods: dict[str, str], paper_items: list[str], config: str) -> None:
    aggregate_rows = []
    seed_rows = []
    for key, label in methods.items():
        seed_path = source_dir / f"{key}_convergence_by_seed.csv"
        raw_path = source_dir / f"{key}.csv"
        seed_df = _safe_read(seed_path)
        raw_df = _safe_read(raw_path) if raw_path.exists() else None
        aggregate_rows.append(_aggregate_from_seed(experiment, dataset, label, seed_df, raw_df))
        seed_rows.append(_normalize_seed_rows(experiment, dataset, label, seed_df))
    _write_release_outputs(output_dir, aggregate_rows, seed_rows, experiment, paper_items, config, str(source_dir))


def summarize_exp3(source_dir: Path, output_dir: Path) -> None:
    aggregate_rows = []
    seed_rows = []
    for method_key, method_label in EXP3_METHODS.items():
        for dataset_key, dataset_label in DATASETS.items():
            seed_path = source_dir / method_key / f"{dataset_key}_convergence_by_seed.csv"
            raw_path = source_dir / method_key / f"{dataset_key}.csv"
            if not seed_path.exists():
                continue
            seed_df = _safe_read(seed_path)
            raw_df = _safe_read(raw_path) if raw_path.exists() else None
            aggregate_rows.append(_aggregate_from_seed("exp3_ancillary_stress_test", dataset_label, method_label, seed_df, raw_df))
            seed_rows.append(_normalize_seed_rows("exp3_ancillary_stress_test", dataset_label, method_label, seed_df))
    _write_release_outputs(
        output_dir,
        aggregate_rows,
        seed_rows,
        "exp3_ancillary_stress_test",
        ["Table 5"],
        "configs/paper/exp3_ancillary_stress_test.json",
        str(source_dir),
    )


def _write_release_outputs(output_dir: Path, aggregate_rows: list[dict], seed_rows: list[pd.DataFrame], experiment: str, paper_items: list[str], config: str, source_dir: str) -> None:
    ensure_dir(output_dir)
    aggregate_path = output_dir / "aggregate_metrics.csv"
    seed_path = output_dir / "convergence_by_seed.csv"
    pd.DataFrame(aggregate_rows).to_csv(aggregate_path, index=False)
    pd.concat(seed_rows, ignore_index=True).to_csv(seed_path, index=False)
    manifest = {
        "experiment": experiment,
        "paper_items": paper_items,
        "config": config,
        "source_experiment_dir": source_dir.replace("\\", "/"),
        "generated_by_script": "scripts/summarize_results.py",
        "created_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "included_in_paper": True,
        "files": [
            {"path": "aggregate_metrics.csv", "sha256": sha256_file(aggregate_path), "paper_items": paper_items},
            {"path": "convergence_by_seed.csv", "sha256": sha256_file(seed_path), "paper_items": paper_items + ["Appendix C"]},
        ],
        "notes": "Minimal aggregate and seed-level result files. Raw per-trial dumps, raw LLM logs, checkpoints, calibration state, and episodic memory dumps are intentionally excluded.",
    }
    write_json(output_dir / "manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp1-dir", type=Path, default=Path("results/exp1_baseline_retry1"))
    parser.add_argument("--exp2-dir", type=Path, default=Path("results/exp2_ablation"))
    parser.add_argument("--exp3-dir", type=Path, default=Path("results/exp3_multi_domain"))
    parser.add_argument("--output-root", type=Path, default=ROOT / "results" / "paper")
    args = parser.parse_args()

    summarize_flat(args.exp1_dir, args.output_root / "exp1_baseline", "exp1_baseline", "CIFAR-100", EXP1_METHODS, ["Table 3", "Figure 2"], "configs/paper/exp1_baseline_comparison.json")
    summarize_flat(args.exp2_dir, args.output_root / "exp2_ablation", "exp2_ablation", "CIFAR-100", EXP2_METHODS, ["Table 4", "Figure 3", "Figure 4"], "configs/paper/exp2_ablation_study.json")
    summarize_exp3(args.exp3_dir, args.output_root / "exp3_ancillary_stress_test")


if __name__ == "__main__":
    main()
