"""Custom dataset HPO entry point.

This runner validates custom data/configuration and supports dry-run candidate
generation. Users can attach their own training objective by replacing
``evaluate_candidate`` with a project-specific function.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

try:
    from .custom_data import load_custom_dataset
    from .result_io import ensure_dir, write_json, write_rows
except ImportError:  # direct script execution fallback
    from custom_data import load_custom_dataset
    from result_io import ensure_dir, write_json, write_rows


def sample_candidate(search_space: dict[str, dict[str, Any]], rng: random.Random) -> dict[str, Any]:
    candidate = {}
    for name, spec in search_space.items():
        kind = spec["type"]
        if kind == "categorical":
            candidate[name] = rng.choice(spec["choices"])
        elif kind == "int":
            low, high = spec["bounds"]
            candidate[name] = rng.randint(int(low), int(high))
        elif kind == "float":
            low, high = float(spec["bounds"][0]), float(spec["bounds"][1])
            if spec.get("scale") == "log":
                import math

                candidate[name] = 10 ** rng.uniform(math.log10(low), math.log10(high))
            else:
                candidate[name] = rng.uniform(low, high)
        else:
            raise ValueError(f"Unsupported search-space type for {name}: {kind}")
    return candidate


def validate_search_space(search_space: dict[str, dict[str, Any]]) -> None:
    if not search_space:
        raise ValueError("search_space must be non-empty")
    for name, spec in search_space.items():
        if spec.get("type") in {"float", "int"} and len(spec.get("bounds", [])) != 2:
            raise ValueError(f"{name} must define two bounds")
        if spec.get("type") == "categorical" and not spec.get("choices"):
            raise ValueError(f"{name} must define choices")


def evaluate_candidate(candidate: dict[str, Any], loaded_dataset) -> dict[str, Any]:
    """Placeholder objective for release smoke tests.

    Full model training is intentionally project-specific for custom datasets.
    This deterministic proxy confirms the HPO plumbing and output schema.
    """

    score = 0.5 + (hash(json.dumps(candidate, sort_keys=True)) % 1000) / 10000.0
    return {"accuracy": min(score, 0.999), "status": "dry_run_proxy", "metadata": loaded_dataset.metadata}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", default="Random", choices=["Random", "TPE", "LLM+Episodic"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default="results/custom_hpo/run.json")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)
    validate_search_space(config["search_space"])
    loaded = load_custom_dataset(config)

    rng = random.Random(int(config.get("seed", 42)))
    n_trials = 1 if args.dry_run else int(config.get("n_trials", 20))
    rows = []
    best = None
    for trial in range(1, n_trials + 1):
        candidate = sample_candidate(config["search_space"], rng)
        result = evaluate_candidate(candidate, loaded)
        score = float(result["accuracy"])
        best = score if best is None else max(best, score)
        rows.append({"trial": trial, "method": args.method, "accuracy": score, "best_so_far": best, "status": result["status"], **candidate})

    output = Path(args.output)
    ensure_dir(output.parent)
    write_json(output, {"config": args.config, "method": args.method, "dataset_metadata": loaded.metadata, "trials": rows})
    write_rows(output.with_suffix(".csv"), list(rows[0].keys()), rows)
    print(f"Wrote {output} and {output.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
