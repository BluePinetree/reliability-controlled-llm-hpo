"""Offline, non-manuscript proposal/validation demonstration; never trains or calls an API."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

# Keep the release free of bytecode even when invoked without python -B.
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from src.calibration import UncertaintyCalibrator
from src.custom_runner import sample_candidate, validate_search_space
from src.prompt_builder import PromptBuilder


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def exact_keys(value, keys):
    return isinstance(value, dict) and set(value) == set(keys)


def validate_config(config):
    if not exact_keys(config, ("schema_version", "mode", "seed", "candidate_count", "objective", "search_space")):
        raise ValueError("Unexpected or missing config fields")
    if config["schema_version"] != "claude-science-offline-v1" or config["mode"] != "offline":
        raise ValueError("Only the offline v1 configuration is supported")
    if type(config["seed"]) is not int or not 0 <= config["seed"] <= 2**32 - 1:
        raise ValueError("seed must be an unsigned 32-bit integer")
    if type(config["candidate_count"]) is not int or not 1 <= config["candidate_count"] <= 3:
        raise ValueError("candidate_count must be between one and three")
    if config["objective"] != {"metric": "validation_accuracy", "maximize": True}:
        raise ValueError("This demonstration uses the fixed validation_accuracy objective")
    space = config["search_space"]
    if not exact_keys(space, ("learning_rate", "batch_size", "optimizer")):
        raise ValueError("Use the three declared demonstration parameters")
    if not all(isinstance(spec, dict) for spec in space.values()):
        raise ValueError("Search-space specifications must be objects")
    validate_search_space(space)
    for name, spec in space.items():
        kind = spec.get("type")
        if name == "optimizer":
            if not exact_keys(spec, ("type", "choices")) or kind != "categorical" or spec["choices"] != ["Adam", "SGD"]:
                raise ValueError("optimizer choices must be Adam and SGD")
            continue
        expected = ("type", "bounds", "scale") if name == "learning_rate" else ("type", "bounds")
        if not exact_keys(spec, expected) or kind != ("float" if name == "learning_rate" else "int"):
            raise ValueError("Invalid numeric search-space specification")
        bounds = spec["bounds"]
        if not isinstance(bounds, list) or len(bounds) != 2 or not all(number(v) for v in bounds):
            raise ValueError("Bounds must be two finite numbers")
        low, high = bounds
        if not 0 < low < high:
            raise ValueError("Bounds must be positive and strictly increasing")
        if name == "learning_rate" and (spec["scale"] != "log" or high > 0.001):
            raise ValueError("Use a log learning-rate range no higher than 0.001")
        if name == "batch_size" and (any(type(v) is not int for v in bounds) or low < 16 or high > 64):
            raise ValueError("Batch bounds must be integers within 16 through 64")


def validate_candidate(candidate, space):
    """Fail closed: no clipping, coercion, unknown fields, or nonfinite numbers."""
    if not exact_keys(candidate, ("params", "mu", "sigma", "reason")):
        return False, "candidate_fields"
    params = candidate["params"]
    if not exact_keys(params, space):
        return False, "parameter_fields"
    if not number(candidate["mu"]) or not 0 <= candidate["mu"] <= 1:
        return False, "invalid_mu"
    if not number(candidate["sigma"]) or not 0 < candidate["sigma"] <= 1:
        return False, "invalid_sigma"
    if not isinstance(candidate["reason"], str) or len(candidate["reason"]) > 200:
        return False, "invalid_reason"
    for name, spec in space.items():
        value = params[name]
        if spec["type"] == "categorical":
            if type(value) is not str or value not in spec["choices"]:
                return False, "invalid_category"
        else:
            if not number(value) or (spec["type"] == "int" and type(value) is not int):
                return False, "invalid_numeric_type"
            if not spec["bounds"][0] <= value <= spec["bounds"][1]:
                return False, "out_of_bounds"
    return True, "pass"


def build_report(config):
    validate_config(config)
    space = config["search_space"]
    rng = random.Random(config["seed"])
    candidates = [
        {"params": sample_candidate(space, rng), "mu": 0.5, "sigma": 0.1,
         "reason": "Synthetic offline placeholder; not a model prediction."}
        for _ in range(config["candidate_count"])
    ]
    # Deliberate invalid fixture demonstrates rejection, not a provider failure.
    invalid = {**candidates[0], "params": {**candidates[0]["params"], "learning_rate": 1.0}}
    candidates.append(invalid)
    rows, seen = [], set()
    for index, candidate in enumerate(candidates, 1):
        valid, reason = validate_candidate(candidate, space)
        if valid:
            fingerprint = canonical(candidate["params"])
            if fingerprint in seen:
                valid, reason = False, "duplicate"
            seen.add(fingerprint)
        # Rejected payloads and free-text rationales are never persisted.
        rows.append({"candidate_id": index, "accepted": valid, "validation_reason": reason,
                     "candidate": {key: candidate[key] for key in ("params", "mu", "sigma")} if valid else None})

    # No invented measured history is used to open the gate.
    gate_pass, gate_reason, cal_report = UncertaintyCalibrator().gate()
    prompt = PromptBuilder().build([], space, config["objective"], True)
    prompt += "\nOffline teaching fixture. mu and sigma are suggestions, never measured evidence."
    return {
        "schema_version": config["schema_version"],
        "scope": "non-manuscript",
        "demo_type": "proposal-only demo",
        "proposal_source": "seeded_random_synthetic_fixture",
        "config_sha256": digest(config),
        "prompt_schema_version": "PromptBuilder-with-offline-note-v1",
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "candidate_schema_basis": "docs/llm_audit_and_prompt_schema.md v1; strict local validation",
        "seed": config["seed"],
        "claude_api_calls": 0,
        "model_identifier": None,
        "training_executed": False,
        "measured_metric": None,
        "predictions_are_synthetic": True,
        "validation": rows,
        "calibration_gate": {"passed": gate_pass, "reason": gate_reason, "record_count": cal_report.n},
        "selection": {"candidate_id": None, "reason": "deferred_pending_measured_history"},
        "risk_scoring_executed": False,
        "evidence_verdict": "schema_and_closed_gate_verified_only",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Compare to the checked-in fixture without writing")
    args = parser.parse_args()
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    report = build_report(config)
    if args.check:
        expected = json.loads((HERE / "expected_output.json").read_text(encoding="utf-8"))
        if canonical(report) != canonical(expected):
            raise SystemExit("Fixture mismatch: inspect the config/code change before updating the fixture")
        print("Offline fixture check passed; no API calls or training.")
    else:
        output = HERE / "output" / "report.json"
        output.parent.mkdir(exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
        print("Wrote examples/claude_for_science/output/report.json; no API calls or training.")


if __name__ == "__main__":
    main()
