from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED = [
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "Dockerfile",
    ".dockerignore",
    "docker-compose.yml",
    "requirements.txt",
    ".env.example",
    "environment/autodl_environment.yml",
    "environment/autodl_pip_freeze.txt",
    "environment/runtime_report.json",
    "configs/paper/exp1_baseline_comparison.json",
    "configs/paper/exp2_ablation_study.json",
    "configs/paper/exp3_ancillary_stress_test.json",
    "results/paper/exp1_baseline/aggregate_metrics.csv",
    "results/paper/exp1_baseline/convergence_by_seed.csv",
    "results/paper/exp2_ablation/aggregate_metrics.csv",
    "results/paper/exp2_ablation/convergence_by_seed.csv",
    "results/paper/exp3_ancillary_stress_test/aggregate_metrics.csv",
    "results/paper/exp3_ancillary_stress_test/convergence_by_seed.csv",
    "figures/fig1_method_overview.png",
    "figures/fig2_exp1_bsf_trajectory.pdf",
    "figures/fig3_exp2_bsf_trajectory.pdf",
    "figures/fig4_exp2_component_effect.pdf",
    "docs/docker.md",
]

FORBIDDEN_SUFFIXES = {".pth", ".jsonl", ".pyc"}
FORBIDDEN_PATH_PARTS = {"wandb", "__pycache__", "third_party", "models", "legacy", "pilot", "smoke"}
SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"OPENAI_API_KEY\s*=\s*(?!replace_me|$)[^\s]+"),
    re.compile(r"C:\\Users\\", re.IGNORECASE),
    re.compile(r"/home/" + r"room" + r"rest/", re.IGNORECASE),
    re.compile(r"jw" + r"seok", re.IGNORECASE),
]
CSV_COLUMNS = {
    "aggregate_metrics.csv": {"experiment", "setting", "dataset", "method", "best_acc", "mean_acc", "std", "best_at_10", "final_bsf", "auc_bsf", "n_trials", "n_seeds"},
    "convergence_by_seed.csv": {"experiment", "dataset", "method", "seed", "trial", "accuracy", "best_so_far", "elapsed_sec"},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fail(message: str, errors: list[str]) -> None:
    errors.append(message)


def check_required(errors: list[str]) -> None:
    for rel in REQUIRED:
        if not (ROOT / rel).exists():
            fail(f"missing required file: {rel}", errors)


def check_forbidden_paths(errors: list[str]) -> None:
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        parts = {p.lower() for p in rel.parts}
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            fail(f"forbidden file suffix: {rel}", errors)
        if parts.intersection(FORBIDDEN_PATH_PARTS):
            fail(f"forbidden path component: {rel}", errors)
        if "adult_income" in str(rel).lower():
            fail(f"adult_income path is outside paper artifact scope: {rel}", errors)


def check_text_secrets(errors: list[str]) -> None:
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() in {".pdf", ".png"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                fail(f"possible secret or local path in {path.relative_to(ROOT)}", errors)


def check_json_and_csv(errors: list[str]) -> None:
    for path in (ROOT / "configs").rglob("*.json"):
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            fail(f"invalid JSON {path.relative_to(ROOT)}: {exc}", errors)
    for path in (ROOT / "results" / "paper").rglob("*.csv"):
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = set(next(reader, []))
        required = CSV_COLUMNS.get(path.name)
        if required and not required.issubset(header):
            fail(f"CSV schema mismatch {path.relative_to(ROOT)} missing {sorted(required - header)}", errors)


def check_manifests(errors: list[str]) -> None:
    for manifest_path in (ROOT / "results" / "paper").rglob("manifest.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in manifest.get("files", []):
            file_path = manifest_path.parent / item["path"]
            if not file_path.exists():
                fail(f"manifest references missing file: {file_path.relative_to(ROOT)}", errors)
                continue
            if item.get("sha256") != sha256(file_path):
                fail(f"sha256 mismatch: {file_path.relative_to(ROOT)}", errors)


def main() -> int:
    errors: list[str] = []
    check_required(errors)
    check_forbidden_paths(errors)
    check_text_secrets(errors)
    check_json_and_csv(errors)
    check_manifests(errors)
    if errors:
        print("Release artifact validation failed:")
        for err in errors:
            print(f"- {err}")
        return 1
    print("Release artifact validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
