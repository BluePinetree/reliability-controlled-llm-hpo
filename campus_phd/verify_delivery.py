"""Check local task preservation, new JSON/links, word counts, and secret patterns."""

import hashlib
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]


def main():
    errors = []
    baseline = json.loads((ROOT / "campus_phd/BASELINE_SHA256.json").read_text(encoding="utf-8"))
    for name, record in baseline["files"].items():
        path = ROOT / name
        if not path.is_file():
            errors.append("Missing baseline file: " + name)
            continue
        content = path.read_bytes()
        if name == "README.md":
            content = content[:record["size_bytes"]]
        if hashlib.sha256(content).hexdigest() != record["sha256"]:
            errors.append("Changed baseline content: " + name)

    scoped = [ROOT / "README.md", ROOT / "docs/claude_for_science_workflow.md"]
    for folder in ("campus_phd", "examples/claude_for_science"):
        scoped.extend(path for path in (ROOT / folder).rglob("*") if path.is_file())
    patterns = [re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
                re.compile(r"(?:OPENAI|ANTHROPIC|CLAUDE)_API_KEY\s*=\s*(?!replace_me(?:\s|$))[^\s]+"),
                re.compile(r"[A-Za-z]:[\\/]Users[\\/]", re.I),
                re.compile(r"/(?:home|Users)/[A-Za-z0-9_.-]+/")]
    for path in scoped:
        rel = path.relative_to(ROOT).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            errors.append("Non-UTF8 file: " + rel)
            continue
        if any(pattern.search(text) for pattern in patterns):
            errors.append("Possible secret or personal absolute path: " + rel)
        if path.suffix == ".json":
            try:
                def reject_constant(value):
                    raise ValueError("Nonfinite JSON number")
                json.loads(text, parse_constant=reject_constant)
            except (ValueError, TypeError):
                errors.append("Invalid JSON: " + rel)
        if path.suffix == ".md":
            link_text = text
            if rel == "README.md":
                link_text = path.read_bytes()[baseline["files"][rel]["size_bytes"]:].decode("utf-8")
            for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", link_text):
                if target.startswith(("https://", "http://", "#", "mailto:")):
                    continue
                target = target.split("#", 1)[0]
                if not (path.parent / target).exists():
                    errors.append("Missing link in " + rel + ": " + target)
        # Existing files retain their original whitespace; check new task files.
        if rel not in baseline["files"]:
            if any(line.rstrip() != line for line in text.splitlines()):
                errors.append("Trailing whitespace: " + rel)
            if not text.endswith("\n"):
                errors.append("Missing final newline: " + rel)

    snippets = (ROOT / "campus_phd/APPLICATION_SNIPPETS.md").read_text(encoding="utf-8")
    sections = re.findall(r"^## ([^\n]+)\n\n(.*?)(?=\n## |\Z)", snippets, re.M | re.S)
    counts = [len(body.split()) for _, body in sections[:4]]
    if len(counts) != 4 or counts[:3] != [50, 100, 150] or not 80 <= counts[3] <= 120:
        errors.append("Application word counts: " + str(counts))
    if len(sections) != 5 or len(re.findall(r"^- ", sections[4][1], re.M)) != 5:
        errors.append("Expected five evidence-backed bullets")
    print("Baseline files checked:", len(baseline["files"]))
    print("Application word counts:", counts)
    if errors:
        for error in errors:
            print("FAIL:", error)
        return 1
    print("Delivery checks passed: preserved baseline (README prefix), JSON, links, whitespace, secret patterns.")
    print("Local baseline HEAD:", baseline.get("git_head", "unavailable"))
    print("Remote provenance is not independently verified; run the release validator separately.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
