# Release Manifest

This artifact includes only paper-scope code, configs, figures, and minimal
aggregate result files. It excludes raw datasets, checkpoints, API keys, raw
LLM logs, per-trial dumps, calibration state dumps, and episodic memory dumps.

Docker support files are included:

- `Dockerfile`
- `.dockerignore`
- `docker-compose.yml`
- `docs/docker.md`

Each `results/paper/<experiment>/manifest.json` records:

- source experiment directory
- generation script
- linked paper table/figure items
- SHA-256 checksums for released CSV files

Before public upload, run:

```bash
python scripts/validate_release_artifact.py
```

This release folder is prepared for public GitHub release. Local home-directory
paths are replaced with redacted values, while citation and license metadata
retain the public author name.
