#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" -m src.custom_runner "$@"
