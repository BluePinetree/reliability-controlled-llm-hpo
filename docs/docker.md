# Docker

The tested server target is:

- Ubuntu 22.04.5 LTS
- Linux kernel 5.15.0-168-generic
- x86_64
- CUDA toolkit 11.8 (`nvcc V11.8.89`)
- Python 3.10.19 in the AutoDL micromamba environment
- PyTorch wheel target: `torch==2.4.1+cu118`
- torchvision wheel target: `torchvision==0.19.1+cu118`
- GPU: 3 x NVIDIA RTX 6000 Ada Generation
- Environment manager used for the non-Docker run: micromamba

The Docker image uses:

```text
nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
```

## Server Checks

Run these on the experiment server before building:

```bash
cat /etc/os-release
uname -a
uname -m
nvidia-smi
nvidia-smi --query-gpu=name,driver_version,cuda_version,memory.total --format=csv
nvcc --version || true
which nvcc || true
docker --version
docker compose version || true
docker run --rm --gpus all nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 nvidia-smi
python --version
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("cudnn:", torch.backends.cudnn.version())
print("gpu count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"gpu[{i}]:", torch.cuda.get_device_name(i))
PY
```

## Build

```bash
docker build -t reliability-controlled-llm-hpo:cu118 .
```

## Validate

```bash
docker run --rm --gpus all reliability-controlled-llm-hpo:cu118 \
  python scripts/validate_release_artifact.py

docker run --rm --gpus all reliability-controlled-llm-hpo:cu118 \
  python scripts/capture_runtime_report.py --out environment/runtime_report.json
```

## Custom Dataset Dry Run

```bash
docker run --rm --gpus all reliability-controlled-llm-hpo:cu118 \
  python -m src.custom_runner \
  --config examples/custom_data/tabular_csv/config.json \
  --method Random \
  --dry-run \
  --output results/custom_hpo/dry_run.json
```

## Paper Experiments

Create `.env` from `.env.example` and set `OPENAI_API_KEY` before LLM-based
methods:

```bash
cp .env.example .env
docker compose run --rm llm-hpo bash scripts/run_paper_experiments.sh exp1 --gpus 0,1,2
docker compose run --rm llm-hpo bash scripts/run_paper_experiments.sh exp2 --gpus 0,1,2
docker compose run --rm llm-hpo bash scripts/run_paper_experiments.sh exp3 --gpus 0,1,2
```

`data/` and `results/runs/` are mounted from the host. Minimal paper result CSVs
under `results/paper/` are included in the artifact and are not overwritten by
the compose defaults.
