# Installation

The paper reproduction environment is the AutoDL virtual environment snapshot:

```bash
micromamba create -n AutoDL -f environment/autodl_environment.yml
micromamba activate AutoDL
python scripts/capture_runtime_report.py --out environment/runtime_report.json
```

For a quick local setup:

```bash
pip install -r requirements.txt
cp .env.example .env
```

Set `OPENAI_API_KEY` only when running LLM proposal methods. Baselines and dry
runs do not require an API key.

Docker users can build the CUDA 11.8 image with:

```bash
docker build -t reliability-controlled-llm-hpo:cu118 .
```

See `docs/docker.md` for full Docker instructions.
