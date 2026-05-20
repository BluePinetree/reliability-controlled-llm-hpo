# Custom Dataset HPO Recipe

Supported examples:

- `custom_tabular`: CSV classification with a label column
- `custom_text`: text classification CSV with text and label columns
- `custom_image_folder`: torchvision `ImageFolder` layout

Run a dry-run smoke test:

```bash
python -m src.custom_runner --config examples/custom_data/tabular_csv/config.json --method Random --dry-run
```

For full custom experiments, replace `evaluate_candidate` in
`src/custom_runner.py` with a domain-specific training objective. Keep search
spaces conservative for NLP learning rates; the paper's Exp3 diagnostic result
shows that naive transfer of CV-scale learning-rate ranges can fail.
