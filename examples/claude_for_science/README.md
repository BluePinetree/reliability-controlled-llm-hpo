# Claude for Science: offline proposal-only demo

**Non-manuscript teaching artifact. No Claude call, model training, or measured
accuracy is produced.** This example demonstrates how a workflow designed to
integrate Claude keeps proposals separate from experimental evidence.

Run from the release artifact root, using Python 3.10+ with pandas and
scikit-learn available (both are existing release dependencies):

```bash
python -B examples/claude_for_science/run_demo.py
python -B examples/claude_for_science/run_demo.py --check
python -B examples/claude_for_science/test_demo.py
```

For an isolated CPU environment, the small dependency subset is sufficient:

```bash
python -m pip install pandas==2.3.3 scikit-learn==1.7.2
```

Installation may require network access; the demo itself does not. No dataset,
GPU, embedding model, API key, or provider SDK is needed. Dependencies are needed
because the reused custom runner imports its dataset loaders, even though this
example never invokes them. The tested local runtime is Python 3.11.4 with pandas
1.5.3 and scikit-learn 1.3.0; the pinned release versions above and other
platforms and Python versions have not been independently tested in this task.

Read [config.json](config.json), run the commands, then inspect
`examples/claude_for_science/output/report.json`. That local output directory is
ignored by Git. The checked-in [expected_output.json](expected_output.json) is a
synthetic fixture for comparison, not a historical Claude response.

In under five minutes, check these four outcomes:

1. Three seeded random candidates pass strict type, bounds, and category checks.
2. One deliberately invalid candidate is rejected with `out_of_bounds`; its
   payload is omitted from the report.
3. The existing calibration gate reports `insufficient_records` with zero
   measured records. No candidate is selected and risk scoring is not run.
4. `claude_api_calls` is zero, `training_executed` is false, and `measured_metric`
   is null. The fixed `mu=0.5` and `sigma=0.1` values are synthetic placeholders.

The example imports `sample_candidate` and `validate_search_space` from
`src/custom_runner.py`, `PromptBuilder` from `src/prompt_builder.py`, and the
gate from `src/calibration.py`. It uses the candidate fields documented in the
[existing audit schema](../../docs/llm_audit_and_prompt_schema.md). Strict
candidate validation is local to the example: it rejects malformed values and
duplicates rather than clipping or coercing them. The release client/runner
validators have different repair behavior and require provider or training
imports; they are not invoked here. No core code is changed.

The generated prompt stays in memory; its SHA-256 and local schema identifier
are recorded. No arbitrary provider input is accepted, and rejected payloads
and free-text rationales are not written. `--check` writes nothing and compares
the report against the fixture. The regular command replaces only the example's
own `output/report.json`. Config edits intentionally invalidate the fixture;
review the resulting behavior before updating it.

There is no live Claude mode in this release. A future integration must use a
separate adapter, explicit model selection, bounded calls, strict response
validation, and an allowlisted evidence report. A key alone does not enable it.

See the [research workflow](../../docs/claude_for_science_workflow.md),
[evidence ledger](../../campus_phd/CLAUDE_RESEARCH_EVIDENCE.md), and
[application wording](../../campus_phd/APPLICATION_SNIPPETS.md).
