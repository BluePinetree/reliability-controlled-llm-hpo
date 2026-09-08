# Claude-Assisted, Evidence-Controlled Hyperparameter Research

This is a **non-manuscript** workflow and offline teaching demonstrator prepared
for researchers considering Claude as a research assistant. The delivered demo
uses deterministic synthetic proposals. No Claude API call or training run was
performed during this task. Historical manuscript results are not reattributed
to Claude.

## Research question

How can a researcher use semantic proposals in a low-budget mixed search space
while keeping experimental conclusions accountable to measurements? An LLM can
suggest interactions among categorical and numeric hyperparameters and critique
an experiment plan. Plausible reasoning can also obscure invalid configurations,
unjustified uncertainty estimates, or inappropriate transfer between domains.
The research workflow therefore places deterministic checks between suggestions
and execution, and measured evidence between execution and claims.

The [manuscript artifact](../README.md) treats Exp1/Exp2 on CIFAR-100 as its primary
empirical evidence. Exp3 is an ancillary failure-mode stress test, not evidence
of cross-domain superiority. This example adds no experimental result to those
claims.

## Existing system boundary

The public release's `src/unified_llm_client.py` is deliberately OpenAI-only,
using `OPENAI_API_KEY`. Its constructor defaults to `gpt-5.2`; paper configs also
name that identifier. These are source-code/config observations, not independent
verification of the model used in every historical run.

There are two relevant implementation paths:

| Path | Responsibilities and boundary |
|---|---|
| Paper `src/experiment_runner.py` | Builds a task/search-space/memory prompt; parses and repairs candidate parameters; calibrates predictions; applies acquisition and risk penalties; selects a candidate; evaluates it; updates history, calibration, and memory |
| `src/optimizer.py` | Uses `PromptBuilder`, `LLMClient`, the gate in `calibration.py`, optional GP reranking/fusion, and fallback search |
| `src/custom_runner.py` | Samples configurations and writes a proxy result; its method label does not implement a Claude proposal path or actual training |
| New offline example | Reuses the custom sampler, prompt builder, and calibration gate; performs local strict validation and stops before selection/evaluation |

`src/acquisition.py` supplies EI/UCB functions. `src/uncertainty_calibration.py`
supplies bias/scale calibration for the experiment runner. `src/calibration.py`
also provides a minimum-records gate for the alternate optimizer. These are
distinct paths; the example does not claim to reproduce the paper's full
confidence-coupled risk controller. `src/episodic_memory.py` stores/retrieves
episodes using embeddings; gating is applied by its caller. It is not loaded by
this example, avoiding embedding downloads.

The released audit document describes what may be shared. Runtime code still
places raw responses in context (`llm_response_raw` in the experiment runner and
`llm_raw_response` in the alternate optimizer); the experiment runner passes
context into result records. The new example does not use these paths or export
their logs. Existing paper result exports intentionally exclude raw responses.

## Where Claude fits

These are intended research roles, not a record of Claude use in this task:

- Candidate proposal: suggest a small set of configurations within a declared
  search space; attach hypotheses separately from measurements.
- Experiment-design critique: identify missing controls, inconsistent budgets,
  premature stopping criteria, and unclear evaluation questions.
- Search-space reasoning: explain parameter interactions and question whether
  domain-specific bounds have supporting evidence.
- Failure analysis: propose explanations for failed trials for subsequent tests.
- Documentation and reproducibility review: inspect commands, schemas, evidence
  links, and claims for omissions or unsupported inferences.

For a future Claude session, a researcher could ask: “Review this declared
search space and experiment plan. Propose three testable hypotheses, identify
which measurements could refute each, and flag assumptions about splits and
training budgets. Do not report unmeasured performance.” A human should review
the inputs before sharing them. This prompt has not been submitted to Claude
during this task.

## What Claude is not trusted to decide alone

Claude does not establish metric truth, whether training actually ran, whether
execution honored a config, statistical significance, absence of data leakage,
split correctness, or the final evidence verdict. Schema validation establishes
structural compliance, not scientific validity. Calibration diagnostics require
real prediction/outcome pairs, and a passing gate is not a proof of significance
or generalization. Human review remains necessary.

## Minimal workflow

```text
research objective
    -> Claude-assisted proposal / critique
    -> strict schema validation
    -> calibration / risk-control layer
    -> actual model evaluation
    -> measured result artifact
    -> human + deterministic checks
```

That is the intended end-to-end research workflow. The delivered offline demo
substitutes a seeded sampler for Claude, rejects malformed candidates, and runs
the existing gate with no measurement history. The gate closes with
`insufficient_records`. Selection and evaluation remain deferred; the example
does not invent an accuracy value to complete the diagram. The original custom
runner's proxy uses Python's process-dependent `hash()`, so its score is neither
a training measurement nor a suitable cross-process reference for this example.

## Reproducible demo instructions

From the release artifact root:

```bash
python -B examples/claude_for_science/run_demo.py
python -B examples/claude_for_science/run_demo.py --check
python -B examples/claude_for_science/test_demo.py
python -B scripts/validate_release_artifact.py
```

The [example README](../examples/claude_for_science/README.md) documents the small
CPU dependency subset, exact expected behavior, and output at
`examples/claude_for_science/output/report.json`. A checked-in
[synthetic fixture](../examples/claude_for_science/expected_output.json) allows
inspection without executing anything. No API key or dataset is needed.

There is no live Claude command: adding a provider would expand the deliberately
restricted release surface and require changing provider-specific schema and
logging boundaries. The task therefore follows Case C and keeps core code and
dependencies unchanged. The development workspace outside this release has
separate Claude support using `CLAUDE_API_KEY`; it is not a portable dependency
of the release and was not used. No `ANTHROPIC_API_KEY` value was inspected.

If a live adapter is implemented later, record the actual model identifier,
UTC timestamp, prompt schema version/hash, sanitized structured candidates, and
validation verdict. Enforce one to three calls, omit raw responses and keys,
and retain the label `proposal-only demo` until actual evaluation is evidenced.

## Evidence versus model suggestion

| Item | What it establishes | What it does not establish |
|---|---|---|
| Candidate parameters | A proposed configuration | That training ran or performance improved |
| Rationale or predicted mu/sigma | A hypothesis or model estimate; synthetic in this demo | Measured accuracy or calibrated uncertainty |
| Strict validation result | Required fields, finite numbers, types, bounds, categories, deduplication | Appropriate scientific ranges or leak-free data |
| Closed calibration gate | No measured records are available in this demo | Empirical reliability of Claude or the paper controller |
| Real evaluation artifact, if later added | Recorded measurement under a documented execution | Significance without suitable statistical analysis |
| Ledger and preservation checks | What was changed and verified during this task | Unobserved historical use of Claude or Claude Code |

## Workshop for PhD/Postdoc researchers: 55 minutes

| Minutes | Activity | Participant outcome |
|---|---|---|
| 0–8 | Define a low-budget question and discuss mixed search spaces | Separate a hypothesis from a claim |
| 8–18 | Inspect the config and an intended Claude critique prompt | Identify assumptions and allowable proposals |
| 18–28 | Run the offline demo and inspect accepted/rejected candidates | Explain type, bounds, and duplicate checks |
| 28–38 | Inspect the closed gate; discuss real history and risk scoring | Explain why evidence is insufficient to proceed |
| 38–48 | Design a future training/split/metric audit and failure analysis | List measurements and checks required for a claim |
| 48–55 | Review the ledger and revise application wording | Produce a defensible evidence-backed description |

No participant needs an API account or GPU. This is a proposed workshop outline,
not evidence that a workshop has been delivered.

## Limitations

The demo is proposal-only and contains synthetic predictions, no real dataset,
no fitted calibration, no risk ranking, and no training. It is neither a paper
result nor cross-domain proof, a Claude benchmark, or a demonstrated improvement
in HPO. Strict local validation differs from the existing paper repair policy.
The local environment is not the recorded AutoDL environment. The original workspace copy
had no Git metadata. The files have since been transferred into the deployment
checkout, with its baseline commit recorded in the evidence ledger. That checkout
has a pre-existing missing `docs/docker.md` failure in its release validator;
resolve that separate issue before claiming release-wide readiness.

The [evidence ledger](../campus_phd/CLAUDE_RESEARCH_EVIDENCE.md) is the source of
truth for task-specific claims. Use only the qualified wording in
[application snippets](../campus_phd/APPLICATION_SNIPPETS.md), and verify the
manuscript's current status before submitting an application.
