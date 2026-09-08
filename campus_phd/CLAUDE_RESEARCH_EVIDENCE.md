# Claude research evidence ledger

## Deployment checkout transfer ? 2026-09-09 (Asia/Seoul)

Files were transferred into the actual deployment repository at commit
`834081c05d0e7794d457d91866384e0f961064c6` (`Refine figures, docs`). The working tree
was clean before transfer. This checkout contains newer paper files than the
original workspace snapshot. Only the new task files were copied; the existing
README received the demo link, and all existing paper/code/config files were
preserved. No commit, staging, push, or remote publication was performed.

`BASELINE_SHA256.json` now records the 75 tracked files in this deployment
checkout before transfer, plus its Git HEAD. It replaces the original workspace's
80-file baseline for the portable preservation check. Historical validation
results below refer to the original workspace unless explicitly identified here.

Deployment validation commands:

- `python -B examples/claude_for_science/run_demo.py --check`
- `python -B examples/claude_for_science/test_demo.py`
- `python -B campus_phd/verify_delivery.py`
- `git diff --check` and `git diff -- README.md`
- `python -B scripts/validate_release_artifact.py`

The demo fixture check, all five tests, delivery checks for 75 baseline files,
and `git diff --check` passed in the deployment checkout. New document links,
JSON, word counts, and secret patterns were checked; the README link check covers
only its appended section so unrelated existing links are not misreported as
part of the new demo. No generated output directory was copied.

The deployment release validator fails because `docs/docker.md` is missing.
This failure was observed before copying any task files and remains unchanged;
the existing validator and paper files were not modified to conceal it. Treat
release-wide readiness as PARTIAL until this separate documentation issue is
resolved. The offline example remains independent of Docker documentation.

## Original workspace scope and provenance

Task date: 2026-09-08. Status: **READY for local offline review**, with Git/remote
provenance and application ownership requiring user verification. This ledger
is the source of truth for claims about this task, not a certification of
historical experiments. The task was performed in a Codex session; no Claude
API call or Claude Code execution was performed or verified during it.

The workspace README identifies `release/npl_llm_hpo_artifact/` as the public
artifact. All paths below are relative to that directory unless stated otherwise.
The task's suggested standalone repository name was not assumed to be a verified
remote. Both the workspace and release directory lack Git metadata. No repository
was initialized, and no push, PR, release, or tag was created.

## What existed before this task

- An under-review manuscript artifact with OpenAI-only public provider code,
  paper configs, compact results and checksums, figures, Docker files, and a
  recorded AutoDL environment. The release validator passed before changes.
- The public `UnifiedLLMClient` used `OPENAI_API_KEY`, defaulting to `gpt-5.2`.
  Paper configs also named `gpt-5.2`. Historical execution identities were not
  independently verified from these config strings.
- Outside the release, the development workspace's client contained OpenAI,
  Claude, and Gemini paths. Its Claude path used `CLAUDE_API_KEY`; the workspace
  installation note instead named `ANTHROPIC_API_KEY`. That mismatch and source
  support are not evidence of a successful historical Claude run. No secret file
  was read and no key values were inspected.
- `PromptBuilder`, acquisition functions, two calibration implementations,
  an alternate calibrated optimizer, embedding-based episodic memory, and a
  paper experiment runner with confidence/risk control.
- The custom runner's `--dry-run` sampled a candidate and used a proxy based on
  process-dependent Python `hash()`. Its labels did not establish actual TPE or
  LLM execution. It was not used as measured performance evidence here.
- The paper runner placed raw responses into context logs and result records;
  the alternate optimizer also returned raw response context. The public result
  exports excluded raw responses, but runtime logging still needs care.

### Sources actually inspected and missing-file mapping

Read the workspace and release READMEs, release audit schema, all paper configs
and custom templates, custom example/result/figure READMEs, requirements,
Dockerfile, compose file, placeholder environment file, and reproduction script.
Read `src/prompt_builder.py`, `src/unified_llm_client.py`, `src/llm_client.py`,
`src/acquisition.py`, `src/calibration.py`, `src/uncertainty_calibration.py`,
`src/optimizer.py`, `src/episodic_memory.py`, `src/custom_runner.py`, and
`src/custom_data.py`; inspected the experiment runner's prompt, parsing,
validation, scoring, selection, and logging paths. Read both
`scripts/validate_release_artifact.py` and `scripts/summarize_results.py`.
The summarizer writes paper CSVs/manifests and was deliberately not run.

| Requested document absent from this release | Actual source inspected |
|---|---|
| `docs/reproduce_paper_results.md` | README reproduction section, paper configs, `scripts/run_paper_experiments.sh` |
| `docs/custom_dataset_hpo_recipe.md` | README custom section, custom runner/loaders, example configs/templates |
| `docs/release_manifest.md` | README results/integrity sections, release validator and summarizer; result checksums validated by the validator |
| `docs/installation.md` | README quick start/environment sections, requirements/Docker files; inspected the workspace's separate `docs/INSTALLATION.md` installation section |

## What was newly done

Selected **Case C for the public artifact**: offline demonstrator plus workflow
documentation. The release intentionally restricts providers and uses OpenAI
response formatting; transplanting the workspace Claude path would add SDK,
environment, schema, and raw-log behavior without a necessary live demo. No
provider, dependency, default, paper config, core module, or validator was changed.

The demo reuses existing custom sampling/search-space checks, prompt construction,
and the calibration gate. Local strict candidate validation rejects missing or
extra fields, invalid categories, nonfinite values, booleans as numbers,
fractional integer values, out-of-bounds values, and duplicates. It does not
claim to exercise the existing paper repair implementation.

Three seeded synthetic proposals pass; a fourth deliberately invalid fixture is
rejected. The measured history remains empty, so the gate closes and selection
is deferred. No calibration is fitted, no risk scoring is run, no dataset is
loaded, and no model is trained. Synthetic mu/sigma are clearly labeled and
the measured metric is null. Only allowlisted candidate fields and verdicts
are written; rejected payloads and free-text rationales are omitted.

## Exact files created or modified

| Path | Purpose |
|---|---|
| `docs/claude_for_science_workflow.md` | English workflow, trust boundary, intended Claude roles, 55-minute workshop |
| `examples/claude_for_science/README.md` | Five-minute entry point and execution instructions |
| `examples/claude_for_science/config.json` | Bounded offline mixed search space and seed |
| `examples/claude_for_science/run_demo.py` | Offline proposal, strict validation, and closed-gate report |
| `examples/claude_for_science/test_demo.py` | Five tests covering validation, deduplication, offline execution, and evidence semantics |
| `examples/claude_for_science/expected_output.json` | Checked-in synthetic output fixture |
| `examples/claude_for_science/.gitignore` | Excludes generated local output directory |
| `campus_phd/CLAUDE_RESEARCH_EVIDENCE.md` | This task ledger |
| `campus_phd/APPLICATION_SNIPPETS.md` | Qualified English application drafts |
| `campus_phd/BASELINE_SHA256.json` | Hashes/sizes of the deployment checkout's 75 pre-transfer tracked files |
| `campus_phd/verify_delivery.py` | Repeatable preservation, JSON, link, wording, and secret checks |
| `README.md` | Appended a short non-manuscript workflow link after validation |

The execution also generated `examples/claude_for_science/output/report.json`,
which is ignored by the example's Git rules. No other pre-existing release file
was modified. Temporary pre-edit snapshot/diff files were kept outside the
release; they contain no inspected key values or raw provider responses.

## Original workspace execution and verification

Commands below run from the release root, except the initial workspace Git
commands. Use `-B` to avoid forbidden Python bytecode inside the release.

| Command/check | Observed result |
|---|---|
| `git status --short` and `git log --oneline --decorate -n 30` in workspace and release root | Failed: no Git repository; branch/commit provenance unavailable |
| `python --version` | Python 3.11.4 |
| `python -B scripts/validate_release_artifact.py` | Passed before and after changes, including paper result checksum validation |
| `python -B examples/claude_for_science/run_demo.py` | Passed; generated the local report with no API call or training |
| `python -B examples/claude_for_science/run_demo.py --check` | Passed against the checked-in synthetic fixture |
| `python -B examples/claude_for_science/test_demo.py` | Five tests passed; includes separate processes with hash seeds 1 and 987 |
| `python -B campus_phd/verify_delivery.py` | Passed; 80 baseline files checked, all unchanged except the preserved README plus appended link; JSON, Markdown file links, new-file whitespace, secret/local-path patterns checked |
| Application paragraph counts | 50 / 100 / 150 / 98 whitespace-separated words; five evidence bullets |
| `git diff --check` and `git diff` | Unavailable as a working-tree review because this copy has no Git metadata |
| `git -c core.autocrlf=false diff --no-index --check <before-snapshot> <release-root>` | No whitespace diagnostics after normalizing new JSON files to LF; no-index exits 1 for the expected differences |
| `git -c core.autocrlf=false diff --no-index <before-snapshot> <release-root>` | Reviewed snapshot diff (also saved with Git's `--output` option); exit 1 denotes expected content differences, not a failed test |

The angle-bracket paths denote local snapshot paths intentionally omitted from
the public ledger. `BASELINE_SHA256.json` and `verify_delivery.py` provide a
portable repeat check without those personal paths. This is local preservation
evidence, not independent remote provenance or a cryptographically signed audit.
The README's original byte prefix is checked; all other baseline files must
match their complete hashes.

Tests remove API-key environment variables in fresh processes, block imports of
OpenAI/Anthropic/dotenv and training libraries, and deny socket connections.
No provider SDK, `.env` loading, or external service is needed by the demo.
Local pandas 1.5.3 and scikit-learn 1.3.0 were used. The pinned AutoDL dependency
stack and Docker build were not executed or independently reproduced.

An initial ad hoc word-count display failed with the Windows console's cp949
encoding. A subsequent display expression was too greedy to count paragraphs;
it was replaced with explicit heading boundaries in the delivered verifier.
The 150-word draft initially had 151 words and was corrected before final
verification. Initial Git comparisons emitted automatic line-ending conversion
warnings; disabling conversion exposed CRLF as whitespace in newly generated
JSON files. Those new files and the demo writer were normalized to LF, then the
check was repeated. Existing file bytes were preserved. These were tooling/draft
issues, not hidden successful checks.

## Claude integration status

- Existing support: **no in the public release**; separate development source
  support exists outside it and was not executed.
- Added support: **no**, Case C; core/default/paper paths preserved.
- Actual Anthropic API/Claude call performed: **no**, zero calls in this task.
- Actual model identifier for a new Claude run: **none**.
- Claude Code usage: **not performed or verified in this task**; historical
  usage is unknown.
- Actual new training or manuscript experiment: **none**.

## What was verified versus proposed

Independently of model suggestions, executable local checks verified the
synthetic fixture, deterministic rejection behavior, closed gate, no-key/no-network
path, source preservation, released checksums, file links, and draft word counts.
The checks were run by the same task agent, not an external research auditor.

Claude-assisted critique, live candidate proposals, experiment design assistance,
future measured evaluations, and workshop delivery remain plans/documentation.
The work supports a claim of building a reproducible workflow **designed to
integrate Claude**, not a claim of having used Claude for the manuscript or having
demonstrated better performance.

## User checks before an application or publication

1. Verify your ownership/contribution and edit first-person wording accordingly.
2. Verify any actual historical Claude/Claude Code use separately; do not infer
   it from provider code, model strings, or folder names.
3. Confirm the manuscript's current status and approved scope of public claims.
4. Review the deployment checkout diff and resolve its pre-existing missing
   Docker documentation before release; authorize publication separately.
5. Rerun the demo and release checks in the environment you plan to share.

Recommended single reviewer-facing path after authorized publication:
`docs/claude_for_science_workflow.md` when the release folder is the repository
root, or `release/npl_llm_hpo_artifact/docs/claude_for_science_workflow.md` when the
whole workspace is the repository. No remote URL was invented.
