# Application snippets

These drafts describe an offline workflow designed to integrate Claude. No
Claude API call or Claude Code usage was verified in this task. The first-person
wording is for the researcher to review and adopt only if it accurately reflects
their ownership and contribution. The section titled “How I use Claude in
research” answers a possible application field without claiming actual use.
Word counts use whitespace-separated words, excluding headings and these notes.

## Project summary — 50 words

I built a reproducible research workflow designed to integrate Claude into hyperparameter experimentation while separating suggestions from evidence. An offline demonstrator validates synthetic candidate configurations and exposes insufficient calibration history. Supporting documentation and an evidence ledger preserve manuscript boundaries, explain verification requirements, and help researchers communicate only defensible experimental claims.

## How I use Claude in research — 100 words

I built a workflow designed to integrate Claude as a proposal and critique assistant for hyperparameter research. The current demonstrator runs offline, without Claude calls or model training. It reuses existing candidate sampling, prompt construction, and calibration gating components while validating configurations against explicit constraints. Synthetic predictions are clearly labeled, and missing measurements keep the gate closed. My intended Claude workflow includes reviewing search spaces, questioning experimental assumptions, and proposing explanations for failures. Deterministic checks and actual evaluations would establish evidence. An accompanying ledger distinguishes implemented behavior, verified results, and future integration plans so application statements remain accurate and auditable.

## Why this matters for PhD researchers — 150 words

PhD researchers often work with limited compute, incomplete evidence, and pressure to explain results clearly. Language models can support exploration, but persuasive suggestions can blur the distinction between a plausible hypothesis and an observed outcome. This project makes that boundary visible through a small, reproducible example. Researchers can inspect proposed configurations, see invalid values rejected, and observe a calibration gate remaining closed when measurements are absent. The demonstration requires no API account, dataset download, or GPU, making its structure accessible for discussion across research groups. A companion workflow outlines where Claude could assist with experiment critique, search space reasoning, failure analysis, and documentation review. An evidence ledger connects claims to files and executed checks while preserving the scope of an existing manuscript artifact. The contribution is practical research discipline: documenting what happened, what remains uncertain, and which measurements would be needed before making stronger scientific claims or deciding experimental steps.

## Workshop description — 80–120 words

This proposed 55 minute workshop helps PhD and postdoctoral researchers distinguish model suggestions from experimental evidence. Participants inspect a mixed hyperparameter search space, run an offline candidate validation example, and examine why a calibration gate stays closed without measured history. They then discuss how Claude could critique experiment designs and propose testable explanations for failures. The final activity uses an evidence ledger to revise research claims and identify missing verification steps. No API account, GPU, or dataset download is required for the demonstration. Participants leave with a reproducible example and a concrete plan for documenting future experiments responsibly.

## Five evidence-backed bullets

- Built a separate non-manuscript demonstrator: [example and scope](../examples/claude_for_science/README.md).
- Checked three synthetic candidates and rejected an out-of-bounds fixture: [expected output](../examples/claude_for_science/expected_output.json).
- Exposed missing measurement evidence through the existing calibration gate: [gate invocation](../examples/claude_for_science/run_demo.py).
- Tested offline execution and validation boundaries without provider imports: [tests](../examples/claude_for_science/test_demo.py).
- Documented intended Claude roles, verification boundaries, and a proposed workshop: [workflow](../docs/claude_for_science_workflow.md).

See the [evidence ledger](CLAUDE_RESEARCH_EVIDENCE.md) for executed checks and
limitations. These bullets do not establish actual Claude use, workshop delivery,
improved HPO performance, or reproduction of the manuscript experiments.
