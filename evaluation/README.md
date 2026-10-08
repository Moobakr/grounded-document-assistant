# Evaluation Protocol

## Scope

`cases.json` is a versioned development regression suite, not a held-out benchmark.
`development_cases.json` preserves historical observations from the first local-model
milestone. Do not silently replace failing cases or change a rubric to make a run pass.
Keep reports in ignored `data/`: they contain retrieved document text.

The default evidence mode selects IDs of prebuilt excerpts. The application copies
the text, verifies its source, and adds citations. A second short call to the same model
reviews direct relevance and rejects uncertain selections. This heuristic is not an
independent judge. These checks prevent invented answer prose,
but does not prove that a selection is relevant, complete, or interpreted correctly.
The experimental draft mode is a free-form baseline, not the recommended default.

## Reproduce

1. Prepare the models and ingest the five original lecture PDFs as described in the main README.
2. Start `python -m src.cli --threads 2 serve` in another terminal.
3. Run `python -m scripts.evaluate --generate --output data/evaluation-evidence-selection.json`.
4. Run `python -m scripts.evaluate --generate --answer-mode draft --output data/evaluation-draft.json`.
5. Stop the model server when finished. Keep both complete reports, including failures.

Compare only runs with matching corpus hashes, dataset hashes, retrieval configuration,
and model revisions. Check code hashes to understand implementation changes. Local
artifact manifests are recorded, but do not attest the model loaded by an independently
started server. Seed 42 and temperature zero help control variation; runtime/hardware
changes can still affect results. First retrieval includes embedding-model loading.

Use `--strict` for regression checks in automation. An operational error always makes
the command fail; without `--strict`, a completed run with quality failures still
produces a report and exits successfully. The evaluator refuses a missing index or
required source document. It does not re-ingest or delete the existing collection.

## Interpret

Read the counts as well as the rates. Page hit at top-k applies only to the eight
answerable cases. It is not chunk recall. Answer checks use required/forbidden regexes,
not an independent semantic judge. Citation and quotation rates apply only to
non-abstained answers, so read them alongside false abstention and answer coverage.
A fully source-backed quote can still answer the wrong question.

Two observed draft failures illustrate the distinction:

- The regression-category question receives the whole syllabus, including clustering
  and principal components analysis, despite citing the actual contents page.
- The user's unknown model test accuracy receives a generic accuracy formula instead
  of an abstention. The formula does not establish an evaluation result.

## Human Review

Before reporting general quality, build a separate test corpus with questions and
source labels before further prompt/model tuning. Freeze it and keep it untouched.
Use an independent reviewer who can inspect the PDF and does not know the mode label.

For each answer record: direct relevance, factual support of every claim, completeness,
correct cited page, and appropriate abstention. Record disagreements and adjudications.
Evaluate explicit adversarial PDF instructions, distractor passages, contradictory
documents, and missing evidence. Current unit tests do not establish prompt-injection
resistance or semantic safety. Do not send private PDFs to an external judge by default.

Measure latency and peak process memory separately under fixed warm/cold conditions,
with the same corpus and thread limits. Current smoke timings are not performance or
scalability claims. The current system remains a single-user CLI, not a multi-user service.
