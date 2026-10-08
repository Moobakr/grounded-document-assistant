# Development Results

Measured locally on 2026-10-08 using the five lecture PDFs (248 indexed chunks),
MiniLM embeddings, and Qwen2.5-1.5B-Instruct Q4_K_M through llama.cpp b11499.
Embedding and generation workloads used two CPU threads. No cloud inference was used.
External TCP was blocked in the evaluator process, not at OS or model-server level.

These 11 cases were used during implementation. They are development checks, not
a held-out benchmark or a claim of general accuracy. Required/forbidden regexes are
answer proxies; the same model's relevance review is not an independent judge.

| Check | Evidence + Review | Free-Form Draft |
| --- | --- | --- |
| Relevant page retrieved at top 3 | 8/8 | 8/8 |
| Mean reciprocal relevant-page rank | 0.9167 | 0.9167 |
| Answerable question checks | 5/8 | 7/8 |
| Unanswerable questions correctly abstained | 3/3 | 2/3 |
| False abstention on answerable questions | 1/8 | 0/8 |
| All question checks | 8/11 | 9/11 |
| Citation IDs in range, answered outputs only | 7/7 | 9/9 |
| At least one citation to a labelled page, answered answerable outputs | 7/7 | 8/8 |
| Exact quote provenance, answered outputs only | 7/7 | Not measured |
| Operational errors | 0 | 0 |

The conservative default provides mechanically verified document excerpts and
improves abstention in these cases, but passes fewer answerable question checks
than the draft baseline. There is no evidence here that overall answer quality
is better. Perfect quote provenance is guaranteed by copying/validation and must
not be marketed as perfect factual correctness or relevance.

## Remaining Failures

- `course_contents`: selects only the heading, not the requested topic list.
- `regression_category`: includes unrelated syllabus topics and examples. The
  heuristic relevance review does not catch this semantic category error.
- `collection_initial_step`: refuses despite a relevant supporting page being retrieved.

The draft mode also fails `regression_category`. For `missing_evaluation_result`,
it returns a generic accuracy formula instead of abstaining about the user's
unknown experiment. The default mode correctly abstains on this question.

The tiny model is still experimental and not production-ready. Further model
comparison and independent semantic review are required; an API/interface alone
will not solve these failures. No peak-RAM or scalability measurements were made.

## Artifacts

`data/evaluation-final.json` records the final default-mode run, including source
passages, answers, errors, configuration, and dataset/code/model metadata.
`data/evaluation-draft.json` records the draft comparison. Earlier excerpt-copying
and unreviewed selection reports were retained, rather than deleting failed runs.
The evaluated draft function was unchanged by the subsequent evidence-mode review.

The final run used `--strict` and exited with code 1 for the three failed answer
checks, not an operational error. The complete offline unit suite passed 56 tests.
Full reports contain private corpus text and remain in ignored `data/`.

See `evaluation/README.md` for reproduction and the held-out human-review protocol.
