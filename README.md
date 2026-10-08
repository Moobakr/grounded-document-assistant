# Intelligent Document Assistant

A PDF learning assistant with local RAG retrieval and opt-in Groq explanations.
Search works offline; Groq tutoring needs internet and your own API key. An optional
local excerpt mode remains available without a cloud account. No Azure subscription is used.

The project provides a local web interface and command-line application.
A multi-user service and hosted demonstration are future milestones.

For a reproducible web deployment, see [Docker operations](docs/operations.md).
The [architecture](docs/architecture.md) explains the data flow, decisions, privacy
boundaries, and current limitations. Software CI is defined in `.github/workflows/ci.yml`;
it runs after the source is published to GitHub, without private PDFs or Groq credentials.

## What Works

- CPU embeddings loaded from local model files, with two threads by default.
- Token-aware chunks that preserve document names and PDF page numbers.
- Multiple documents, content-hash deduplication, and transactional SQLite storage.
- Exact cosine search without starting a language model.
- Duplicate retrieved passages are omitted so repeated lecture headers do not fill the context.
- Optional llama.cpp excerpt selection with verified quotations and application-added citations.
- Model fingerprint checks that prevent mixing incompatible embedding spaces.
- Document deletion, JSON output, and a configurable index location.
- A responsive document library and research workspace, with real local PDF previews.
- PDF upload, scoped search, source inspection, clipboard copy, and JSON export.
- Groq-powered explanations, topic summaries, practice questions, and bounded follow-up context.

SQLite and streaming exact search keep this first single-user version simple.
They do not establish large-scale or multi-user performance. PostgreSQL/pgvector
and a recoverable ingestion worker can be introduced with the service milestone.

## Setup

Python 3.12 is recommended. In PowerShell, from the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.cli prepare
```

Native requirements pin top-level versions. The Docker build also constrains transitive
Python dependencies with the tested `requirements-linux.lock` and pins its Python base
by image digest. This is version pinning, not a wheel-hash lock or a promise of
bit-identical images on all platforms. The explicit `prepare` command downloads the Apache-2.0
`sentence-transformers/all-MiniLM-L6-v2` model. It resolves the requested revision
to a commit and records it in `models/embeddings/download.json`. Internet is needed
for dependency/model setup, not ordinary ingestion or search. Do not copy an old
Azure `.env` into a container or repository; this application does not read it.

## Local Web Interface

After the base setup, install the web dependencies and start the interface:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-web.txt
.\.venv\Scripts\python.exe -m src.web --port 8765
```

Open [the local workspace](http://127.0.0.1:8765). It reuses `data/index.sqlite3`
and the local embedding files. The browser's Search mode does not need a language
model. Assistant mode defaults to Groq: open **AI settings** in the top toolbar,
enter your Groq key, and choose a model. Choose Explain simply, Answer, Summarize
topic, or Practice questions. Ask a follow-up to continue the topic. New conversation,
scope changes, and provider changes clear the recent conversation. Only the last two
successful turns are included, with each previous answer capped at 3000 characters.
For offline excerpts, select Local excerpts and start `python -m src.cli --threads 2 serve`
in another terminal. That small local model remains experimental.
If either generator fails, retrieved passages remain available.

This machine also has an isolated `.web-venv` that reuses its preinstalled AI packages
through `--system-site-packages`, avoiding another PyTorch download. Its launcher is:

```powershell
.\.web-venv\Scripts\python.exe -m src.web --port 8765
```

Native startup binds to `127.0.0.1`. Docker explicitly uses `--host 0.0.0.0` inside
the container and publishes only to the host loopback address. The app accepts at most 25 MB and 300 pages per uploaded
PDF, with batches of up to 10 files in the UI. Indexing, querying, and deletion are
serialized within the web process; another active operation returns a busy response.
SQLite connections are created and closed in their worker thread, while the embedding
model is reused. It loads lazily and remains in memory until the web server stops.
These are lightweight single-user controls, not a distributed worker queue or hard
CPU/RAM quotas. Do not run concurrent CLI writes during web operations.

Uploaded originals are stored by content hash in `data/uploads/`; rendered pages are
cached in `data/previews/`. For older CLI-ingested files, the app looks in `documents/`
and verifies the PDF content hash before serving it. If an original is absent, search
still works, while previews are unavailable. Upload the identical PDF to restore access.
Deleting through the UI removes the index entry, managed upload, and cached previews.
Original files in `documents/` are never deleted. CLI deletion removes only the index
entry and can leave managed files behind; use the web UI for managed-copy cleanup.

Web mutations require a process-local workspace token and reject cross-origin requests.
There is no permissive CORS or public binding in the documented deployments. A key entered in AI settings travels
once from the browser to the loopback server; the input is cleared and never saved in
browser storage. Keys are never returned by status APIs, embedded in assets, or written
to disk. The trusted local server holds the key in memory until disconnect/restart.
Static assets are served only from `src/static/`; the index, models, and `.env` are not
public directories. Private responses are marked no-store, and document text is rendered
as text, not HTML. PDF parsing/rendering is not process-sandboxed: this is a trusted,
local single-user tool, not a safe public upload endpoint or an authenticated hosted app.

Fonts and Lucide icons are local, with no runtime CDN requests. The pinned icon package
and its license are vendored under `src/static/vendor/`; `python -m scripts.prepare_web`
is an explicit online asset-rebuild command that checks npm's SHA-512 package integrity.
For the local provider, the UI checks the model's loopback `/health` endpoint, which is exempt from resetting
the idle timer in [llama.cpp's sleep mode](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md#sleeping-on-idle).
Stop the web server with Ctrl+C when running it in a terminal, and stop the model
server separately when finished. There is no automatic model download or training.

## Groq Learning Assistant

This is still RAG: PDF -> local token-aware chunks -> local MiniLM embeddings ->
SQLite cosine retrieval -> selected passages + question -> Groq -> generated explanation
with clickable source citations. Whole PDFs and embedding vectors are not uploaded.
Cloud requests include passage text, filename/page labels, the question, and recent
conversation. Do not use Groq mode for material you cannot send to a third party.
The browser talks only to your server; only the backend calls Groq over HTTPS.

The default model is `openai/gpt-oss-120b`, with `openai/gpt-oss-20b` selectable.
Both support [Groq strict structured outputs](https://console.groq.com/docs/structured-outputs).
The adapter uses the existing httpx dependency and a fixed Groq endpoint. It validates
response shape and inline citation IDs against the current retrieved passages; it does
not establish semantic entailment or guarantee factual accuracy. The tutor is instructed
to distinguish illustrative examples from PDF facts and abstain on unsupported topics.
Summaries cover retrieved material, not guaranteed full-document coverage. Follow-up
retrieval carries the recent question topics; it does not perform model-based query rewriting.

Groq generation uses remote hardware, so the local Qwen server is unnecessary in this mode.
Search/indexing still uses your CPU. Groq has [free-plan limits](https://console.groq.com/docs/rate-limits)
and paid plans; your account determines billing. The app never upgrades your plan or retries
automatically. Set account spend limits before using a paid key. Groq's
[data controls](https://console.groq.com/docs/your-data) govern provider-side retention;
local storage does not make cloud requests private or offline.

An existing `GROQ_API_KEY` environment variable is also recognized at server startup.
The app does not load `.env` or reuse Azure credentials. Disconnect removes the running
server's copy, even if the startup environment still contains it; restarting can reload it.
For CLI generation, explicitly choose the cloud provider (key must be in the environment):

```powershell
python -m src.cli ask "Explain correlation with an example" --provider groq --task explain
python -m src.cli ask "Linear regression" --provider groq --task quiz
```

CLI defaults remain local. Do not pass keys as command-line arguments or commit them.
The web key is process-local and is not shared with a separate CLI process.

## Index And Search

Use your environment's Python executable in place of `python` if necessary:

```powershell
python -m src.cli ingest documents
python -m src.cli list
python -m src.cli search "What regression methods are covered?"
python -m src.cli search "What is preprocessing?" --top-k 5 --json
python -m src.cli delete FULL_DOCUMENT_ID
```

Documents and vectors persist in `data/index.sqlite3`. Re-uploading identical PDF
bytes skips embedding. Modified PDFs are separate versions; delete the previous
document by ID if it should no longer be searchable. Identical files with different
names retain the first indexed name. Use `--document FULL_DOCUMENT_ID` to restrict
search or answers to one PDF.

Global options go before the command:

```powershell
python -m src.cli --threads 2 --index data/research.sqlite3 ingest documents
```

Chunk size and overlap are embedding-model tokens. The default 180-token chunks
fit MiniLM's sequence length. Blank pages are skipped. Image-only and encrypted
PDFs are rejected; OCR is not implemented in this milestone.

## Optional Local Answers

On Windows x64, run this explicit one-time online setup:

```powershell
python -m src.cli prepare-generation
```

It downloads Qwen's Apache-2.0 `Qwen2.5-1.5B-Instruct` Q4_K_M GGUF (1.12 GB)
and the official llama.cpp b11499 Windows CPU runtime (20 MB). The model revision
is recorded and the runtime ZIP is verified against the release SHA-256 checksum.
On other platforms, install a CPU build from the official
[llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases) and supply the
binary and model paths to `serve`. There is no need to train a model or use the GPU.

In one terminal:

```powershell
python -m src.cli --threads 2 serve
```

In another terminal:

```powershell
python -m src.cli ask "What regression methods are covered?"
```

The launcher uses CPU-only inference, one generation slot, a 4096-token context,
and a 384-token output limit. It creates a local authentication key in
`data/local-server.key`, restricts browser origins, and unloads the model after
120 idle seconds. The CLI reads that key automatically; it is not a paid API key.
The client checks prompt length with the local
server's tokenizer and reserves room for the chat template. Generation requests
only accept loopback HTTP URLs, ignore HTTP proxy environment settings, and do
not follow redirects. Stop the server when you finish to release its memory.

The default `evidence` answer mode selects numbered excerpts instead of writing a
free-form explanation. Excerpts are prepared from PDF bullets and sentence boundaries;
the model returns IDs, and the application copies the selected text. Long excerpts are
split at 600 characters, so poorly formatted PDFs can still yield incomplete phrases.
Every excerpt must occur in its cited passage after only
whitespace normalization; case and punctuation must still match. If any quote is
invented or paraphrased, the entire answer becomes a controlled abstention. Empty
selections also abstain. A second short local-model call reviews whether the excerpts
directly answer the question; a rejected review becomes an abstention. This is a
heuristic from the same small model, not an independent judge or an entailment guarantee.
It adds latency and can falsely refuse answerable questions.
Invalid IDs, malformed JSON, and truncated responses fail
with an error. The application adds citations, rather than trusting model-written IDs
inside prose. This establishes text provenance, not relevance or semantic correctness:
the model can still select an irrelevant quote or omit part of the answer.

The original free-form generator remains available as an explicitly experimental
baseline, without quotation validation:

```powershell
python -m src.cli ask "What regression methods are covered?" --answer-mode draft
```

Both modes use temperature zero and seed 42 for controlled comparisons. This does
not guarantee identical results across runtime versions or hardware.
Cosine similarity is not a probability of correctness. Retrieval oversamples threefold
before removing duplicate text and may return fewer than the requested passages.

The small Qwen 1.5B model is an experimental baseline. During development it
incorrectly classified unrelated syllabus topics as regression methods, even when
retrieval found the correct page. That failure is recorded in
`evaluation/development_cases.json`. Structured source IDs fix reference formatting,
not factual grounding. Evidence mode reduces unsupported prose by returning actual
document text; model selection and held-out answer-quality evaluation remain open work.

## Docker Web Application

Docker is optional. The image now includes the full interface, API, PDF renderer,
and CPU embedding dependencies. It contains neither private PDFs nor API keys.
Building and one-time model setup need internet; search uses local model files.

```powershell
docker compose build
docker compose run --rm model-setup
docker compose up -d app
```

Open [the Docker workspace](http://127.0.0.1:8766). It uses separate persistent
model/data volumes; your native workspace at port 8765 is untouched. The service
runs as UID 10001 with two CPUs, a 4 GB memory limit, a read-only root filesystem,
and host-loopback port publishing. Groq connection is through the local settings
dialog, and its key is kept in memory. Stop with `docker compose down` without
`--volumes` to preserve data. Publishing an image does not host a website.

The optional local Qwen server is not bundled in Compose; container operation
supports local search and Groq tutoring. Native local generation remains available.
The image also retains the CLI: override its default web command with
`python -m src.cli ...`. See [operations](docs/operations.md) for limits, backups,
diagnostics, private-data handling, and CI behavior.

## Verification

The complete suite includes web/API tests. Tests inject model/HTTP providers, so
the lightweight `requirements-test.txt` is sufficient without AI model downloads:

```powershell
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -v
```

Tests use controlled embeddings and an HTTP mock: no model downloads, API keys,
or external network access are needed. They check persistence, deduplication,
model compatibility, failed ingestion rollback, page metadata, source propagation,
and local-generation failures. The reliability tests cover invented/paraphrased quotes,
wrong-source quotes, malformed/truncated responses, and evaluation score denominators.
Groq tests use a mock HTTPS transport, including structured answers, prompt/history
propagation, invalid citations, rate limits, timeout handling, and secret non-disclosure.
These are correctness tests, not RAG quality scores or live Groq evaluation.

For browser verification, with the web server running and the five lecture PDFs indexed:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-browser-test.txt
.\.venv\Scripts\python.exe -m scripts.verify_web
```

The browser check uses installed Chrome (or `--channel msedge`), so it does not need
another browser download. It verifies real search, filtering, scoped queries, copy,
JSON export, upload/deduplication/deletion, model-offline fallback, and mobile navigation.
If the local model is running, it also checks answer rendering and citation clicks.
Only its own fixture PDF is added/deleted; original document IDs must remain unchanged.
Browser external requests are blocked and reported. This does not audit all server or
OS network traffic. Screenshots and reports are stored in private
`data/browser-verification/`, not committed publicly with the corpus text.

Test the tutor's complete desktop/mobile flow without a real key or external inference:

```powershell
python -m scripts.verify_assistant
```

This starts and stops its own isolated temporary workspace, exercising the production
Groq adapter with a mock HTTP transport. It checks connection/disconnect, cloud disclosure,
safe answer formatting, nonblank PDF previews, citations, follow-ups, learning modes,
export, and rate-limit fallback. Its report/screenshots are in `data/assistant-verification/`.
Synthetic responses prove integration behavior, not model intelligence; a live Groq run
and held-out factual/tutoring evaluation still require a real key and separate review.

## Development Evaluation

The versioned `evaluation/cases.json` contains 11 source-labelled development cases:
eight answerable questions and three missing-information questions. They include
the previously observed regression-category failure. The cases were used while
building this implementation and must not be advertised as a held-out benchmark.

Run retrieval alone, or compare both answer modes while the local server is running:

```powershell
python -m scripts.evaluate --output data/evaluation-retrieval.json
python -m scripts.evaluate --generate --output data/evaluation-evidence.json
python -m scripts.evaluate --generate --answer-mode draft --output data/evaluation-draft.json
```

Reports include all questions, retrieved passages, answers, errors, timings, dataset
and code hashes, embedding fingerprint, local model manifests, and dependency versions.
Your existing index is not re-ingested or deleted. External TCP connections are blocked
in the evaluator process; this is not an OS-wide firewall or a model-server audit.
The report contains document excerpts and belongs in private `data/`, not a public repo.

Metrics are deliberately separate:

- Relevant-page hit at top-k and mean reciprocal page rank (not chunk recall).
- Required/forbidden phrase checks, which are proxies for the human rubric, not semantic grading.
- Correct abstention on unanswerable questions and false abstention on answerable ones.
- Citation numbers in range and citations to at least one labelled relevant page.
- Exact-quote grounding for evidence outputs only; draft grounding is not measured.

Every failed case stays in the answer-check denominator. Citation/quotation rates
are reported only for non-abstained outputs with their explicit denominators. Always
inspect answer coverage and false abstention alongside them. First retrieval includes
embedding-model load, so recorded timings are smoke measurements, not a benchmark.
`--strict` exits nonzero for failed checks; operational errors always exit nonzero.
Before claiming general quality, add an untouched test corpus and human semantic review.
See [development results](evaluation/results.md) for the measured tradeoff: the
default currently passes 5/8 answerable checks and 3/3 missing-information checks.
The regression-category error remains; neither mode is production-ready.

After model setup, verify the real lecture PDFs with external TCP connections
blocked in the Python process:

```powershell
python -m scripts.verify_offline
python -m scripts.verify_offline --generate
```

The second command requires the local model server to be running. The check uses
a temporary database, tests deduplication and reopening, and records sources and
elapsed times in `data/verification.json`. It does not modify your main index.
This is a smoke check, not an OS firewall audit or a quality benchmark.
The generation check includes a course-contents question and a controlled
unanswerable-question check. Development examples must not be used as a held-out
test set when reporting final quality scores.
