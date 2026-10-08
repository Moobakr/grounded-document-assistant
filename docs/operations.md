# Run And Maintain The Application

## Docker Quick Start

Use Docker Desktop with Linux containers and Docker Compose v2. The tested image
targets Python 3.12 on Linux amd64 with CPU-only PyTorch. Windows/macOS hosts need
Docker's Linux VM; do not assume the 4 GB app limit bounds Docker Desktop itself.

The base image is pinned by digest. Python dependencies are constrained to the
tested `requirements-linux.lock` snapshot; the CPU PyTorch wheel comes from its
official index. BuildKit caches package downloads. The model revision is pinned
in Compose. This provides a repeatable tested version set, not wheel-hash verification
or bit-for-bit build identity. Updates require deliberate re-resolution and tests.

From the repository root:

```powershell
docker compose build
docker compose run --rm model-setup
docker compose up -d app
```

Build and model setup require internet. Setup downloads the pinned MiniLM revision,
not a generation model. After setup, local ingestion/search loads only local files.
Open [the container workspace](http://127.0.0.1:8766). Upload your own PDFs, then
open AI settings if you want Groq tutoring. No key is required for search.

The container workspace is separate from the native app's `data/` folder. Existing
native PDFs/index entries are not automatically copied. This prevents accidental
mutation of the original workspace during container verification.

Model setup is idempotent for the configured revision. Use a fresh model/data
workspace when changing embedding models; do not mix incompatible vector spaces.

## Ports And Startup

The default host port is 8766, leaving the native app at 8765 untouched. If occupied,
set `ASSISTANT_PORT` before startup, for example in PowerShell:

```powershell
$env:ASSISTANT_PORT = '8770'
docker compose up -d app
```

In Bash, use `ASSISTANT_PORT=8770 docker compose up -d app`. The mapping remains
host-loopback only. Do not replace it with `0.0.0.0:PORT:8765` for public access.

Native operation still uses `python -m src.web --port 8765` and loopback binding.
Qwen/llama.cpp is optional in native mode only; it is not part of this Compose stack.

## Status And Troubleshooting

```powershell
docker compose ps
docker compose logs --tail 50 app
docker compose exec app python -m pip check
```

`GET /api/health` checks that the web process is responsive without loading the
embedding model or returning secrets. `embedding_ready` checks for the model
manifest, not whether inference succeeded. Docker health can be green before model
setup; the interface then disables indexing/search and reports the missing model.

Common problems:

- **No embedding model:** run `docker compose run --rm model-setup`, then refresh the browser.
- **Groq key needed:** enter your own key in AI settings; it is not stored in the image/volume.
- **Groq unavailable/limited:** retrieved passages remain available; use Search or retry later.
- **Workspace busy:** wait for the current upload/query/deletion. Do not run concurrent CLI writes.
- **Container exited under load:** inspect logs and Docker status before increasing limits. PDF parsing is not isolated in a separate worker.
- **Port already occupied:** choose another `ASSISTANT_PORT`; never stop an unrelated process.

## Persistence And Backups

The Compose model volume contains downloaded embedding files. The private data
volume contains SQLite, managed PDFs, and preview PNGs. Keys entered through the
UI are not persistent; reconnect after restarting the app.

To copy a consistent private workspace backup, stop the app first and use a new
backup destination. Keep it out of public repositories:

```powershell
docker compose stop app
docker compose cp app:/app/data ./workspace-backup
docker compose start app
```

This backup is private corpus data, not a portfolio artifact. Protect it accordingly.
It does not include models; those can be provisioned again at the pinned revision.
Do not copy a live SQLite file while writes are occurring. Restore only into an
empty compatible workspace with the app stopped and correct ownership for UID 10001.

`docker compose down` stops/removes containers and preserves named volumes.
**Do not add `--volumes` / `-v` unless you intend to delete indexed documents and models.**
There is no automated remote backup or account-level access control.

## Verification And CI

The lightweight suite requires no PyTorch, downloaded models, or paid credentials:

```powershell
python -m pip install -r requirements-test.txt -c requirements-linux.lock
python -m unittest discover -s tests -v
```

For browser checks using an already installed Chrome:

```powershell
python -m pip install -r requirements-browser-test.txt
python -m scripts.verify_assistant
```

CI installs Playwright Chromium and passes `--channel chromium`. Browser responses
are synthetic and the workspace is temporary. This proves interface/API integration,
not Groq model quality. The tests never contact Groq or require your real corpus.

With Compose running and its model provisioned, check actual CPU retrieval:

```powershell
python -m scripts.verify_container
```

Use an unconfigured cloud connection so no real inference key is active. The check
uploads a generated fixture, performs scoped semantic search and PDF rendering,
and deletes only that fixture. It never modifies the native workspace. Add
`--restart` only when it is safe to interrupt the Compose app; that option verifies
the URL belongs to this repository's app before restarting it to check persistence.
Its private report is written to `data/container-verification.json`.

`.github/workflows/ci.yml` defines Windows/Linux unit/API jobs, a desktop/mobile
browser job, and a CPU-container build/test/startup job. Actions are pinned to
commit IDs, permissions are read-only, and checkout does not persist credentials.
No deployment, registry publication, or billing change is performed by the workflow.
Publish the source to a GitHub repository to run it remotely; do not publish private
documents, `.env`, keys, databases, local models, or backup folders.

## Local Verification Record

On 2026-10-08, the clean Windows test environment passed 90 tests without model
libraries. The Linux CPU image passed the same 90 tests as UID 10001 with Docker
networking disabled. Desktop/mobile tutor checks passed using mocked Groq responses.
Real MiniLM inference produced 384-dimensional embeddings with CPU PyTorch and no
CUDA runtime while Docker networking was disabled. Compose smoke checks verified
upload, deduplication, search, nonblank previews, missing-key fallback, restart
persistence, and cleanup. These are software/integration checks, not quality scores.

The running Compose configuration was inspected: UID 10001, host-loopback port
8766, read-only root filesystem, 4 GB memory, two CPUs, 128 processes, dropped
capabilities, and no-new-privileges. Remote GitHub Actions results remain pending
until publication and an actual workflow run.

## Portfolio Claims

Accurate: local RAG retrieval, optional cloud tutoring, CPU-focused deployment,
transactional indexing, content/model fingerprints, tested software boundaries,
and a documented single-user container configuration.

Not established: production readiness, multi-user scalability, factual accuracy
rates for Groq, improved student learning, or measured cost/performance improvements.
Earlier local-model development measurements are not a held-out Groq evaluation.
