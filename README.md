# APRAG-Lab

APRAG-Lab is a local-first multimodal RAG benchmark workspace. It lets you upload documents, images, audio, and video, build a searchable knowledge base, run multiple RAG strategies against the same evidence, and compare the answers with citations, traces, graph signals, queue progress, and grounding notes.

The product is designed for teams that need to evaluate whether a RAG system is really grounded in source material before using it in production. It solves the repeated setup problem by keeping the app, queue, metadata store, and vector databases in Docker, while running heavy model and media processing on the host laptop where Ollama, FFmpeg, OCR, and transcription can use the machine's native resources.

## What It Does

- Upload TXT, Markdown, CSV, JSON, PDF, DOCX, image, audio, and video files.
- Extract text, tables, OCR text, video frames, audio, and transcripts.
- Store metadata and graph state in SQLite.
- Store embeddings in Chroma or Qdrant.
- Use host Ollama for real LLM, VLM, and embedding models.
- Use host media tooling for OCR, FFmpeg extraction, and transcription.
- Run Traditional RAG, Agentic RAG, and Hybrid Graph RAG for the same question.
- Compare quality, latency, citations, retrieval coverage, graph usage, and grounding.
- Show upload and benchmark queue progress in the frontend.
- Let users inspect source chunks, resolved citations, run traces, and local database dashboards.
- Publish benchmark spans to OpenTelemetry/Jaeger and Phoenix, with optional LangSmith tracing.

## Architecture

Docker runs the product shell:

- `web`: React/Vite frontend at `http://localhost:5173`
- `api`: FastAPI backend at `http://localhost:8000`
- `worker`: local benchmark and ingestion worker
- `redis`: queue and job coordination
- `redis-commander`: Redis dashboard
- `chroma`: vector database, enabled with the `vector` profile
- `qdrant`: vector database, enabled with the `vector` profile
- `jaeger`: OpenTelemetry trace dashboard
- `phoenix`: AI/RAG observability dashboard

The host laptop runs heavy processing:

- Ollama for LLM, VLM, and embedding inference
- FFmpeg and ffprobe for audio/video handling
- Tesseract for OCR
- faster-whisper or whisper.cpp for transcription

This split is intentional. Containers make the app reproducible, while local host inference avoids Docker memory limits and makes real benchmarking practical on a laptop.

## Requirements

- Docker Desktop with Docker Compose.
- Ollama installed and running on the host.
- Python 3.11 or newer for the host media runtime virtual environment.
- FFmpeg and ffprobe on the host `PATH`.
- Tesseract on the host `PATH`.
- At least 16 GB Docker Desktop memory allocation is recommended for comfortable local use.

On macOS, the host tools can be installed with Homebrew:

```bash
brew install ffmpeg tesseract ollama
```

Start Ollama:

```bash
ollama serve
```

If Ollama is installed as a desktop app, opening the app is usually enough.

## Model Profiles

The default profile is `lite`, which is the recommended first run:

- LLM: `qwen3:8b`
- VLM: `qwen3-vl:4b`
- Embeddings: `bge-m3`
- Transcription: `faster-whisper:large-v3-turbo`

Stronger profiles are available when the host has enough resources:

- `standard`: `qwen3:14b`, `qwen3-vl:8b`, `bge-m3`
- `high_quality`: `qwen3:32b`, `qwen3-vl:8b`, `bge-m3`

You can pull models before launch:

```bash
ollama pull qwen3:8b
ollama pull qwen3-vl:4b
ollama pull bge-m3
```

You can also start the app first and use Settings -> Pull missing models. The backend queues model pulls through the worker and shows progress in the UI.

## Setup From Scratch

From the project folder:

```bash
cd APRAG-Lab
```

Create the host media runtime virtual environment:

```bash
./scripts/setup_host_media_runtime.sh
```

Start the host media runtime in a separate terminal:

```bash
APRAG_HOST_DATA_DIR=$PWD/data .venv/host-media/bin/python scripts/host_media_runtime.py --host 0.0.0.0
```

Start the product containers:

```bash
docker compose --profile vector up -d --build
```

Open the app:

```text
http://localhost:5173
```

The first load creates a local benchmark project automatically.

## How To Use The App

1. Open `http://localhost:5173`.
2. In Upload And Processing, choose source files.
3. Choose the upload behavior:
   - Add to current knowledge base
   - Clear current data and start fresh
   - Create a new knowledge base
4. Click Upload and watch the extraction queue progress.
5. Use Project Sources to select or inspect files.
6. Use Source Viewer to read extracted chunks and source content.
7. In Benchmark Run, ask a question.
8. Choose Independent or Follow-up mode.
9. Click Run all RAG flows.
10. Watch the RAG Flow Queue until Traditional, Agentic, and Hybrid Graph runs finish.
11. Review the Comparison tab, individual flow tabs, citations, trace data, graph usage, and grounding notes.
12. Use Settings to inspect provider health, model overrides, missing model pulls, resource profile, database dashboards, and monitoring dashboards.

Run exports are available from the API:

```text
http://localhost:8000/api/runs/{run_id}/export.json
http://localhost:8000/api/runs/{run_id}/export.md
```

## Local URLs

- Frontend: `http://localhost:5173`
- API health: `http://localhost:8000/health`
- API docs: `http://localhost:8000/docs`
- SQLite query console: `http://localhost:8000/api/database/sqlite-dashboard`
- Chroma API docs: `http://localhost:8001/docs`
- Qdrant dashboard: `http://localhost:6333/dashboard`
- Redis Commander: `http://localhost:8083`
- Jaeger traces: `http://localhost:16686`
- Phoenix AI observability: `http://localhost:6006`
- LangSmith project: `https://smith.langchain.com` when `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` are set

The database and monitoring dashboard buttons are also available in the frontend Settings panel.

## Monitoring And Logs

APRAG-Lab records three levels of benchmark observability:

- In-app traceability: run queue events, per-pipeline Trace Viewer steps, citation resolution, metrics, graph state snapshots, and JSONL trace exports.
- Local observability: OpenTelemetry spans are exported to Jaeger and Phoenix for API requests, benchmark pipeline runs, retrieval, fusion RAG, LLM calls, VLM calls, and embedding calls.
- Optional external AI observability: LangSmith tracing can be enabled for teams that already use LangGraph/LangChain monitoring.

Local dashboards:

```text
http://localhost:16686   Jaeger OpenTelemetry traces
http://localhost:6006    Phoenix AI/RAG traces
```

Raw local logs and run trace endpoints:

```text
http://localhost:8000/api/diagnostics/logs?limit=200
http://localhost:8000/api/runs/{run_id}/trace.jsonl
http://localhost:8000/api/runs/{run_id}/trace-events
http://localhost:8000/api/runs/{run_id}/graph-states
```

To enable LangSmith, set these environment variables before starting Docker Compose:

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=your_langsmith_key
export LANGSMITH_PROJECT=APRAG-Lab
```

## Sample Data

Realistic manual test files are in:

```text
sample-data/manual-test-suite
```

The sample is a connected Northstar field-service case study. It uses one real-world image, one audio memo, one CSV, and one PDF to prove the multimodal path while keeping the scenario small enough for repeatable manual testing. The files share customers, ticket IDs, owners, firmware versions, incident metrics, SLA decisions, and mitigation actions.

Files:

- `01_vaccine_administration_event.jpg`: real-world clinical vaccine administration photo with no added text overlay, sourced from Wikimedia Commons / NIH public-domain media.
- `02_lakeside_dispatch_memo.wav`: spoken dispatch memo for INC-1043 and Priya Shah's field action.
- `03_service_tickets.csv`: ticket metrics, severity, downtime, inventory risk, owners, and SLA flags.
- `04_incident_review.pdf`: incident narrative, root cause, rollback decision, rollout plan, and customer outcomes.

Useful benchmark questions:

- What caused the Lakeside Clinic outage, and which CSV ticket metrics prove it was the highest-risk case?
- What does the vaccination image show, and how does it relate to the Lakeside vaccine freezer incident?
- What did the audio dispatch memo say Priya Shah did for INC-1043?
- Which customers were on firmware 4.8.2, and why did only Lakeside qualify for an SLA credit?
- What did the review board decide about rollback versus staged firmware 4.8.3 rollout?
- Was Pine Ridge Foods part of the firmware defect, or was it a different issue?

Regenerate the sample files when needed:

```bash
python3 scripts/create_manual_test_dataset.py
```

Supported upload extensions:

- Text and documents: `.txt`, `.md`, `.markdown`, `.pdf`, `.docx`
- Structured data: `.csv`, `.json`
- Images: `.png`, `.jpg`, `.jpeg`, `.webp`, `.gif`
- Audio: `.wav`, `.mp3`, `.m4a`, `.ogg`
- Video: `.mp4`, `.mov`, `.mkv`, `.webm`

## Configuration

Common environment variables:

- `APRAG_MODEL_PROFILE=auto|lite|standard|high_quality`
- `APRAG_PROVIDER_MODE=real|deterministic`
- `APRAG_VECTOR_STORE=chroma|qdrant|sqlite`
- `APRAG_MEDIA_RUNTIME=host|container`
- `OLLAMA_BASE_URL=http://host.docker.internal:11434`
- `APRAG_HOST_MEDIA_BASE_URL=http://host.docker.internal:8765`
- `APRAG_DEFAULT_OLLAMA_LLM=qwen3:8b`
- `APRAG_DEFAULT_OLLAMA_VLM=qwen3-vl:4b`
- `APRAG_DEFAULT_EMBEDDINGS=bge-m3`
- `APRAG_DEFAULT_TRANSCRIPTION=faster-whisper:large-v3-turbo`
- `APRAG_OBSERVABILITY_ENABLED=true|false`
- `APRAG_OTEL_EXPORTER_OTLP_ENDPOINTS=http://jaeger:4318/v1/traces,http://phoenix:6006/v1/traces`
- `LANGSMITH_TRACING=true|false`
- `LANGSMITH_API_KEY=...`
- `LANGSMITH_PROJECT=APRAG-Lab`

For normal local use, keep the defaults in `docker-compose.yml`.

## Verification

Backend tests:

```bash
docker compose exec -T api pytest -q
```

Frontend tests:

```bash
docker compose exec -T web npm test -- --run
```

Frontend production build:

```bash
docker compose exec -T web npm run build
```

Provider health:

```bash
curl http://localhost:8000/api/settings/provider-health
curl http://localhost:8000/api/settings/resource-profile
curl http://localhost:8000/api/settings/models
```

Logs:

```bash
curl "http://localhost:8000/api/diagnostics/logs?limit=200"
tail -f data/logs/APRAG-Lab.log
```

Browser-level manual testing can be started with:

```bash
cd apps/web
APRAG_PLAYWRIGHT_HEADLESS=0 node scripts/manual-product-test.cjs
```

## Stop And Clean Up

Stop containers:

```bash
docker compose down
```

Stop the host media runtime with `Ctrl+C` in its terminal.

Persistent local product data is stored under:

```text
data/
```

Delete that folder only when you intentionally want to remove local projects, sources, extracted chunks, runs, logs, and SQLite metadata.

## Troubleshooting

If the app says Ollama is unreachable:

```bash
curl http://localhost:11434/api/tags
```

If that fails, start Ollama and reload the app.

If models are missing, use Settings -> Pull missing models or run:

```bash
ollama pull qwen3:8b
ollama pull qwen3-vl:4b
ollama pull bge-m3
```

If media extraction fails, make sure the host media runtime is running and the tools exist:

```bash
ffmpeg -version
ffprobe -version
tesseract --version
curl http://localhost:8765/health
```

If vector dashboards do not open, make sure the app was started with:

```bash
docker compose --profile vector up -d --build
```

If Docker runs out of memory, keep Ollama on the host, use the `lite` model profile, close other heavy apps, and keep Docker Desktop memory near its available maximum.

## Privacy

APRAG-Lab is local-first. By default, source files, extracted content, embeddings, model calls, traces, and benchmark results stay on the local machine. The default product path does not require a cloud LLM provider.
