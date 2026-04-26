## Summary

## What Changed

## Verification

- [ ] API tests: `cd apps/api && pytest -q`
- [ ] Web tests: `cd apps/web && npm test -- --run`
- [ ] Web build: `cd apps/web && npm run build`
- [ ] Docker startup checked when infrastructure changed: `docker compose --profile vector up -d --build`

## Product Impact

State whether this affects uploads, extraction, embeddings, vector search, graph RAG, agentic RAG, citations, metrics, queue processing, dashboards, or model/runtime settings.

## Security And Privacy Impact

State whether this affects local source files, extracted content, embeddings, SQLite metadata, logs, traces, Ollama calls, host media processing, dashboard exposure, or export artifacts.

## Screenshots

## Branch Policy

- this PR targets `main` from a non-`main` feature branch
- required CI is expected to pass before merge
- review conversations are expected to be resolved before merge
- this PR is expected to land with squash merge
