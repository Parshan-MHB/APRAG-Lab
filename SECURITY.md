# Security Policy

## Supported Scope

Security reports are most helpful when they affect:

- local uploaded files or extracted source content
- embeddings, vector database contents, or SQLite metadata
- RAG answers, citations, exports, logs, traces, or dashboard exposure
- Ollama model calls, model configuration, or prompt handling
- host media runtime boundaries for OCR, audio, video, and transcription
- Docker service exposure, queue processing, or API endpoints
- privacy boundaries between local-only processing and optional external observability

## Reporting

Please do not open a public issue for a suspected security vulnerability before maintainers have had a chance to assess it.

Use one of these paths:

1. GitHub security advisory reporting for this repository, if available.
2. A private report to the repository owner.

Include:

- affected commit or version
- reproduction steps
- impact
- whether private source files, extracted content, embeddings, logs, or traces may be exposed
- any proposed containment or fix

## Disclosure

- reports will be triaged privately first
- public discussion should wait until impact and containment are understood
- fixes should avoid exposing sensitive exploit details before users can update
- security-sensitive test data should not include private user documents
