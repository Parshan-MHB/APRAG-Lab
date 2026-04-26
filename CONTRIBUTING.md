# Contributing

APRAG-Lab is a local-first multimodal RAG benchmark lab. Contributions should preserve that direction: real local workflows, clear citations, reproducible tests, and minimal setup surprise for new users.

## Development Flow

1. Create a non-`main` branch.
2. Keep changes focused on one product, infrastructure, or documentation concern.
3. Run the relevant tests locally.
4. Open a pull request into `main`.
5. Wait for `aprag-lab-ci` to pass and resolve review conversations before merge.

Direct pushes to `main` are not part of the normal workflow after repository bootstrap.

## Local Verification

API:

```bash
cd apps/api
pytest -q
```

Web:

```bash
cd apps/web
npm test -- --run
npm run build
```

Docker startup:

```bash
docker compose --profile vector up -d --build
```

Use Docker for app services where possible. Keep Ollama and heavy media processing on the host unless a change explicitly targets container-runtime behavior.

## Testing Expectations

- Backend behavior should have focused API or unit coverage.
- Frontend workflows should have React/Vitest coverage for visible behavior and error states.
- RAG, citation, queue, graph, export, and observability changes should include tests that catch user-facing regressions.
- Infrastructure changes should document the profile or command used for validation.

## Privacy And Test Data

Do not commit private user documents, private prompts, extracted source content, embeddings, traces, logs, or generated local project data.

Use `sample-data/manual-test-suite` or generated synthetic data for tests and demos.

## Pull Request Notes

Use the pull request template and call out:

- product impact
- security or privacy impact
- tests run
- Docker/profile behavior if infrastructure changed
- screenshots for visible UI changes
