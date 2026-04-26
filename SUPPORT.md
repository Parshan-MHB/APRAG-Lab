# Support

Use GitHub issues for reproducible product defects and concrete feature tasks.

Before opening an issue, include the details that make APRAG-Lab problems diagnosable:

- OS and Docker Desktop version
- Docker memory allocation
- Ollama version and installed models
- selected APRAG-Lab model profile
- whether the host media runtime is running
- source file type and approximate size
- API logs, browser error text, run ID, or trace/dashboard link when available

Security-sensitive reports should follow [SECURITY.md](SECURITY.md) instead of a public issue.

General local checks:

```bash
curl http://localhost:8000/health
curl http://localhost:8765/health
curl http://localhost:11434/api/tags
```
