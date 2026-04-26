# Architecture Notes

The API, web app, queue worker, SQLite, Chroma, and Qdrant run in Docker.
Ollama, OCR, FFmpeg media extraction, and transcription run on the host laptop.
Ollama runs on the host laptop at http://host.docker.internal:11434.
SQLite is the required metadata store for this product scope.
The graph layer stores entities such as API, Web, Worker, Chroma, Qdrant, SQLite, and Ollama.
Hybrid Graph RAG should connect retrieved chunks with graph relationships before answering.
