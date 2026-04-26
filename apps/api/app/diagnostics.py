from __future__ import annotations

import json
import os
import shutil
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .database import connect, data_dir

SECRET_KEYS = {"api_key", "apikey", "authorization", "password", "secret", "token"}


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def scrub(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            if key.lower() in SECRET_KEYS or any(secret in key.lower() for secret in SECRET_KEYS):
                cleaned[key] = "[redacted]"
            else:
                cleaned[key] = scrub(item)
        return cleaned
    if isinstance(value, list):
        return [scrub(item) for item in value]
    if isinstance(value, str):
        root = str(data_dir())
        text = value.replace(root, "$DATA_DIR")
        for marker in ("api_key=", "token=", "password=", "secret="):
            if marker in text.lower():
                return "[redacted]"
        return text
    return value


def local_log_path() -> Path:
    path = data_dir() / "logs" / "APRAG-Lab.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_local_log(event_type: str, payload: dict[str, Any]) -> None:
    entry = {"created_at": now_iso(), "event_type": event_type, "payload": scrub(payload)}
    with local_log_path().open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")


def read_local_logs(limit: int = 200) -> list[dict[str, Any]]:
    path = local_log_path()
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()[-max(1, min(limit, 1000)) :]
    entries: list[dict[str, Any]] = []
    for line in lines:
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            entries.append({"created_at": now_iso(), "event_type": "unparseable_log_line", "payload": {"line": line[:500]}})
    return entries


def create_job(project_id: str, run_id: str | None = None, status: str = "queued", job_type: str = "generic", payload: dict[str, Any] | None = None) -> str:
    job_id = str(uuid.uuid4())
    timestamp = now_iso()
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO jobs (id, project_id, run_id, job_type, payload_json, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (job_id, project_id, run_id, job_type, json.dumps(scrub(payload or {})), status, timestamp, timestamp),
        )
    write_local_log("job_created", {"job_id": job_id, "project_id": project_id, "run_id": run_id, "status": status})
    return job_id


def record_job_event(
    job_id: str,
    project_id: str,
    event_type: str,
    status: str,
    message: str,
    payload: dict[str, Any] | None = None,
    run_id: str | None = None,
) -> None:
    event_id = str(uuid.uuid4())
    timestamp = now_iso()
    clean_payload = scrub(payload or {})
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO job_events
              (id, job_id, run_id, project_id, event_type, status, message, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (event_id, job_id, run_id, project_id, event_type, status, message, json.dumps(clean_payload), timestamp),
        )
        conn.execute("UPDATE jobs SET status = ?, updated_at = ?, run_id = COALESCE(?, run_id) WHERE id = ?", (status, timestamp, run_id, job_id))
    write_local_log(event_type, {"job_id": job_id, "project_id": project_id, "run_id": run_id, "status": status, **clean_payload})


def _event_row_to_dict(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "job_id": row["job_id"],
        "run_id": row["run_id"],
        "project_id": row["project_id"],
        "event_type": row["event_type"],
        "status": row["status"],
        "message": row["message"],
        "payload": json.loads(row["payload_json"]),
        "created_at": row["created_at"],
    }


def list_job_events(run_id: str) -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM job_events
            WHERE run_id = ?
            ORDER BY created_at ASC
            """,
            (run_id,),
        ).fetchall()
    return [_event_row_to_dict(row) for row in rows]


def list_job_events_by_job(job_id: str) -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM job_events
            WHERE job_id = ?
            ORDER BY created_at ASC
            """,
            (job_id,),
        ).fetchall()
    return [_event_row_to_dict(row) for row in rows]


def trace_dir(project_id: str, run_id: str) -> Path:
    path = data_dir() / "projects" / project_id / "runs" / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_trace_jsonl(project_id: str, run_id: str, results: list[dict[str, Any]], recommendation: dict[str, Any]) -> Path:
    path = trace_dir(project_id, run_id) / "trace.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(scrub({"run_id": run_id, "event": "recommendation", "payload": recommendation}), sort_keys=True) + "\n")
        for result in results:
            for index, step in enumerate(result.get("trace", [])):
                handle.write(
                    json.dumps(
                        scrub(
                            {
                                "run_id": run_id,
                                "pipeline_type": result["pipeline_type"],
                                "event": "trace_step",
                                "index": index,
                                "payload": step,
                            }
                        ),
                        sort_keys=True,
                    )
                    + "\n"
                )
    write_local_log("trace_jsonl_written", {"project_id": project_id, "run_id": run_id, "path": str(path)})
    return path


def read_trace_jsonl(project_id: str, run_id: str) -> str:
    path = trace_dir(project_id, run_id) / "trace.jsonl"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def dependency_status() -> dict[str, Any]:
    from .provider_health import provider_health
    from .host_media import host_media_base_url, host_media_enabled
    from .providers import DEFAULT_EMBEDDING_MODEL, DEFAULT_OLLAMA_LLM_MODEL, DEFAULT_OLLAMA_VLM_MODEL

    ffmpeg = shutil.which("ffmpeg")
    tesseract = shutil.which("tesseract")
    ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    health = provider_health()
    ollama = health["ollama"]
    return {
        "python": {"provider": "container", "available": True, "version": sys.version.split()[0]},
        "node": {"provider": "web-container", "available": True, "version": "managed by Docker web image"},
        "ffmpeg": (
            {
                "provider": "host-media-runtime",
                "available": bool(health["media"]["ffmpeg"]["available"]),
                "url": host_media_base_url(),
                "path": health["media"]["ffmpeg"].get("path", ""),
                "warning": "" if health["media"]["ffmpeg"]["available"] else "Start the host media runtime with host FFmpeg on PATH.",
            }
            if host_media_enabled()
            else {
                "provider": "system-path",
                "available": bool(ffmpeg),
                "path": ffmpeg or "",
                "warning": "" if ffmpeg else "Video frame/audio extraction is unavailable until FFmpeg is installed in the container.",
            }
        ),
        "tesseract": (
            {
                "provider": "host-media-runtime",
                "available": bool(health["media"]["tesseract"]["available"]),
                "url": host_media_base_url(),
                "path": health["media"]["tesseract"].get("path", ""),
                "warning": "" if health["media"]["tesseract"]["available"] else "Start the host media runtime with host Tesseract on PATH.",
            }
            if host_media_enabled()
            else {
                "provider": "system-path",
                "available": bool(tesseract),
                "path": tesseract or "",
                "warning": "" if tesseract else "OCR is unavailable until Tesseract is installed in the container.",
            }
        ),
        "ollama": {
            "provider": "host-service",
            "available": bool(ollama.get("available")),
            "url": ollama_url,
            "warning": "" if ollama.get("available") else "Host Ollama is unavailable; install/start Ollama on the laptop and keep it listening on port 11434.",
            "models": ollama.get("models", {}),
        },
        "llm": {"provider": "ollama", "available": bool(ollama.get("models", {}).get(os.environ.get("APRAG_DEFAULT_OLLAMA_LLM", DEFAULT_OLLAMA_LLM_MODEL), {}).get("installed")), "mode": health["mode"]},
        "vlm": {"provider": "ollama_vlm", "available": bool(ollama.get("models", {}).get(os.environ.get("APRAG_DEFAULT_OLLAMA_VLM", DEFAULT_OLLAMA_VLM_MODEL), {}).get("installed")), "mode": health["mode"]},
        "embeddings": {
            "provider": "deterministic_lexical" if health["deterministic_fallback"] else "ollama_embeddings",
            "available": True if health["deterministic_fallback"] else bool(ollama.get("models", {}).get(os.environ.get("APRAG_DEFAULT_EMBEDDINGS", DEFAULT_EMBEDDING_MODEL), {}).get("installed")),
            "mode": health["mode"],
        },
        "transcription": health["media"].get("transcription", health["media"]["faster_whisper"]),
        "telemetry": {"provider": "none", "available": False, "enabled": False, "warning": "Telemetry is disabled by default."},
        "cloud_providers": {"provider": "none", "available": False, "enabled": False, "warning": "Paid/cloud providers are disabled by default."},
        "disk": {
            "provider": "local-volume",
            "available": True,
            "free_bytes": shutil.disk_usage(data_dir()).free,
        },
    }
