from __future__ import annotations

import importlib.util
import os
import shutil
import socket
import sys
import urllib.error
import urllib.request
from typing import Any

from .providers import DEFAULT_EMBEDDING_MODEL, DEFAULT_OLLAMA_LLM_MODEL, DEFAULT_OLLAMA_VLM_MODEL, http_json, ollama_model_installed
from .resource_profile import preflight_report
from .host_media import HostMediaClient, host_media_base_url, host_media_enabled


REQUIRED_OLLAMA_MODELS = [DEFAULT_OLLAMA_LLM_MODEL, DEFAULT_OLLAMA_VLM_MODEL, DEFAULT_EMBEDDING_MODEL]


def provider_mode() -> str:
    return os.environ.get("RAGBENCH_PROVIDER_MODE", "real").strip().lower()


def deterministic_mode() -> bool:
    return provider_mode() == "deterministic"


def remediation(model: str) -> str:
    return f"ollama pull {model}"


def _tcp_check(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _http_available(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return 200 <= response.status < 500
    except (urllib.error.URLError, TimeoutError):
        return False


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def ollama_health() -> dict[str, Any]:
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434").rstrip("/")
    status: dict[str, Any] = {
        "provider": "ollama",
        "runtime": "host",
        "base_url": base_url,
        "available": False,
        "models": {},
        "mode": provider_mode(),
        "remediation": "Install Ollama on the host laptop, start it, then run the listed ollama pull commands.",
    }
    try:
        tags = http_json(f"{base_url}/api/tags", None, timeout=3.0)
    except Exception as exc:
        status["warning"] = f"Ollama service is unavailable: {exc}"
        return status
    installed = {model.get("name") for model in tags.get("models", [])}
    status["available"] = True
    for model in REQUIRED_OLLAMA_MODELS:
        model_ready = ollama_model_installed(model, installed)
        status["models"][model] = {
            "installed": model_ready,
            "remediation": "" if model_ready else remediation(model),
        }
    status["ready"] = all(item["installed"] for item in status["models"].values())
    return status


def media_health() -> dict[str, Any]:
    if host_media_enabled():
        host = HostMediaClient().health()
        host_ready = bool(host.get("ok"))
        ffmpeg = host.get("ffmpeg", {})
        ffprobe = host.get("ffprobe", {})
        tesseract = host.get("tesseract", {})
        faster = host.get("faster_whisper", {})
        whisper_cpp = host.get("whisper_cpp", {})
        transcription_available = bool(faster.get("available") or whisper_cpp.get("available"))
        return {
            "runtime": "host",
            "host_media_runtime": {
                "provider": "host-media-runtime",
                "available": host_ready,
                "url": host_media_base_url(),
                "data_dir": host.get("data_dir", ""),
                "warning": "" if host_ready else host.get("error", "Host media runtime is unavailable."),
                "setup": "Run: RAGBENCH_HOST_DATA_DIR=$PWD/data python3 scripts/host_media_runtime.py",
            },
            "ffmpeg": {
                "provider": "ffmpeg",
                "available": bool(host_ready and ffmpeg.get("available")),
                "path": ffmpeg.get("path", ""),
                "runtime": "host",
            },
            "ffprobe": {
                "provider": "ffprobe",
                "available": bool(host_ready and ffprobe.get("available")),
                "path": ffprobe.get("path", ""),
                "runtime": "host",
            },
            "tesseract": {
                "provider": "tesseract",
                "available": bool(host_ready and tesseract.get("available")),
                "path": tesseract.get("path", ""),
                "runtime": "host",
            },
            "easyocr": {
                "provider": "easyocr",
                "available": False,
                "runtime": "host",
                "warning": "EasyOCR is not part of the single supported host-media runtime.",
            },
            "faster_whisper": {
                "provider": "faster-whisper",
                "available": bool(host_ready and faster.get("available")),
                "runtime": "host",
                "warning": faster.get("warning", ""),
            },
            "whisper_cpp": {
                "provider": "whisper.cpp",
                "available": bool(host_ready and whisper_cpp.get("available")),
                "runtime": "host",
            },
            "transcription": {
                "provider": "faster-whisper-or-whisper.cpp",
                "available": bool(host_ready and transcription_available),
                "runtime": "host",
            },
        }
    return {
        "runtime": "container",
        "ffmpeg": {
            "provider": "ffmpeg",
            "available": shutil.which("ffmpeg") is not None,
            "path": shutil.which("ffmpeg") or "",
            "container": "api/media-tools",
        },
        "tesseract": {
            "provider": "tesseract",
            "available": shutil.which("tesseract") is not None,
            "path": shutil.which("tesseract") or "",
            "container": "api/media-tools",
        },
        "easyocr": {
            "provider": "easyocr",
            "available": _module_available("easyocr"),
            "container": "custom api image only",
            "warning": "" if _module_available("easyocr") else "EasyOCR is optional secondary OCR and is not installed in the slim default image.",
        },
        "faster_whisper": {
            "provider": "faster-whisper",
            "available": _module_available("faster_whisper"),
            "container": "api/media-tools",
        },
        "whisper_cpp": {
            "provider": "whisper.cpp",
            "available": shutil.which("whisper-cli") is not None or shutil.which("main") is not None,
            "container": "custom api image only",
        },
    }


def vector_health() -> dict[str, Any]:
    chroma_host = os.environ.get("CHROMA_HOST", "chroma")
    chroma_port = int(os.environ.get("CHROMA_PORT", "8000"))
    qdrant_url = os.environ.get("QDRANT_URL", "http://qdrant:6333").rstrip("/")
    return {
        "chroma": {
            "provider": "chroma",
            "available": _tcp_check(chroma_host, chroma_port) or _module_available("chromadb"),
            "host": chroma_host,
            "port": chroma_port,
            "container": "chroma",
        },
        "qdrant": {
            "provider": "qdrant",
            "available": _http_available(f"{qdrant_url}/collections") or _module_available("qdrant_client"),
            "url": qdrant_url,
            "container": "qdrant",
        },
    }


def queue_health() -> dict[str, Any]:
    redis_url = os.environ.get("REDIS_URL", "redis://redis:6379/0")
    host = redis_url.split("@")[-1].split(":")[0].replace("redis://", "")
    return {
        "runtime_boundary": {
            "provider": "queue-orchestration",
            "runtime": "docker",
            "heavy_processing": False,
            "decision": "Keep queue workers in Docker; model inference, embeddings, OCR, media extraction, and transcription run on the host.",
        },
        "local_worker": {
            "provider": "sqlite-local-worker",
            "available": True,
            "mode": os.environ.get("RAGBENCH_QUEUE_MODE", "local"),
            "heavy_processing": False,
        },
        "redis": {"provider": "redis", "available": _tcp_check(host, 6379), "url": redis_url, "heavy_processing": False},
        "rq": {"provider": "rq", "available": _module_available("rq"), "container": "rq-worker", "heavy_processing": False},
        "celery": {"provider": "celery", "available": _module_available("celery"), "container": "celery-worker", "heavy_processing": False},
    }


def storage_health() -> dict[str, Any]:
    from .database import data_dir

    disk = shutil.disk_usage(data_dir())
    return {"data_dir": str(data_dir()), "available": True, "free_bytes": disk.free}


def provider_health() -> dict[str, Any]:
    mode = provider_mode()
    resource_preflight = preflight_report()
    health = {
        "mode": mode,
        "real_mode": mode == "real",
        "deterministic_fallback": mode == "deterministic",
        "resource_preflight": resource_preflight,
        "ollama": ollama_health(),
        "media": media_health(),
        "vector_stores": vector_health(),
        "queues": queue_health(),
        "storage": storage_health(),
    }
    if mode == "real":
        missing = []
        if not resource_preflight["safe_to_pull"]:
            missing.append("resource_preflight")
        ollama = health["ollama"]
        if not ollama.get("available"):
            missing.append("ollama")
        missing.extend(model for model, info in ollama.get("models", {}).items() if not info.get("installed"))
        required_media = ["ffmpeg", "tesseract"]
        if health["media"].get("runtime") != "host":
            required_media.append("faster_whisper")
        for required in required_media:
            if not health["media"][required]["available"]:
                missing.append(required)
        if health["media"].get("runtime") == "host" and not health["media"].get("transcription", {}).get("available"):
            missing.append("host_transcription")
        health["required_missing"] = missing
        health["ready"] = not missing
    else:
        health["required_missing"] = []
        health["ready"] = True
    return health


def main() -> None:
    import json

    report = media_health() if "--media-only" in sys.argv else provider_health()
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
