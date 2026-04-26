from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Iterator

from .database import data_dir

SINGLE_DOCUMENT_INGESTION_TARGET_SECONDS = 60.0
TRADITIONAL_RAG_TARGET_SECONDS = 20.0
AGENTIC_OR_HYBRID_TARGET_SECONDS = 90.0
DEFAULT_TOP_K_CHUNKS = 8
DEFAULT_RERANKED_CHUNKS = 5
PROCESSING_STATES = ("processing", "completed", "failed", "waiting_for_local_models", "partial_success")


def disk_warning_threshold_bytes() -> int:
    return int(os.environ.get("RAGBENCH_DISK_WARNING_BYTES", str(1_000_000_000)))


def disk_status() -> dict[str, object]:
    usage = os.statvfs(data_dir())
    free_bytes = usage.f_bavail * usage.f_frsize
    threshold = disk_warning_threshold_bytes()
    return {
        "free_bytes": free_bytes,
        "warning_threshold_bytes": threshold,
        "warning": free_bytes < threshold,
    }


def reliability_status() -> dict[str, object]:
    return {
        "targets_seconds": {
            "single_document_ingestion": SINGLE_DOCUMENT_INGESTION_TARGET_SECONDS,
            "traditional_rag": TRADITIONAL_RAG_TARGET_SECONDS,
            "agentic_or_hybrid": AGENTIC_OR_HYBRID_TARGET_SECONDS,
        },
        "defaults": {"top_k_chunks": DEFAULT_TOP_K_CHUNKS, "reranked_chunks": DEFAULT_RERANKED_CHUNKS},
        "processing_states": list(PROCESSING_STATES),
        "disk": disk_status(),
    }


@contextmanager
def timing() -> Iterator[dict[str, float]]:
    started = time.perf_counter()
    data = {"elapsed_seconds": 0.0}
    try:
        yield data
    finally:
        data["elapsed_seconds"] = round(time.perf_counter() - started, 4)


def controlled_timeout_warning(provider: str, timeout_seconds: float) -> dict[str, object]:
    return {
        "provider": provider,
        "status": "waiting_for_local_models",
        "timeout_seconds": timeout_seconds,
        "progress_event": f"Waiting for {provider} container before timeout.",
    }
