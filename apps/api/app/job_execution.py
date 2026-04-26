from __future__ import annotations

import json
import os
from typing import Any, Callable, Protocol

from .database import connect
from .diagnostics import create_job, list_job_events_by_job, record_job_event, scrub

JOB_STATES = ("queued", "running", "succeeded", "failed", "cancelled", "partial_success")

INGESTION_STEPS = (
    "validate_file",
    "extract_content",
    "create_content_blocks",
    "chunk_content",
    "embed_chunks",
    "extract_entities",
    "update_graph",
    "finalize_version",
)

BENCHMARK_STEPS = (
    "create_run",
    "start_traditional_pipeline",
    "start_agentic_pipeline",
    "start_hybrid_graph_pipeline",
    "collect_results",
    "compute_comparison",
    "persist_outputs",
)

TERMINAL_STATES = {"succeeded", "failed", "cancelled", "partial_success"}


class JobRunnerContract(Protocol):
    contract_name: str

    def enqueue(self, project_id: str, job_type: str, work: Callable[[], dict[str, Any]] | None = None) -> dict[str, Any]:
        ...

    def cancel(self, job_id: str) -> dict[str, Any]:
        ...

    def status(self, job_id: str) -> dict[str, Any]:
        ...


class LocalJobRunner:
    contract_name = "local_background_tasks"

    def enqueue(self, project_id: str, job_type: str, work: Callable[[], dict[str, Any]] | None = None) -> dict[str, Any]:
        job_id = create_job(project_id, status="queued", job_type=job_type)
        if work is None:
            record_job_event(job_id, project_id, "queued", "queued", f"{job_type} job queued.")
            return get_job(job_id)
        record_job_event(job_id, project_id, "start_job", "running", f"{job_type} job started.")
        try:
            payload = work()
        except Exception as exc:
            record_job_event(job_id, project_id, "job_failed", "failed", str(exc))
            return get_job(job_id)
        record_job_event(job_id, project_id, "job_succeeded", "succeeded", f"{job_type} job finished.", payload)
        return get_job(job_id)

    def cancel(self, job_id: str) -> dict[str, Any]:
        return cancel_job(job_id)

    def status(self, job_id: str) -> dict[str, Any]:
        return get_job(job_id)


class RedisRQJobRunner(LocalJobRunner):
    contract_name = "redis_rq"

    def enqueue(self, project_id: str, job_type: str, work: Callable[[], dict[str, Any]] | None = None) -> dict[str, Any]:
        job_id = create_job(project_id, status="queued", job_type=job_type)
        record_job_event(job_id, project_id, "rq_queued", "queued", f"{job_type} job queued in Redis/RQ.", {"queue": "ragbench"})
        try:
            from redis import Redis
            from rq import Queue

            connection = Redis.from_url(os.environ.get("REDIS_URL", "redis://redis:6379/0"))
            rq_job = Queue("ragbench", connection=connection).enqueue("app.worker.process_job_by_id", job_id)
            job = get_job(job_id)
            job["queue_backend"] = self.contract_name
            job["rq_job_id"] = rq_job.id
            return job
        except Exception as exc:
            record_job_event(job_id, project_id, "rq_enqueue_failed", "failed", str(exc))
            return get_job(job_id)


class CeleryJobRunner(LocalJobRunner):
    contract_name = "celery"

    def enqueue(self, project_id: str, job_type: str, work: Callable[[], dict[str, Any]] | None = None) -> dict[str, Any]:
        job_id = create_job(project_id, status="queued", job_type=job_type)
        record_job_event(job_id, project_id, "celery_queued", "queued", f"{job_type} job queued in Celery.", {"queue": "ragbench"})
        try:
            from .worker import process_job_by_id_task

            celery_result = process_job_by_id_task.delay(job_id)
            job = get_job(job_id)
            job["queue_backend"] = self.contract_name
            job["celery_task_id"] = celery_result.id
            return job
        except Exception as exc:
            record_job_event(job_id, project_id, "celery_enqueue_failed", "failed", str(exc))
            return get_job(job_id)


def job_row_to_dict(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "project_id": row["project_id"],
        "run_id": row["run_id"],
        "job_type": row["job_type"],
        "payload": json.loads(row["payload_json"]),
        "cancellation_requested": bool(row["cancellation_requested"]),
        "status": row["status"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def get_job(job_id: str) -> dict[str, Any]:
    with connect() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not row:
        raise KeyError(job_id)
    job = job_row_to_dict(row)
    job["events"] = list_job_events_by_job(job_id)
    return job


def get_job_events(job_id: str) -> list[dict[str, Any]]:
    get_job(job_id)
    return list_job_events_by_job(job_id)


def create_ingestion_job(project_id: str, source_ids: list[str] | None = None, fail_steps: list[str] | None = None) -> dict[str, Any]:
    job_id = create_job(project_id, status="queued", job_type="ingestion", payload={"source_ids": source_ids or [], "fail_steps": fail_steps or []})
    source_ids = source_ids or []
    fail_steps = fail_steps or []
    for step in INGESTION_STEPS:
        status = "running"
        message = f"Ingestion step {step} completed."
        if step in fail_steps:
            status = "partial_success"
            message = f"Ingestion step {step} completed with recoverable failures."
        if step == INGESTION_STEPS[-1]:
            status = "partial_success" if fail_steps else "succeeded"
            message = "Ingestion job completed." if status == "succeeded" else "Ingestion job completed with partial success."
        record_job_event(job_id, project_id, step, status, message, {"source_ids": source_ids})
    return get_job(job_id)


def cancel_job(job_id: str) -> dict[str, Any]:
    job = get_job(job_id)
    if job["status"] == "cancelled":
        return job
    with connect() as conn:
        conn.execute(
            "UPDATE jobs SET status = ?, cancellation_requested = 1, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            ("cancelled", job_id),
        )
    record_job_event(job_id, job["project_id"], "cancel_job", "cancelled", "Job cancellation requested.", run_id=job["run_id"])
    return get_job(job_id)


def cancel_run(run_id: str) -> dict[str, Any]:
    with connect() as conn:
        row = conn.execute("SELECT id FROM jobs WHERE run_id = ? ORDER BY created_at DESC LIMIT 1", (run_id,)).fetchone()
    if not row:
        raise KeyError(run_id)
    return cancel_job(row["id"])


def sse_lines(events: list[dict[str, Any]]) -> str:
    return "".join(f"event: {event['event_type']}\ndata: {json.dumps(scrub(event), sort_keys=True)}\n\n" for event in events)


def failure_pipeline_result(pipeline_type: str, error: Exception) -> dict[str, Any]:
    return {
        "pipeline_type": pipeline_type,
        "answer": "I could not complete this pipeline, but other pipeline results are still available.",
        "citations": [],
        "trace": [{"step": "pipeline_failed", "detail": str(error)}],
        "metrics": {
            "latency_ms": 0,
            "retrieved_chunks": 0,
            "citation_count": 0,
            "evidence_score": 0.0,
            "insufficient_evidence": True,
            "pipeline_failed": True,
        },
        "techniques": [pipeline_type, "partial_failure_recovery"],
        "warnings": ["pipeline_failed", str(error)],
    }


def create_queued_job(project_id: str, job_type: str, payload: dict[str, Any], run_id: str | None = None) -> dict[str, Any]:
    job_id = create_job(project_id, run_id=run_id, status="queued", job_type=job_type, payload=payload)
    record_job_event(job_id, project_id, "queued", "queued", f"{job_type} job queued.", payload, run_id)
    return get_job(job_id)


def fetch_next_queued_job() -> dict[str, Any] | None:
    with connect() as conn:
        row = conn.execute(
            """
            SELECT * FROM jobs
            WHERE status = 'queued'
            ORDER BY created_at ASC
            LIMIT 1
            """
        ).fetchone()
        if not row:
            return None
        conn.execute("UPDATE jobs SET status = 'running', updated_at = CURRENT_TIMESTAMP WHERE id = ?", (row["id"],))
    return get_job(row["id"])


def job_cancel_requested(job_id: str) -> bool:
    with connect() as conn:
        row = conn.execute("SELECT cancellation_requested, status FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return bool(row and (row["cancellation_requested"] or row["status"] == "cancelled"))
