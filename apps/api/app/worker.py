from __future__ import annotations

import signal
import sys
import time
import os

from .database import init_db
from .diagnostics import record_job_event
from .job_execution import CeleryJobRunner, LocalJobRunner, RedisRQJobRunner, fetch_next_queued_job, get_job, job_cancel_requested

try:
    from celery import Celery
except Exception:  # pragma: no cover - dependency health reports this separately
    Celery = None

celery_app = Celery("ragbench", broker=os.environ.get("REDIS_URL", "redis://redis:6379/0"), backend=os.environ.get("REDIS_URL", "redis://redis:6379/0")) if Celery else None

RUNNERS = {
    "local": LocalJobRunner,
    "rq": RedisRQJobRunner,
    "celery": CeleryJobRunner,
}


def process_job(job: dict) -> None:
    if job_cancel_requested(job["id"]):
        return
    try:
        if job["job_type"] == "benchmark":
            from .main import execute_queued_benchmark

            execute_queued_benchmark(job)
        elif job["job_type"] == "source_upload":
            from .main import process_staged_sources

            payload = job["payload"]
            record_job_event(job["id"], job["project_id"], "source_upload_started", "running", "Queued source upload processing started.")
            print(
                f"ragbench worker processing source_upload job_id={job['id']} project_id={job['project_id']} sources={len(payload.get('source_ids', []))}",
                flush=True,
            )
            result = process_staged_sources(job["project_id"], payload.get("source_ids", []), payload.get("change_type", "append_sources"))
            record_job_event(job["id"], job["project_id"], "source_upload_complete", "succeeded", "Queued source upload processed.", result)
            print(f"ragbench worker finished source_upload job_id={job['id']}", flush=True)
        elif job["job_type"] == "model_pull":
            from .main import execute_model_pull_job

            print(
                f"ragbench worker processing model_pull job_id={job['id']} project_id={job['project_id']} models={job['payload'].get('models', [])}",
                flush=True,
            )
            execute_model_pull_job(job)
            print(f"ragbench worker finished model_pull job_id={job['id']}", flush=True)
        elif job["job_type"] == "ingestion":
            from .job_execution import create_ingestion_job

            payload = job["payload"]
            create_ingestion_job(job["project_id"], payload.get("source_ids", []), payload.get("fail_steps", []))
            record_job_event(job["id"], job["project_id"], "ingestion_complete", "succeeded", "Queued ingestion job completed.")
        else:
            record_job_event(job["id"], job["project_id"], "unknown_job_type", "failed", f"Unknown job type: {job['job_type']}")
    except Exception as exc:
        print(f"ragbench worker failed job_id={job['id']} type={job['job_type']} error={exc}", flush=True)
        record_job_event(job["id"], job["project_id"], "job_failed", "failed", str(exc), run_id=job.get("run_id"))


def process_job_by_id(job_id: str) -> None:
    job = get_job(job_id)
    record_job_event(job["id"], job["project_id"], "queue_worker_started", "running", "Queued job picked up by worker.", run_id=job.get("run_id"))
    process_job(get_job(job_id))


if celery_app:
    @celery_app.task(name="app.worker.process_job_by_id")
    def process_job_by_id_task(job_id: str) -> None:
        process_job_by_id(job_id)
else:
    class _MissingCeleryTask:
        id = "celery-unavailable"

        def delay(self, _: str) -> "_MissingCeleryTask":
            raise RuntimeError("Celery is not installed in the API container.")

    process_job_by_id_task = _MissingCeleryTask()


def main() -> None:
    init_db()
    mode = sys.argv[1] if len(sys.argv) > 1 else "local"
    runner_cls = RUNNERS.get(mode, LocalJobRunner)
    runner = runner_cls()
    running = True

    def stop(_: int, __: object) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    print(f"ragbench worker ready: {runner.contract_name}", flush=True)
    while running:
        job = fetch_next_queued_job()
        if job:
            record_job_event(job["id"], job["project_id"], "queue_worker_started", "running", "Queued job picked up by worker.", run_id=job.get("run_id"))
            print(f"ragbench worker picked job_id={job['id']} type={job['job_type']} project_id={job['project_id']}", flush=True)
            process_job(job)
            continue
        time.sleep(1)


if __name__ == "__main__":
    main()
