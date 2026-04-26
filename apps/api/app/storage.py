from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .database import data_dir
from .diagnostics import scrub


ARTIFACT_LAYOUT = {
    "project_manifest": "project.json",
    "original_sources": "sources/original",
    "derived_ocr": "sources/derived/ocr",
    "derived_transcripts": "sources/derived/transcripts",
    "derived_captions": "sources/derived/captions",
    "derived_frames": "sources/derived/frames",
    "runs": "runs",
    "exports": "exports",
    "sqlite": "sqlite/APRAG-Lab.db",
    "vector_store": "vector_store/chroma",
}


def ensure_storage_layout(project_id: str | None = None) -> dict[str, str]:
    root = data_dir()
    (root / "vector_store" / "chroma").mkdir(parents=True, exist_ok=True)
    if project_id:
        project_root = root / "projects" / project_id
        for relative in [
            "sources/original",
            "sources/derived/ocr",
            "sources/derived/transcripts",
            "sources/derived/captions",
            "sources/derived/frames",
            "runs",
            "exports",
        ]:
            (project_root / relative).mkdir(parents=True, exist_ok=True)
        return {key: str((project_root / value).relative_to(root)) for key, value in ARTIFACT_LAYOUT.items() if not value.startswith(("sqlite", "vector_store"))}
    return {key: value for key, value in ARTIFACT_LAYOUT.items()}


def relative_to_data_root(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(data_dir()))
    except ValueError:
        return str(path)


def write_project_manifest(project: dict[str, Any]) -> Path:
    ensure_storage_layout(project["id"])
    path = data_dir() / "projects" / project["id"] / "project.json"
    payload = scrub({"schema": "APRAG-Lab.project.v1", "project": project, "artifact_layout": ensure_storage_layout(project["id"])})
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_run_artifacts(run: dict[str, Any]) -> dict[str, str]:
    ensure_storage_layout(run["project_id"])
    run_dir = data_dir() / "projects" / run["project_id"] / "runs" / run["id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for result in run["results"]:
        filename = f"{result['pipeline_type']}.json"
        path = run_dir / filename
        path.write_text(json.dumps(scrub(result), indent=2, sort_keys=True), encoding="utf-8")
        paths[result["pipeline_type"]] = relative_to_data_root(path)
    comparison_path = run_dir / "comparison.json"
    comparison_path.write_text(json.dumps(scrub(run["comparison"]), indent=2, sort_keys=True), encoding="utf-8")
    paths["comparison"] = relative_to_data_root(comparison_path)
    return paths


def parse_trace_jsonl(text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at line {line_number}") from exc
        if not isinstance(event, dict) or "event" not in event or "run_id" not in event:
            raise ValueError(f"Trace event line {line_number} is missing required fields")
        events.append(event)
    return events
