from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from .database import connect, data_dir


def resolve_citation(chunk_id: str | None = None, label: str | None = None, run_id: str | None = None) -> dict[str, Any] | None:
    if not chunk_id and not label:
        return None
    with connect() as conn:
        row = None
        if chunk_id:
            row = conn.execute(
                """
                SELECT c.*, cb.block_type, cb.text AS block_text, cb.frame_path, cb.image_region, cb.confidence,
                       cb.metadata_json AS block_metadata_json, s.filename, s.source_type, s.local_path
                FROM chunks c
                JOIN content_blocks cb ON cb.id = c.content_block_id
                JOIN sources s ON s.id = c.source_id
                WHERE c.id = ?
                """,
                (chunk_id,),
            ).fetchone()
        if not row and label:
            row = conn.execute(
                """
                SELECT c.*, cb.block_type, cb.text AS block_text, cb.frame_path, cb.image_region, cb.confidence,
                       cb.metadata_json AS block_metadata_json, s.filename, s.source_type, s.local_path
                FROM chunks c
                JOIN content_blocks cb ON cb.id = c.content_block_id
                JOIN sources s ON s.id = c.source_id
                WHERE c.citation = ? OR c.source_reference_label = ?
                """,
                (label, label),
            ).fetchone()
        if not row:
            snapshot = None
            if run_id and (chunk_id or label):
                if chunk_id:
                    snapshot = conn.execute(
                        """
                        SELECT resolution_json FROM run_evidence_snapshots
                        WHERE run_id = ? AND chunk_id = ?
                        ORDER BY created_at DESC LIMIT 1
                        """,
                        (run_id, chunk_id),
                    ).fetchone()
                if not snapshot and label:
                    snapshot = conn.execute(
                        """
                        SELECT resolution_json FROM run_evidence_snapshots
                        WHERE run_id = ? AND label = ?
                        ORDER BY created_at DESC LIMIT 1
                        """,
                        (run_id, label),
                    ).fetchone()
            if snapshot:
                resolution = json.loads(snapshot["resolution_json"])
                resolution["snapshot_restored"] = True
                resolution["source_live"] = False
                return resolution
            return None
        chunk_metadata = json.loads(row["metadata_json"])
        block_metadata = json.loads(row["block_metadata_json"])
        relationship = chunk_metadata.get("graph_relationship")
        artifact_path = row["frame_path"] or block_metadata.get("derived_path") or row["local_path"]
        artifact_exists = bool(artifact_path and (data_dir() / artifact_path).exists())
        resolution = {
            "run_id": run_id,
            "chunk_id": row["id"],
            "source_id": row["source_id"],
            "label": row["citation"],
            "filename": row["filename"],
            "source_type": row["source_type"],
            "block_type": row["block_type"],
            "text": row["text"],
            "block_text": row["block_text"],
            "page_number": row["page_number"],
            "timestamp_start": row["timestamp_start"],
            "timestamp_end": row["timestamp_end"],
            "frame_path": row["frame_path"],
            "artifact_path": artifact_path,
            "artifact_exists": artifact_exists,
            "image_region": row["image_region"],
            "confidence": row["confidence"],
            "metadata": chunk_metadata,
            "block_metadata": block_metadata,
            "evidence_kind": evidence_kind(row["block_type"], relationship),
            "graph_relationship": relationship,
        }
        conn.execute(
            """
            INSERT INTO citation_resolutions (id, run_id, chunk_id, source_id, label, resolution_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (str(uuid.uuid4()), run_id, row["id"], row["source_id"], row["citation"], json.dumps(resolution)),
        )
        return resolution


def snapshot_run_citations(run_id: str, results: list[dict[str, Any]]) -> None:
    seen: set[tuple[str | None, str | None]] = set()
    snapshots: list[dict[str, Any]] = []
    for result in results:
        for citation in result.get("citations", []):
            chunk_id = citation.get("chunk_id")
            label = citation.get("label") or citation.get("citation")
            key = (chunk_id, label)
            if key in seen:
                continue
            seen.add(key)
            resolution = resolve_citation_readonly(chunk_id=chunk_id, label=label, run_id=run_id)
            if not resolution:
                continue
            snapshots.append(resolution)
    with connect() as conn:
        for resolution in snapshots:
            conn.execute(
                """
                INSERT INTO run_evidence_snapshots (id, run_id, chunk_id, source_id, label, resolution_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    run_id,
                    resolution.get("chunk_id"),
                    resolution.get("source_id"),
                    resolution.get("label", ""),
                    json.dumps(resolution),
                ),
            )


def resolve_citation_readonly(chunk_id: str | None = None, label: str | None = None, run_id: str | None = None) -> dict[str, Any] | None:
    if not chunk_id and not label:
        return None
    with connect() as conn:
        row = None
        if chunk_id:
            row = conn.execute(
                """
                SELECT c.*, cb.block_type, cb.text AS block_text, cb.frame_path, cb.image_region, cb.confidence,
                       cb.metadata_json AS block_metadata_json, s.filename, s.source_type, s.local_path
                FROM chunks c
                JOIN content_blocks cb ON cb.id = c.content_block_id
                JOIN sources s ON s.id = c.source_id
                WHERE c.id = ?
                """,
                (chunk_id,),
            ).fetchone()
        if not row and label:
            row = conn.execute(
                """
                SELECT c.*, cb.block_type, cb.text AS block_text, cb.frame_path, cb.image_region, cb.confidence,
                       cb.metadata_json AS block_metadata_json, s.filename, s.source_type, s.local_path
                FROM chunks c
                JOIN content_blocks cb ON cb.id = c.content_block_id
                JOIN sources s ON s.id = c.source_id
                WHERE c.citation = ? OR c.source_reference_label = ?
                """,
                (label, label),
            ).fetchone()
        if not row:
            snapshot = None
            if run_id and (chunk_id or label):
                if chunk_id:
                    snapshot = conn.execute(
                        """
                        SELECT resolution_json FROM run_evidence_snapshots
                        WHERE run_id = ? AND chunk_id = ?
                        ORDER BY created_at DESC LIMIT 1
                        """,
                        (run_id, chunk_id),
                    ).fetchone()
                if not snapshot and label:
                    snapshot = conn.execute(
                        """
                        SELECT resolution_json FROM run_evidence_snapshots
                        WHERE run_id = ? AND label = ?
                        ORDER BY created_at DESC LIMIT 1
                        """,
                        (run_id, label),
                    ).fetchone()
            if snapshot:
                resolution = json.loads(snapshot["resolution_json"])
                resolution["snapshot_restored"] = True
                resolution["source_live"] = False
                return resolution
            return None
        chunk_metadata = json.loads(row["metadata_json"])
        block_metadata = json.loads(row["block_metadata_json"])
        relationship = chunk_metadata.get("graph_relationship")
        artifact_path = row["frame_path"] or block_metadata.get("derived_path") or row["local_path"]
        artifact_exists = bool(artifact_path and (data_dir() / artifact_path).exists())
        return {
            "run_id": run_id,
            "chunk_id": row["id"],
            "source_id": row["source_id"],
            "label": row["citation"],
            "filename": row["filename"],
            "source_type": row["source_type"],
            "block_type": row["block_type"],
            "text": row["text"],
            "block_text": row["block_text"],
            "page_number": row["page_number"],
            "timestamp_start": row["timestamp_start"],
            "timestamp_end": row["timestamp_end"],
            "frame_path": row["frame_path"],
            "artifact_path": artifact_path,
            "artifact_exists": artifact_exists,
            "image_region": row["image_region"],
            "confidence": row["confidence"],
            "metadata": chunk_metadata,
            "block_metadata": block_metadata,
            "evidence_kind": evidence_kind(row["block_type"], relationship),
            "graph_relationship": relationship,
        }


def evidence_kind(block_type: str, relationship: dict[str, Any] | None) -> str:
    if relationship:
        return "graph_relationship"
    if block_type in {"ocr", "frame_ocr", "caption"}:
        return "ocr_visual"
    if block_type == "transcript":
        return "transcript"
    if block_type == "table":
        return "structured_table"
    return "source_text"


def artifact_bytes(path: str) -> tuple[bytes, str]:
    target = data_dir() / path
    if not target.exists() or not target.is_file():
        raise FileNotFoundError(path)
    suffix = Path(path).suffix.lower()
    media_type = "application/octet-stream"
    if suffix in {".txt", ".md", ".csv", ".json"}:
        media_type = "text/plain"
    elif suffix in {".jpg", ".jpeg"}:
        media_type = "image/jpeg"
    elif suffix == ".png":
        media_type = "image/png"
    elif suffix == ".pdf":
        media_type = "application/pdf"
    return target.read_bytes(), media_type
