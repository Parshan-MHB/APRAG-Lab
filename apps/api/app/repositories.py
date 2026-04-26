from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from .database import connect


class Repository(Protocol):
    provider: str

    def create_event(self, aggregate_type: str, aggregate_id: str, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        ...

    def get_project(self, project_id: str) -> dict[str, Any] | None:
        ...

    def list_sources(self, project_id: str) -> list[dict[str, Any]]:
        ...

    def list_chunks(self, project_id: str) -> list[dict[str, Any]]:
        ...


@dataclass
class SQLiteRepository:
    provider: str = "sqlite"

    def create_event(self, aggregate_type: str, aggregate_id: str, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        event = {
            "id": str(uuid.uuid4()),
            "aggregate_type": aggregate_type,
            "aggregate_id": aggregate_id,
            "event_type": event_type,
            "payload": payload,
        }
        with connect() as conn:
            conn.execute(
                """
                INSERT INTO repository_events (id, aggregate_type, aggregate_id, event_type, payload_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (event["id"], aggregate_type, aggregate_id, event_type, json.dumps(payload)),
            )
        return event

    def get_project(self, project_id: str) -> dict[str, Any] | None:
        with connect() as conn:
            row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        return dict(row) if row else None

    def list_sources(self, project_id: str) -> list[dict[str, Any]]:
        with connect() as conn:
            rows = conn.execute("SELECT * FROM sources WHERE project_id = ? ORDER BY created_at", (project_id,)).fetchall()
        return [dict(row) for row in rows]

    def list_chunks(self, project_id: str) -> list[dict[str, Any]]:
        with connect() as conn:
            rows = conn.execute("SELECT * FROM chunks WHERE project_id = ? ORDER BY source_id, id", (project_id,)).fetchall()
        return [dict(row) for row in rows]


def repository() -> Repository:
    return SQLiteRepository()


def repository_status() -> dict[str, Any]:
    return {
        "selected": "sqlite",
        "sqlite": {"provider": "sqlite", "available": True, "schema_managed": True},
        "postgres": {"provider": "postgres", "available": False, "required": False, "removed_from_required_scope": True},
    }
