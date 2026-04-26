from __future__ import annotations

import json
import uuid
from typing import Any

from .database import connect


def now_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


def app_modes_status() -> dict[str, Any]:
    return {
        "benchmark_mode": {"available": True, "default": "independent"},
        "conversation_mode": {
            "available": True,
            "source_authority": "uploaded_sources_only",
            "generated_answers_indexed": False,
        },
        "desktop_app": {
            "available": True,
            "local_first_storage": True,
            "cloud_required": False,
            "launch_command": "cd apps/desktop && npm install && npm start",
            "smoke_command": "cd apps/desktop && npm run smoke",
            "healthcheck": "/health",
        },
        "browser_only_local_mode": {
            "available": True,
            "entrypoint": "apps/web/browser-only.html",
            "feature_matrix": {
                "project_and_run_model": "supported",
                "text_markdown_parsing": "supported_in_browser",
                "deterministic_keyword_retrieval": "supported_in_browser",
                "pdf_docx_parsing": "requires_backend_container",
                "image_audio_video_processing": "requires_backend_container",
                "local_llm_vlm_inference": "requires_host_ollama",
                "persistent_sqlite_storage": "requires_backend_container",
            },
            "controlled_unavailable_state": "requires_backend_or_host_ollama",
        },
    }


def conversation_summary(question: str, previous_citations: list[dict[str, Any]]) -> str:
    labels = [citation.get("label", "") for citation in previous_citations if citation.get("label")]
    if not labels:
        return f"Follow-up context: {question[:120]}"
    return f"Follow-up context from previous citations: {', '.join(labels[:3])}. Current question: {question[:120]}"


def create_session(project_id: str, title: str) -> dict[str, Any]:
    session_id = str(uuid.uuid4())
    timestamp = now_iso()
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO conversation_sessions (id, project_id, title, summary, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (session_id, project_id, title, "", timestamp, timestamp),
        )
    return get_session(session_id)


def get_session(session_id: str) -> dict[str, Any]:
    with connect() as conn:
        session = conn.execute("SELECT * FROM conversation_sessions WHERE id = ?", (session_id,)).fetchone()
        if not session:
            raise KeyError(session_id)
        messages = conn.execute(
            "SELECT * FROM conversation_messages WHERE session_id = ? ORDER BY created_at ASC",
            (session_id,),
        ).fetchall()
    return {
        "id": session["id"],
        "project_id": session["project_id"],
        "title": session["title"],
        "summary": session["summary"],
        "created_at": session["created_at"],
        "updated_at": session["updated_at"],
        "messages": [
            {
                "id": message["id"],
                "role": message["role"],
                "content": message["content"],
                "run_id": message["run_id"],
                "source_citations": json.loads(message["source_citations_json"]),
                "generated_answer_indexed": bool(message["generated_answer_indexed"]),
                "created_at": message["created_at"],
            }
            for message in messages
        ],
    }


def add_message_pair(
    session_id: str,
    project_id: str,
    question: str,
    answer: str,
    run_id: str,
    citations: list[dict[str, Any]],
    summary: str,
) -> dict[str, Any]:
    timestamp = now_iso()
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO conversation_messages
              (id, session_id, project_id, role, content, run_id, source_citations_json, generated_answer_indexed, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (str(uuid.uuid4()), session_id, project_id, "user", question, None, "[]", 0, timestamp),
        )
        conn.execute(
            """
            INSERT INTO conversation_messages
              (id, session_id, project_id, role, content, run_id, source_citations_json, generated_answer_indexed, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (str(uuid.uuid4()), session_id, project_id, "assistant", answer, run_id, json.dumps(citations), 0, timestamp),
        )
        conn.execute(
            "UPDATE conversation_sessions SET summary = ?, updated_at = ? WHERE id = ?",
            (summary, timestamp, session_id),
        )
    return get_session(session_id)
