from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .migrations import apply_migrations


def data_dir() -> Path:
    root = Path(os.environ.get("DATA_DIR", Path.cwd() / "data"))
    root.mkdir(parents=True, exist_ok=True)
    (root / "projects").mkdir(parents=True, exist_ok=True)
    (root / "sqlite").mkdir(parents=True, exist_ok=True)
    (root / "vector_store" / "chroma").mkdir(parents=True, exist_ok=True)
    return root


def db_path() -> Path:
    return data_dir() / "sqlite" / "APRAG-Lab.db"


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    timeout = float(os.environ.get("APRAG_SQLITE_TIMEOUT", "30"))
    conn = sqlite3.connect(db_path(), timeout=timeout)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA busy_timeout = 30000")
        try:
            conn.execute("PRAGMA journal_mode = WAL")
        except sqlite3.OperationalError:
            pass
        conn.execute("PRAGMA foreign_keys = ON")
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with connect() as conn:
        apply_migrations(conn)
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS projects (
              id TEXT PRIMARY KEY,
              name TEXT NOT NULL,
              description TEXT NOT NULL DEFAULT '',
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              processing_status TEXT NOT NULL DEFAULT 'empty',
              embedding_model TEXT NOT NULL DEFAULT 'deterministic_lexical',
              graph_status TEXT NOT NULL DEFAULT 'empty',
              active_version_id TEXT
            );

            CREATE TABLE IF NOT EXISTS knowledge_base_versions (
              id TEXT PRIMARY KEY,
              project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
              version_number INTEGER NOT NULL,
              created_at TEXT NOT NULL,
              change_type TEXT NOT NULL,
              source_ids_json TEXT NOT NULL DEFAULT '[]',
              chunk_count INTEGER NOT NULL DEFAULT 0,
              embedding_model TEXT NOT NULL DEFAULT 'deterministic_lexical',
              graph_version TEXT NOT NULL DEFAULT '1',
              notes TEXT NOT NULL DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS sources (
              id TEXT PRIMARY KEY,
              project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
              knowledge_base_version_id TEXT,
              filename TEXT NOT NULL,
              source_type TEXT NOT NULL,
              mime_type TEXT NOT NULL DEFAULT '',
              local_path TEXT NOT NULL,
              status TEXT NOT NULL,
              error TEXT,
              created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS content_blocks (
              id TEXT PRIMARY KEY,
              project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
              source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
              block_type TEXT NOT NULL,
              text TEXT NOT NULL,
              page_number INTEGER,
              timestamp_start REAL,
              timestamp_end REAL,
              frame_path TEXT,
              image_region TEXT,
              confidence REAL,
              metadata_json TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS chunks (
              id TEXT PRIMARY KEY,
              project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
              source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
              content_block_id TEXT REFERENCES content_blocks(id) ON DELETE CASCADE,
              text TEXT NOT NULL,
              citation TEXT NOT NULL,
              token_estimate INTEGER NOT NULL,
              embedding_id TEXT,
              page_number INTEGER,
              timestamp_start REAL,
              timestamp_end REAL,
              source_reference_label TEXT,
              metadata_json TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS vector_records (
              id TEXT PRIMARY KEY,
              project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
              source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
              chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
              embedding_model TEXT NOT NULL,
              vector_json TEXT NOT NULL,
              metadata_json TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS runs (
              id TEXT PRIMARY KEY,
              project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
              knowledge_base_version_id TEXT,
              question TEXT NOT NULL,
              selected_sources_json TEXT NOT NULL DEFAULT '[]',
              run_mode TEXT NOT NULL DEFAULT 'all_pipelines',
              traditional_result_id TEXT,
              agentic_result_id TEXT,
              hybrid_result_id TEXT,
              recommendation_json TEXT NOT NULL,
              parent_run_id TEXT,
              user_feedback TEXT,
              run_config_json TEXT NOT NULL DEFAULT '{}',
              created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS pipeline_results (
              id TEXT PRIMARY KEY,
              run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
              pipeline_type TEXT NOT NULL,
              answer TEXT NOT NULL,
              citations_json TEXT NOT NULL,
              trace_json TEXT NOT NULL,
              metrics_json TEXT NOT NULL,
              techniques_json TEXT NOT NULL,
              warnings_json TEXT NOT NULL,
              prompt_versions_json TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS entities (
              id TEXT PRIMARY KEY,
              project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
              name TEXT NOT NULL,
              type TEXT NOT NULL DEFAULT 'unknown',
              normalized_name TEXT NOT NULL,
              source_count INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS relationships (
              id TEXT PRIMARY KEY,
              project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
              from_entity_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
              to_entity_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
              relationship_type TEXT NOT NULL,
              evidence_chunk_id TEXT REFERENCES chunks(id) ON DELETE SET NULL,
              confidence REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS jobs (
              id TEXT PRIMARY KEY,
              project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
              run_id TEXT,
              job_type TEXT NOT NULL DEFAULT 'generic',
              payload_json TEXT NOT NULL DEFAULT '{}',
              cancellation_requested INTEGER NOT NULL DEFAULT 0,
              status TEXT NOT NULL,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS job_events (
              id TEXT PRIMARY KEY,
              job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
              run_id TEXT,
              project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
              event_type TEXT NOT NULL,
              status TEXT NOT NULL,
              message TEXT NOT NULL,
              payload_json TEXT NOT NULL DEFAULT '{}',
              created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS graph_states (
              id TEXT PRIMARY KEY,
              run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
              pipeline_type TEXT NOT NULL,
              state_json TEXT NOT NULL,
              created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS conversation_sessions (
              id TEXT PRIMARY KEY,
              project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
              title TEXT NOT NULL,
              summary TEXT NOT NULL DEFAULT '',
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS conversation_messages (
              id TEXT PRIMARY KEY,
              session_id TEXT NOT NULL REFERENCES conversation_sessions(id) ON DELETE CASCADE,
              project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
              role TEXT NOT NULL,
              content TEXT NOT NULL,
              run_id TEXT,
              source_citations_json TEXT NOT NULL DEFAULT '[]',
              generated_answer_indexed INTEGER NOT NULL DEFAULT 0,
              created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS repository_events (
              id TEXT PRIMARY KEY,
              aggregate_type TEXT NOT NULL,
              aggregate_id TEXT NOT NULL,
              event_type TEXT NOT NULL,
              payload_json TEXT NOT NULL DEFAULT '{}',
              created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS source_summaries (
              id TEXT PRIMARY KEY,
              project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
              source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
              summary TEXT NOT NULL,
              provider TEXT NOT NULL DEFAULT 'deterministic',
              model TEXT NOT NULL DEFAULT 'deterministic',
              created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
              UNIQUE(project_id, source_id)
            );

            CREATE TABLE IF NOT EXISTS structured_facts (
              id TEXT PRIMARY KEY,
              project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
              source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
              content_block_id TEXT REFERENCES content_blocks(id) ON DELETE CASCADE,
              fact_type TEXT NOT NULL,
              key TEXT,
              value TEXT NOT NULL,
              citation TEXT NOT NULL,
              metadata_json TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS citation_resolutions (
              id TEXT PRIMARY KEY,
              run_id TEXT,
              chunk_id TEXT,
              source_id TEXT,
              label TEXT NOT NULL,
              resolution_json TEXT NOT NULL,
              created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS run_evidence_snapshots (
              id TEXT PRIMARY KEY,
              run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
              chunk_id TEXT,
              source_id TEXT,
              label TEXT NOT NULL,
              resolution_json TEXT NOT NULL,
              created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS browser_only_projects (
              id TEXT PRIMARY KEY,
              name TEXT NOT NULL,
              payload_json TEXT NOT NULL,
              created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS acceptance_audits (
              id TEXT PRIMARY KEY,
              audit_type TEXT NOT NULL,
              status TEXT NOT NULL,
              report_json TEXT NOT NULL,
              created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        _ensure_column(conn, "projects", "processing_status", "TEXT NOT NULL DEFAULT 'empty'")
        _ensure_column(conn, "projects", "embedding_model", "TEXT NOT NULL DEFAULT 'deterministic_lexical'")
        _ensure_column(conn, "projects", "graph_status", "TEXT NOT NULL DEFAULT 'empty'")
        _ensure_column(conn, "projects", "active_version_id", "TEXT")
        _ensure_column(conn, "sources", "knowledge_base_version_id", "TEXT")
        _ensure_column(conn, "chunks", "content_block_id", "TEXT")
        _ensure_column(conn, "chunks", "embedding_id", "TEXT")
        _ensure_column(conn, "chunks", "page_number", "INTEGER")
        _ensure_column(conn, "chunks", "timestamp_start", "REAL")
        _ensure_column(conn, "chunks", "timestamp_end", "REAL")
        _ensure_column(conn, "chunks", "source_reference_label", "TEXT")
        _ensure_column(conn, "runs", "knowledge_base_version_id", "TEXT")
        _ensure_column(conn, "runs", "selected_sources_json", "TEXT NOT NULL DEFAULT '[]'")
        _ensure_column(conn, "runs", "run_mode", "TEXT NOT NULL DEFAULT 'all_pipelines'")
        _ensure_column(conn, "runs", "traditional_result_id", "TEXT")
        _ensure_column(conn, "runs", "agentic_result_id", "TEXT")
        _ensure_column(conn, "runs", "hybrid_result_id", "TEXT")
        _ensure_column(conn, "runs", "parent_run_id", "TEXT")
        _ensure_column(conn, "runs", "user_feedback", "TEXT")
        _ensure_column(conn, "runs", "run_config_json", "TEXT NOT NULL DEFAULT '{}'")
        _ensure_column(conn, "entities", "type", "TEXT NOT NULL DEFAULT 'unknown'")
        _ensure_column(conn, "entities", "aliases_json", "TEXT NOT NULL DEFAULT '[]'")
        _ensure_column(conn, "entities", "metadata_json", "TEXT NOT NULL DEFAULT '{}'")
        _ensure_column(conn, "relationships", "label", "TEXT NOT NULL DEFAULT 'co_occurs_with'")
        _ensure_column(conn, "relationships", "metadata_json", "TEXT NOT NULL DEFAULT '{}'")
        _ensure_column(conn, "pipeline_results", "prompt_versions_json", "TEXT NOT NULL DEFAULT '{}'")
        _ensure_column(conn, "conversation_sessions", "summary", "TEXT NOT NULL DEFAULT ''")
        _ensure_column(conn, "jobs", "job_type", "TEXT NOT NULL DEFAULT 'generic'")
        _ensure_column(conn, "jobs", "payload_json", "TEXT NOT NULL DEFAULT '{}'")
        _ensure_column(conn, "jobs", "cancellation_requested", "INTEGER NOT NULL DEFAULT 0")


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
