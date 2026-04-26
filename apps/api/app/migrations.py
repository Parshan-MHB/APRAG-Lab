from __future__ import annotations

import sqlite3
from typing import Callable

LATEST_SCHEMA_VERSION = 34

Migration = tuple[int, str, Callable[[sqlite3.Connection], None]]


def _executescript(sql: str) -> Callable[[sqlite3.Connection], None]:
    def _migration(conn: sqlite3.Connection) -> None:
        conn.executescript(sql)

    return _migration


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    table_exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    if not table_exists:
        return
    columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


def _base_schema(conn: sqlite3.Connection) -> None:
    from .database import init_db

    # database.init_db owns the complete SQLite schema for existing installations.
    # Migrations below add upgradeable version markers and post-Epic-28 fields.
    return None


def _epic_29_repositories(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS repository_events (
          id TEXT PRIMARY KEY,
          aggregate_type TEXT NOT NULL,
          aggregate_id TEXT NOT NULL,
          event_type TEXT NOT NULL,
          payload_json TEXT NOT NULL DEFAULT '{}',
          created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """
    )


def _epic_30_graph_fields(conn: sqlite3.Connection) -> None:
    _ensure_column(conn, "entities", "aliases_json", "TEXT NOT NULL DEFAULT '[]'")
    _ensure_column(conn, "entities", "metadata_json", "TEXT NOT NULL DEFAULT '{}'")
    _ensure_column(conn, "relationships", "label", "TEXT NOT NULL DEFAULT 'co_occurs_with'")
    _ensure_column(conn, "relationships", "metadata_json", "TEXT NOT NULL DEFAULT '{}'")


def _epic_31_advanced_rag_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
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
        """
    )


def _epic_32_citation_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS citation_resolutions (
          id TEXT PRIMARY KEY,
          run_id TEXT,
          chunk_id TEXT,
          source_id TEXT,
          label TEXT NOT NULL,
          resolution_json TEXT NOT NULL,
          created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """
    )


def _epic_33_app_modes(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS browser_only_projects (
          id TEXT PRIMARY KEY,
          name TEXT NOT NULL,
          payload_json TEXT NOT NULL,
          created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """
    )


def _epic_34_acceptance(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS acceptance_audits (
          id TEXT PRIMARY KEY,
          audit_type TEXT NOT NULL,
          status TEXT NOT NULL,
          report_json TEXT NOT NULL,
          created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """
    )


MIGRATIONS: list[Migration] = [
    (1, "initial_project_source_run_tables", _base_schema),
    (2, "knowledge_base_versioning", _base_schema),
    (3, "content_blocks_and_vectors", _base_schema),
    (4, "retrieval_metadata", _base_schema),
    (5, "agentic_contracts", _base_schema),
    (6, "hybrid_graph_tables", _base_schema),
    (7, "comparison_exports", _base_schema),
    (8, "frontend_contract_support", _base_schema),
    (9, "jobs_traces_diagnostics", _base_schema),
    (10, "provider_adapters", _base_schema),
    (11, "graph_state_persistence", _base_schema),
    (12, "run_config_prompt_versions", _base_schema),
    (13, "privacy_lifecycle", _base_schema),
    (14, "sample_acceptance_suite", _base_schema),
    (15, "roadmap_guardrails", _base_schema),
    (16, "stable_jobs_api", _base_schema),
    (17, "storage_layout_reproducibility", _base_schema),
    (18, "non_functional_reliability_gates", _base_schema),
    (19, "advanced_rag_variants", _base_schema),
    (20, "storage_runtime_adapters", _base_schema),
    (21, "conversation_desktop_browser_modes", _base_schema),
    (22, "paid_cloud_extension_interfaces", _base_schema),
    (23, "real_provider_bootstrap_health", _base_schema),
    (24, "real_model_provider_wiring", _base_schema),
    (25, "real_vector_store_lifecycle", _base_schema),
    (26, "real_media_ingestion", _base_schema),
    (27, "real_langgraph_orchestration", _base_schema),
    (28, "real_queue_execution", _base_schema),
    (29, "real_repository_layer", _epic_29_repositories),
    (30, "graph_entity_relationship_metadata", _epic_30_graph_fields),
    (31, "advanced_rag_work_tables", _epic_31_advanced_rag_tables),
    (32, "citation_resolution_cache", _epic_32_citation_tables),
    (33, "app_mode_persistence", _epic_33_app_modes),
    (34, "final_acceptance_audit", _epic_34_acceptance),
]


def ensure_migration_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
          version INTEGER PRIMARY KEY,
          name TEXT NOT NULL,
          applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


def applied_versions(conn: sqlite3.Connection) -> set[int]:
    ensure_migration_table(conn)
    return {row["version"] for row in conn.execute("SELECT version FROM schema_migrations").fetchall()}


def apply_migrations(conn: sqlite3.Connection, target_version: int = LATEST_SCHEMA_VERSION) -> list[dict[str, object]]:
    ensure_migration_table(conn)
    applied = applied_versions(conn)
    newly_applied: list[dict[str, object]] = []
    for version, name, migration in MIGRATIONS:
        if version > target_version or version in applied:
            continue
        migration(conn)
        conn.execute("INSERT INTO schema_migrations (version, name) VALUES (?, ?)", (version, name))
        newly_applied.append({"version": version, "name": name})
    return newly_applied


def migration_status(conn: sqlite3.Connection) -> dict[str, object]:
    versions = applied_versions(conn)
    return {
        "latest_schema_version": LATEST_SCHEMA_VERSION,
        "applied_versions": sorted(versions),
        "current_version": max(versions) if versions else 0,
        "pending_versions": [version for version, _, _ in MIGRATIONS if version not in versions],
    }
