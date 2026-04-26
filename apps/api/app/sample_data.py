from __future__ import annotations

import json
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .database import connect, data_dir
from .rag import deterministic_vector, rebuild_project_graph, run_pipeline, split_chunks, tokenize, recommend, compute_comparison


SAMPLE_SOURCES = [
    {
        "filename": "01_northstar_incident_brief.md",
        "source_type": "markdown",
        "block_type": "document_text",
        "text": (
            "Northstar Appliances runs connected refrigeration systems for grocery and healthcare customers. "
            "Customer Lakeside Clinic account C-104 at site SEA-17 opened ticket INC-1043 on 2026-03-04 after vaccine freezer A entered safe mode. "
            "The freezer held temperature-sensitive inventory, so the case was severity P1. "
            "Logs showed NS-900 firmware 4.8.2 retried the condenser fan check too aggressively after a transient sensor fault. "
            "The retry storm increased controller CPU load, delayed telemetry, and triggered safe mode for 47 minutes. "
            "The immediate workaround was to pin the controller to profile cold_chain_safe and restart the condenser fan service. "
            "Harbor Market account C-118 at site PDX-04 opened INC-1051 for intermittent alarm noise on aisle freezer 3. "
            "Harbor also ran NS-900 firmware 4.8.2, but cooling stayed in range and there was no safe-mode transition. "
            "Pine Ridge Foods account C-122 at BOI-02 opened INC-1062 for a delayed defrost cycle on firmware 4.8.3; root cause was a local night schedule misconfiguration. "
            "The review board decided not to roll back all customers to firmware 4.7.9. "
            "Instead, Marco Diaz will ship firmware 4.8.3 to cold-chain healthcare accounts first, then grocery accounts. "
            "Priya Shah owns the field runbook update. Elena Brooks owns customer communication for Lakeside Clinic and Harbor Market. "
            "Lakeside Clinic qualifies for an SLA credit because medical inventory was at risk and manual intervention was required. "
            "The ticket metrics for INC-1043 recorded P1 severity, 184000 dollars of inventory at risk, and 47 minutes of downtime. "
            "Harbor Market does not qualify for an SLA credit because cooling stayed within range. "
            "Pine Ridge Foods does not qualify because the issue was a local schedule configuration error."
        ),
    },
    {
        "filename": "02_service_tickets.csv",
        "source_type": "csv",
        "block_type": "table_text",
        "text": (
            "ticket_id,date,customer,account_id,site_id,asset,firmware,severity,symptom,cooling_within_range,inventory_at_risk_usd,downtime_minutes,root_cause,mitigation,sla_credit,owner\n"
            "INC-1043,2026-03-04,Lakeside Clinic,C-104,SEA-17,vaccine freezer A,NS-900 4.8.2,P1,safe mode after three compressor restart attempts,no,184000,47,condenser fan retry storm after transient sensor fault,cold_chain_safe profile and staged firmware 4.8.3 rollout,yes,Priya Shah\n"
            "INC-1051,2026-03-05,Harbor Market,C-118,PDX-04,aisle freezer 3,NS-900 4.8.2,P2,intermittent alarm noise,yes,0,0,same firmware retry warning but no safe-mode transition,preventive maintenance notice and firmware 4.8.3 maintenance window,no,Elena Brooks\n"
            "INC-1062,2026-03-08,Pine Ridge Foods,C-122,BOI-02,controller 7,NS-900 4.8.3,P3,delayed defrost cycle,yes,0,0,local night schedule misconfiguration,correct night schedule and keep site in pilot group,no,Marco Diaz\n"
            "INC-1069,2026-03-10,Lakeside Clinic,C-104,SEA-17,backup freezer B,NS-900 4.8.3,P3,post-mitigation telemetry delay check,yes,0,0,verification event after firmware 4.8.3 pilot,monitor telemetry delay for 72 hours,no,Priya Shah"
        ),
    },
    {
        "filename": "03_service_review_notes.md",
        "source_type": "markdown",
        "block_type": "document_text",
        "text": (
            "The 2026-03-11 service review compared March incident tickets with fleet metrics. "
            "The team agreed the problem was not a general refrigeration failure. "
            "It was a firmware-specific control-loop defect affecting NS-900 version 4.8.2 when condenser fan sensor data briefly dropped out. "
            "Priya Shah reported field technicians could apply the cold_chain_safe profile in under 12 minutes. "
            "Marco Diaz confirmed firmware 4.8.3 changes the retry policy from three immediate retries to one retry followed by a 90-second cooldown. "
            "Elena Brooks requested separate customer messaging for healthcare customers and grocery customers. "
            "Decision 1: do not perform a broad rollback to firmware 4.7.9. "
            "Decision 2: deploy firmware 4.8.3 first to healthcare cold-chain sites, starting with Lakeside Clinic SEA-17. "
            "Decision 3: send Harbor Market a preventive notice and maintenance window, but no SLA credit. "
            "Decision 4: issue Lakeside Clinic an SLA credit and provide a compliance incident summary. "
            "Decision 5: keep Pine Ridge Foods in the pilot group and correct its night schedule. "
            "Open actions: Priya Shah publishes the cold_chain_safe runbook by 2026-03-13; Marco Diaz releases the staged rollout package by 2026-03-14; Elena Brooks sends the Lakeside credit memo and Harbor preventive notice by 2026-03-15."
        ),
    },
]

DEMO_QUESTIONS = [
    {
        "id": "root_cause_and_risk",
        "question": "What caused the Lakeside Clinic outage, and which ticket metrics prove it was the highest-risk case?",
        "expected_terms": ["safe mode", "control-loop", "SLA credit"],
    },
    {
        "id": "firmware_customer_impact",
        "question": "Which customers were affected by firmware 4.8.2, and why did only one qualify for an SLA credit?",
        "expected_terms": ["Lakeside Clinic", "Harbor Market", "SLA credit"],
    },
    {
        "id": "rollback_decision",
        "question": "What did the review board decide about rollback versus staged firmware 4.8.3 rollout?",
        "expected_terms": ["rollback", "4.8.3", "healthcare"],
    },
    {
        "id": "site_comparison",
        "question": "Compare Lakeside Clinic, Harbor Market, and Pine Ridge Foods by root cause, severity, and mitigation.",
        "expected_terms": ["Lakeside Clinic", "Harbor Market", "Pine Ridge Foods", "mitigation"],
    },
    {
        "id": "owners_and_actions",
        "question": "Which owner is responsible for each follow-up action, and what evidence connects the owner to the ticket?",
        "expected_terms": ["Priya Shah", "Marco Diaz", "Elena Brooks"],
    },
    {
        "id": "false_positive_case",
        "question": "Was Pine Ridge Foods part of the firmware defect, or was it a different issue?",
        "expected_terms": ["Pine Ridge Foods", "night schedule", "different"],
    },
]


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def create_sample_dataset(name: str = "RAGBench Sample Dataset") -> dict[str, Any]:
    project_id = str(uuid.uuid4())
    timestamp = now_iso()
    project_path = data_dir() / "projects" / project_id
    original_dir = project_path / "sources" / "original"
    original_dir.mkdir(parents=True, exist_ok=True)
    source_ids: list[str] = []

    with connect() as conn:
        conn.execute(
            """
            INSERT INTO projects (id, name, description, created_at, updated_at, processing_status)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (project_id, name, "Built-in deterministic acceptance/demo dataset.", timestamp, timestamp, "processed"),
        )
        for sample in SAMPLE_SOURCES:
            source_id = str(uuid.uuid4())
            source_ids.append(source_id)
            source_path = original_dir / f"{source_id}_{sample['filename']}"
            source_path.write_text(sample["text"], encoding="utf-8")
            conn.execute(
                """
                INSERT INTO sources
                  (id, project_id, knowledge_base_version_id, filename, source_type, mime_type, local_path, status, error, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source_id,
                    project_id,
                    None,
                    sample["filename"],
                    sample["source_type"],
                    "application/octet-stream",
                    str(source_path.relative_to(data_dir())),
                    "processed",
                    None,
                    timestamp,
                ),
            )
            block_id = str(uuid.uuid4())
            conn.execute(
                """
                INSERT INTO content_blocks
                  (id, project_id, source_id, block_type, text, page_number, timestamp_start, timestamp_end,
                   frame_path, image_region, confidence, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    block_id,
                    project_id,
                    source_id,
                    sample["block_type"],
                    sample["text"],
                    1 if sample["source_type"] == "pdf" else None,
                    0.0 if sample["source_type"] in {"audio", "video"} else None,
                    12.0 if sample["source_type"] in {"audio", "video"} else None,
                    "sources/derived/frames/sample_frame.txt" if sample["source_type"] == "video" else None,
                    None,
                    0.85,
                    json.dumps({"sample_dataset": True, "filename": sample["filename"]}),
                ),
            )
            for index, chunk in enumerate(split_chunks(sample["text"])):
                chunk_id = str(uuid.uuid4())
                embedding_id = str(uuid.uuid4())
                citation = f"{sample['filename']}#chunk-{index + 1}"
                metadata = {
                    "chunk_index": index + 1,
                    "source_type": sample["source_type"],
                    "block_type": sample["block_type"],
                    "sample_dataset": True,
                    "filename": sample["filename"],
                }
                conn.execute(
                    """
                    INSERT INTO chunks
                      (id, project_id, source_id, content_block_id, text, citation, token_estimate,
                       embedding_id, page_number, timestamp_start, timestamp_end, source_reference_label, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk_id,
                        project_id,
                        source_id,
                        block_id,
                        chunk,
                        citation,
                        len(tokenize(chunk)),
                        embedding_id,
                        1 if sample["source_type"] == "pdf" else None,
                        0.0 if sample["source_type"] in {"audio", "video"} else None,
                        12.0 if sample["source_type"] in {"audio", "video"} else None,
                        citation,
                        json.dumps(metadata),
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO vector_records
                      (id, project_id, source_id, chunk_id, embedding_model, vector_json, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        embedding_id,
                        project_id,
                        source_id,
                        chunk_id,
                        "deterministic_lexical",
                        json.dumps(deterministic_vector(chunk)),
                        json.dumps({"source_reference_label": citation}),
                    ),
                )

        version_id = str(uuid.uuid4())
        conn.execute(
            """
            INSERT INTO knowledge_base_versions
              (id, project_id, version_number, created_at, change_type, source_ids_json, chunk_count, embedding_model, graph_version, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (version_id, project_id, 1, timestamp, "sample_dataset", json.dumps(source_ids), len(SAMPLE_SOURCES), "deterministic_lexical", "1", "Built-in sample dataset."),
        )
        conn.execute("UPDATE sources SET knowledge_base_version_id = ? WHERE project_id = ?", (version_id, project_id))
        conn.execute("UPDATE projects SET active_version_id = ?, graph_status = ? WHERE id = ?", (version_id, "ready", project_id))

    rebuild_project_graph(project_id)
    return {"project_id": project_id, "source_count": len(SAMPLE_SOURCES), "questions": DEMO_QUESTIONS, "normal_ingestion_path": True}


def run_acceptance_suite(project_id: str | None = None) -> dict[str, Any]:
    start = time.perf_counter()
    dataset = create_sample_dataset() if project_id is None else {"project_id": project_id, "questions": DEMO_QUESTIONS}
    results = []
    for item in DEMO_QUESTIONS:
        question_start = time.perf_counter()
        pipeline_results = [run_pipeline(dataset["project_id"], item["question"], pipeline) for pipeline in ["traditional", "agentic", "hybrid_graph"]]
        recommendation = recommend(pipeline_results)
        comparison = compute_comparison(pipeline_results)
        joined = " ".join(result["answer"] for result in pipeline_results).lower()
        expected_hits = [term for term in item["expected_terms"] if term.lower() in joined]
        results.append(
            {
                "id": item["id"],
                "question": item["question"],
                "passed": bool(expected_hits) and any(result["citations"] for result in pipeline_results),
                "expected_hits": expected_hits,
                "recommended_flow": recommendation["recommended_flow"],
                "best_pipeline": comparison["best_pipeline"],
                "elapsed_seconds": round(time.perf_counter() - question_start, 4),
            }
        )
    return {
        "project_id": dataset["project_id"],
        "passed": all(result["passed"] for result in results),
        "results": results,
        "elapsed_seconds": round(time.perf_counter() - start, 4),
    }
