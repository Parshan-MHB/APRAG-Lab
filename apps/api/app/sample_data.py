from __future__ import annotations

import json
import shutil
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .database import connect, data_dir
from .rag import deterministic_vector, rebuild_project_graph, run_pipeline, split_chunks, tokenize, recommend, compute_comparison


def find_sample_file_dir() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        candidate = parent / "sample-data" / "manual-test-suite"
        if candidate.exists():
            return candidate
    return Path.cwd() / "sample-data" / "manual-test-suite"


SAMPLE_FILE_DIR = find_sample_file_dir()

SAMPLE_SOURCES = [
    {
        "filename": "01_vaccine_administration_event.jpg",
        "source_type": "image",
        "block_type": "visual_observation",
        "text": (
            "Visual observation from a real-world vaccine administration event: a clinician wearing blue gloves administers a vaccine by syringe into a patient's upper arm. "
            "The image has no added text overlay and visually anchors the Lakeside Clinic cold-chain scenario because INC-1043 involved vaccine freezer A and temperature-sensitive medical inventory. "
            "The image evidence should be used only for visual context; ticket metrics and root cause come from the CSV, audio memo, and PDF review."
        ),
    },
    {
        "filename": "02_lakeside_dispatch_memo.wav",
        "source_type": "audio",
        "block_type": "transcript",
        "text": (
            "Dispatch memo transcript for incident INC-1043. "
            "Lakeside Clinic vaccine freezer A entered safe mode after a condenser fan retry storm. "
            "Priya Shah applied the cold_chain_safe profile and restarted the condenser fan service. "
            "Harbor Market stayed within range and should receive a preventive notice, not an SLA credit."
        ),
    },
    {
        "filename": "03_service_tickets.csv",
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
        "filename": "04_incident_review.pdf",
        "source_type": "pdf",
        "block_type": "page_text",
        "text": (
            "Northstar Appliances March 2026 incident review. "
            "Northstar runs connected refrigeration systems for grocery and healthcare customers. "
            "Lakeside Clinic account C-104 site SEA-17 operated vaccine freezer A. Harbor Market account C-118 site PDX-04 operated aisle freezer 3. "
            "Pine Ridge Foods account C-122 site BOI-02 operated controller 7. "
            "Firmware under review was NS-900 version 4.8.2, and replacement firmware approved for rollout was NS-900 version 4.8.3. "
            "On 2026-03-04, Lakeside Clinic opened INC-1043 after vaccine freezer A entered safe mode after three compressor restart attempts. "
            "The case was P1 because temperature-sensitive medical inventory was at risk. "
            "Logs showed firmware 4.8.2 retried the condenser fan check too aggressively after a transient sensor fault. "
            "The retry storm increased controller CPU load, delayed telemetry, and triggered safe mode for 47 minutes. "
            "The CSV metrics for INC-1043 were severity P1, 184000 dollars of inventory at risk, and 47 downtime minutes. "
            "The board decided not to roll back every customer to firmware 4.7.9 because rollback would disable telemetry compression required by service analytics. "
            "Instead, firmware 4.8.3 rolls out first to healthcare cold-chain accounts, then grocery accounts. "
            "Lakeside Clinic qualifies for an SLA credit and a compliance incident summary. "
            "Harbor Market receives a preventive maintenance notice but no SLA credit because cooling stayed in range. "
            "Pine Ridge Foods stays in the pilot group because its delayed defrost was a local night schedule error. "
            "Priya Shah owns the field runbook update, Marco Diaz owns the firmware rollout, and Elena Brooks owns customer communication."
        ),
    },
]

DEMO_QUESTIONS = [
    {
        "id": "root_cause_and_risk",
        "question": "What caused the Lakeside Clinic outage, and which CSV ticket metrics prove it was the highest-risk case?",
        "expected_terms": ["safe mode", "184000", "47"],
    },
    {
        "id": "image_context",
        "question": "What does the vaccination image show, and how does it relate to the Lakeside vaccine freezer incident?",
        "expected_terms": ["vaccine", "image", "medical inventory"],
    },
    {
        "id": "audio_dispatch_action",
        "question": "What did the audio dispatch memo say Priya Shah did for INC-1043?",
        "expected_terms": ["Priya Shah", "cold_chain_safe", "condenser fan"],
    },
    {
        "id": "firmware_customer_impact",
        "question": "Which customers were on firmware 4.8.2, and why did only Lakeside qualify for an SLA credit?",
        "expected_terms": ["Lakeside Clinic", "Harbor Market", "SLA credit"],
    },
    {
        "id": "rollback_decision",
        "question": "What did the review board decide about rollback versus staged firmware 4.8.3 rollout?",
        "expected_terms": ["rollback", "4.8.3", "healthcare"],
    },
    {
        "id": "false_positive_case",
        "question": "Was Pine Ridge Foods part of the firmware defect, or was it a different issue?",
        "expected_terms": ["Pine Ridge Foods", "night schedule", "different"],
    },
]


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def create_sample_dataset(name: str = "APRAG-Lab Sample Dataset") -> dict[str, Any]:
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
            sample_file = SAMPLE_FILE_DIR / sample["filename"]
            if sample_file.exists():
                shutil.copyfile(sample_file, source_path)
            else:
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
