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
        "filename": "product_requirements.pdf",
        "source_type": "pdf",
        "block_type": "page_text",
        "text": "Checkout requires authentication through API Gateway and Token Store. Main risks are payment latency, deployment coordination, and unclear retry behavior.",
    },
    {
        "filename": "architecture_diagram.png",
        "source_type": "image",
        "block_type": "caption",
        "text": "Architecture diagram shows API Gateway connected to Auth Service, Token Store, Checkout Service, and Payment Processor.",
    },
    {
        "filename": "meeting_notes.docx",
        "source_type": "docx",
        "block_type": "text",
        "text": "Meeting decision: keep local-first processing, use host Ollama for real model benchmarks, and review remaining risk around video transcription speed.",
    },
    {
        "filename": "meeting_audio.wav",
        "source_type": "audio",
        "block_type": "transcript",
        "text": "Transcript: the team decided to prioritize checkout authentication and document the remaining payment retry risk.",
    },
    {
        "filename": "product_demo.mp4",
        "source_type": "video",
        "block_type": "transcript",
        "text": "Demo video transcript: checkout failed after payment timeout, then recovered after retry from the payment processor.",
    },
]

DEMO_QUESTIONS = [
    {
        "id": "document_only",
        "question": "What are the requirements and risks for checkout authentication?",
        "expected_terms": ["authentication", "API Gateway", "Token Store", "latency"],
    },
    {
        "id": "image_only",
        "question": "What does the architecture diagram show about authentication flow?",
        "expected_terms": ["API Gateway", "Auth Service", "Token Store"],
    },
    {
        "id": "video_only",
        "question": "What checkout problem appears in the product demo video?",
        "expected_terms": ["checkout", "payment timeout", "retry"],
    },
    {
        "id": "mixed_sources",
        "question": "What decisions and remaining risks are mentioned across the sources?",
        "expected_terms": ["local-first", "payment", "risk", "transcription"],
    },
    {
        "id": "meeting_audio",
        "question": "What decision was made in the meeting audio?",
        "expected_terms": ["decision", "local-first", "Ollama"],
    },
    {
        "id": "source_coverage",
        "question": "Which sources mention checkout, authentication, or deployment?",
        "expected_terms": ["checkout", "authentication", "deployment"],
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
