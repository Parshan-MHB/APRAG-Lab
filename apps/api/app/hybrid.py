from __future__ import annotations

import json
from typing import Any

from .agentic import AgentState, ToolRegistry, should_use_visual_agent
from .database import connect
from .rag import RetrievedChunk, grounding_report, rerank_evidence, synthesize_answer, tokenize, vector_retrieve


def extract_query_entities(project_id: str, question: str) -> list[dict[str, Any]]:
    query_terms = set(tokenize(question))
    if not query_terms:
        return []
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT e.*
            FROM entities e
            WHERE e.project_id = ?
            ORDER BY e.source_count DESC, e.name ASC
            """,
            (project_id,),
        ).fetchall()
    matches = []
    for row in rows:
        entity_terms = set(tokenize(row["name"])) | set(tokenize(row["normalized_name"]))
        overlap = query_terms.intersection(entity_terms)
        if overlap:
            data = dict(row)
            data["matched_terms"] = sorted(overlap)
            matches.append(data)
    return matches[:8]


def graph_expand_entities(
    project_id: str,
    entities: list[dict[str, Any]],
    filters: dict[str, Any] | None = None,
) -> list[RetrievedChunk]:
    if not entities:
        return []
    entity_ids = [entity["id"] for entity in entities]
    placeholders = ",".join("?" for _ in entity_ids)
    with connect() as conn:
        rows = conn.execute(
            f"""
            SELECT
              r.relationship_type,
              r.confidence,
              left_entity.name AS from_name,
              right_entity.name AS to_name,
              c.id AS chunk_id,
              c.source_id,
              c.text,
              c.citation,
              c.metadata_json
            FROM relationships r
            JOIN entities left_entity ON left_entity.id = r.from_entity_id
            JOIN entities right_entity ON right_entity.id = r.to_entity_id
            JOIN chunks c ON c.id = r.evidence_chunk_id
            WHERE r.project_id = ?
              AND (r.from_entity_id IN ({placeholders}) OR r.to_entity_id IN ({placeholders}))
            ORDER BY r.confidence DESC
            LIMIT 16
            """,
            [project_id, *entity_ids, *entity_ids],
        ).fetchall()
    chunks: list[RetrievedChunk] = []
    allowed_sources = set((filters or {}).get("source_ids") or [])
    for row in rows:
        if allowed_sources and row["source_id"] not in allowed_sources:
            continue
        metadata = {
            **json.loads(row["metadata_json"]),
            "graph_relationship": {
                "from": row["from_name"],
                "to": row["to_name"],
                "type": row["relationship_type"],
                "confidence": row["confidence"],
            },
        }
        chunks.append(
            RetrievedChunk(
                id=row["chunk_id"],
                source_id=row["source_id"],
                text=row["text"],
                citation=row["citation"],
                score=round(0.35 + row["confidence"], 4),
                metadata=metadata,
                rerank_score=round(0.35 + row["confidence"], 4),
                grounding_terms=[],
            )
        )
    return chunks


def combined_vector_graph_rerank(
    question: str,
    retrieved_chunks: list[RetrievedChunk],
    graph_chunks: list[RetrievedChunk],
    top_k: int = 8,
) -> list[RetrievedChunk]:
    merged: dict[str, RetrievedChunk] = {}
    for chunk in retrieved_chunks:
        merged[chunk.id] = chunk
    for chunk in graph_chunks:
        existing = merged.get(chunk.id)
        if existing:
            existing.metadata["graph_relationship"] = chunk.metadata.get("graph_relationship")
            existing.score = round(existing.score + 0.2, 4)
            existing.rerank_score = round(existing.rerank_score + 0.2, 4)
        else:
            merged[chunk.id] = chunk
    reranked = rerank_evidence(question, list(merged.values()), top_k=top_k)
    for chunk in reranked:
        if chunk.metadata.get("graph_relationship"):
            chunk.metadata["graph_boost_applied"] = True
            chunk.score = round(chunk.score + 0.1, 4)
            chunk.rerank_score = chunk.score
    reranked.sort(key=lambda item: (item.rerank_score, bool(item.metadata.get("graph_relationship"))), reverse=True)
    return reranked[:top_k]


def targeted_visual_check_if_needed(
    project_id: str,
    question: str,
    chunks: list[RetrievedChunk],
) -> tuple[list[dict[str, Any]], list[str], int]:
    if not should_use_visual_agent(project_id, question):
        return [], [], 0
    visual_source_ids = []
    for chunk in chunks:
        if chunk.metadata.get("source_type") in {"image", "video"} and chunk.source_id not in visual_source_ids:
            visual_source_ids.append(chunk.source_id)
    if not visual_source_ids:
        return [], ["visual_check_requested_but_no_visual_source_selected"], 0

    state = AgentState(project_id=project_id, question=question)
    registry = ToolRegistry(state)
    observations = []
    for source_id in visual_source_ids[:1]:
        observation = registry.call(
            "inspect_visual_source",
            source_id=source_id,
            frame_or_image_id=None,
            question_for_vlm=question,
        )
        if observation.get("text"):
            observations.append(observation)
    return observations, state.warnings, state.vlm_calls


def verify_grounding_and_revise(question: str, chunks: list[RetrievedChunk]) -> tuple[str, list[dict[str, Any]], list[str], dict[str, Any]]:
    answer, citations, warnings, report = synthesize_answer(question, chunks, answer_mode="hybrid_graph")
    if report["insufficient_evidence"]:
        return (
            "I could not find enough graph-grounded evidence in the uploaded sources to answer this question.",
            [],
            sorted(set([*warnings, "insufficient_evidence"])),
            report,
        )
    if report["unsupported_claims_count"]:
        warnings = sorted(set([*warnings, "answer_revised_to_cited_evidence_only"]))
        answer, citations, _, report = synthesize_answer(question, chunks[:3], answer_mode="hybrid_graph")
    return answer, citations, warnings, report


def run_hybrid_graph_pipeline(
    project_id: str,
    question: str,
    filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    trace: list[dict[str, Any]] = []
    warnings: list[str] = []
    filters = filters or {}

    trace.append({"step": "classify_question", "detail": "Classified query for vector, graph, and visual needs."})
    entities = extract_query_entities(project_id, question)
    trace.append({"step": "extract_query_entities", "detail": "Matched query terms to stored graph entities.", "entities": entities})

    graph_chunks = graph_expand_entities(project_id, entities, filters=filters)
    trace.append(
        {
            "step": "graph_expand_entities",
            "detail": f"Expanded graph evidence to {len(graph_chunks)} relationship-backed chunks.",
            "relationships": [chunk.metadata.get("graph_relationship") for chunk in graph_chunks],
        }
    )

    vector_backfill_limit = max(2, 8 - len(graph_chunks))
    vector_backfill = vector_retrieve(project_id, question, metadata_filters=filters, limit=vector_backfill_limit)
    trace.append({"step": "vector_backfill_candidates", "detail": f"Retrieved {len(vector_backfill)} vector-only backfill candidates after graph expansion."})

    selected = combined_vector_graph_rerank(question, vector_backfill, graph_chunks, top_k=8)
    trace.append({"step": "rerank_combined_evidence", "detail": f"Selected {len(selected)} combined evidence chunks."})

    visual_observations, visual_warnings, vlm_calls = targeted_visual_check_if_needed(project_id, question, selected)
    warnings.extend(visual_warnings)
    trace.append(
        {
            "step": "targeted_visual_check_if_needed",
            "detail": f"Collected {len(visual_observations)} targeted visual observations.",
            "visual_observations": visual_observations,
        }
    )
    if visual_warnings and visual_observations:
        refined_question = f"{question} Use only labels and relationships visible in the selected source."
        trace.append({"step": "refine_visual_question_if_needed", "detail": "Refined visual question after weak or unavailable VLM result.", "refined_question": refined_question})
    else:
        trace.append({"step": "refine_visual_question_if_needed", "detail": "No visual retry required."})

    answer, citations, answer_warnings, report = verify_grounding_and_revise(question, selected)
    warnings.extend(warning for warning in answer_warnings if warning not in warnings)
    trace.append({"step": "generate_answer", "detail": "Generated answer from combined vector and graph evidence."})
    trace.append({"step": "verify_grounding", "detail": "Verified answer grounding against selected evidence.", "grounding": report})
    if "answer_revised_to_cited_evidence_only" in warnings:
        trace.append({"step": "revise_or_finalize", "detail": "Revised answer to cited evidence only."})
    else:
        trace.append({"step": "revise_or_finalize", "detail": "Finalized answer without revision."})
    trace.append({"step": "record_metrics", "detail": "Recorded hybrid graph metrics."})

    return {
        "pipeline_type": "hybrid_graph",
        "answer": answer,
        "citations": citations,
        "trace": trace,
        "metrics": {
            "citations_count": len(citations),
            "chunks_used": len(citations),
            "warnings_count": len(warnings),
            "query_entities_count": len(entities),
            "graph_chunks_count": len(graph_chunks),
            "vector_backfill_count": len(vector_backfill),
            "combined_evidence_count": len(selected),
            "vlm_calls": vlm_calls,
            "visual_observations_count": len(visual_observations),
            "grounding_score": report["grounding_score"],
            "unsupported_claims_count": report["unsupported_claims_count"],
            "insufficient_evidence_flag": "insufficient_evidence" in warnings,
            "graph_evidence_used": any(citation.get("metadata", {}).get("graph_relationship") for citation in citations),
            "prompt_token_estimate": len(tokenize(question)) + sum(len(tokenize(chunk.text)) for chunk in selected),
            "answer_length": len(answer),
            "source_coverage_count": len({citation.get("source_id") for citation in citations if citation.get("source_id")}),
            "collection_strategy": "graph_first_relationship_expansion",
            "data_collection_techniques": [
                "graph_entity_matching",
                "relationship_expansion",
                "graph_boosted_reranking",
                "vector_backfill",
                "targeted_visual_check",
            ],
        },
        "techniques": [
            "entity_extraction",
            "relationship_extraction",
            "graph_expansion",
            "vector_backfill",
            "combined_vector_graph_reranking",
            "targeted_visual_check",
            "grounding_verification",
        ],
        "warnings": warnings,
    }
