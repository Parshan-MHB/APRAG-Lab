from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any

from .database import connect
from .observability import observe_span
from .rag import RetrievedChunk, retrieve, rerank_evidence, tokenize

ADVANCED_VARIANTS = (
    "advanced_rag",
    "hybrid_search_rag",
    "fusion_rag",
    "corrective_rag",
    "self_rag",
    "hierarchical_rag",
    "structured_rag",
    "long_context_rag",
    "multimodal_rag",
    "graph_rag",
    "adaptive_rag",
)


def rewrite_query(question: str) -> str:
    terms = tokenize(question)
    if not terms:
        return question
    if not terms:
        return question
    try:
        from .providers import deterministic_mode, provider_registry

        if not deterministic_mode():
            result = provider_registry().llm().generate(
                "Rewrite this retrieval query to maximize source recall. Return one concise query only:\n" + question
            )
            return result.text.strip() or f"{question} {' '.join(sorted(set(terms))[:6])}"
    except Exception:
        pass
    return f"{question} {' '.join(sorted(set(terms))[:6])}"


def fusion_query_variants(question: str, limit: int = 3) -> list[str]:
    base = question.strip()
    variants = [base, rewrite_query(base), re.sub(r"\bwhat\b", "which evidence", base, flags=re.I)]
    deduped: list[str] = []
    for variant in variants:
        if variant and variant not in deduped:
            deduped.append(variant)
    return deduped[:limit]


def fusion_retrieve(project_id: str, question: str, filters: dict[str, Any], top_k: int = 8) -> tuple[list[RetrievedChunk], list[str]]:
    with observe_span(
        "rag.fusion_retrieve",
        {"project_id": project_id, "top_k": top_k, "question_length": len(question), "filter_count": len(filters)},
        {"question": question, "filters": filters},
    ) as span:
        merged: dict[str, RetrievedChunk] = {}
        variants = fusion_query_variants(question)
        for variant in variants:
            for chunk in retrieve(project_id, variant, top_k=top_k, metadata_filters=filters):
                existing = merged.get(chunk.id)
                if existing:
                    existing.score = max(existing.score, chunk.score)
                    existing.rerank_score = max(existing.rerank_score, chunk.rerank_score)
                    existing.grounding_terms = sorted(set(existing.grounding_terms or []).union(chunk.grounding_terms or []))
                else:
                    merged[chunk.id] = chunk
        reranked = rerank_evidence(question, list(merged.values()), top_k=top_k)
        if span:
            span.set_attribute("query_variant_count", len(variants))
            span.set_attribute("merged_hit_count", len(merged))
            span.set_attribute("reranked_hit_count", len(reranked))
        return reranked, variants


def source_summaries(project_id: str) -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT s.id, s.filename, s.source_type, COALESCE(ss.summary, GROUP_CONCAT(substr(c.text, 1, 180), ' ')) AS summary
            FROM sources s
            LEFT JOIN chunks c ON c.source_id = s.id
            LEFT JOIN source_summaries ss ON ss.source_id = s.id
            WHERE s.project_id = ?
            GROUP BY s.id, s.filename, s.source_type, ss.summary
            ORDER BY s.filename ASC
            """,
            (project_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def structured_facts(project_id: str) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    with connect() as conn:
        stored = conn.execute(
            """
            SELECT sf.*, s.filename
            FROM structured_facts sf
            JOIN sources s ON s.id = sf.source_id
            WHERE sf.project_id = ?
            """,
            (project_id,),
        ).fetchall()
        for row in stored:
            facts.append(
                {
                    "fact_type": row["fact_type"],
                    "source_id": row["source_id"],
                    "filename": row["filename"],
                    "key": row["key"],
                    "value": row["value"],
                    "citation": row["citation"],
                    "metadata": json.loads(row["metadata_json"]),
                }
            )
        rows = conn.execute(
            """
            SELECT cb.id, cb.source_id, cb.block_type, cb.text, cb.metadata_json, s.filename
            FROM content_blocks cb
            JOIN sources s ON s.id = cb.source_id
            WHERE cb.project_id = ?
            """,
            (project_id,),
        ).fetchall()
    for row in rows:
        metadata = json.loads(row["metadata_json"])
        if row["block_type"] == "table" or "|" in row["text"]:
            facts.append(
                {
                    "fact_type": "table",
                    "source_id": row["source_id"],
                    "filename": row["filename"],
                    "text": row["text"],
                    "metadata": metadata,
                }
            )
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                facts.append(
                    {
                        "fact_type": "metadata",
                        "source_id": row["source_id"],
                        "filename": row["filename"],
                        "key": key,
                        "value": value,
                    }
                )
    return facts


def query_structured_facts(project_id: str, question: str) -> list[dict[str, Any]]:
    terms = set(tokenize(question))
    matches = []
    for fact in structured_facts(project_id):
        text = json.dumps(fact, sort_keys=True)
        if terms.intersection(tokenize(text)):
            matches.append(fact)
    return matches[:8]


def assemble_long_context(chunks: list[RetrievedChunk], summaries: list[dict[str, Any]], max_tokens: int = 1800) -> dict[str, Any]:
    context_parts: list[str] = []
    citations: list[str] = []
    token_count = 0
    for summary in summaries[:5]:
        text = f"Source summary {summary['filename']}: {summary.get('summary') or ''}".strip()
        tokens = len(tokenize(text))
        if token_count + tokens > max_tokens:
            break
        context_parts.append(text)
        token_count += tokens
    for chunk in chunks:
        tokens = len(tokenize(chunk.text))
        if token_count + tokens > max_tokens:
            break
        context_parts.append(f"{chunk.citation}: {chunk.text}")
        citations.append(chunk.citation)
        token_count += tokens
    return {"context": "\n\n".join(context_parts), "citations": citations, "token_estimate": token_count, "max_tokens": max_tokens}


def activated_advanced_variants(project_id: str, question: str, chunks: list[RetrievedChunk], warnings: list[str]) -> dict[str, Any]:
    summaries = source_summaries(project_id)
    structured = query_structured_facts(project_id, question)
    source_types = defaultdict(int)
    for chunk in chunks:
        source_types[chunk.metadata.get("source_type", "unknown")] += 1
    weak_evidence = not chunks or "insufficient_evidence" in warnings
    corrective_variants = fusion_query_variants(rewrite_query(question), limit=3) if weak_evidence else []
    self_rag_action = "retrieve_more" if weak_evidence else "finalize"
    if warnings and not weak_evidence:
        self_rag_action = "revise_with_citations"
    return {
        "advanced_rag": {"activated": True, "features": ["query_rewrite", "metadata_filters", "reranking", "improved_chunking"]},
        "hybrid_search_rag": {"activated": True, "features": ["vector_search", "keyword_search", "deduplication", "score_normalization"]},
        "fusion_rag": {"activated": True, "query_variants": fusion_query_variants(question)},
        "corrective_rag": {"activated": weak_evidence, "action": "retrieve_more" if weak_evidence else "not_required", "retry_queries": corrective_variants},
        "self_rag": {"activated": True, "critic": "grounding_and_sufficiency", "action": self_rag_action, "warnings_seen": warnings},
        "hierarchical_rag": {"activated": bool(summaries), "source_summaries_count": len(summaries)},
        "structured_rag": {"activated": bool(structured), "facts_count": len(structured), "facts": structured},
        "long_context_rag": {"activated": bool(chunks or summaries), "context": assemble_long_context(chunks, summaries)},
        "multimodal_rag": {"activated": any(kind in source_types for kind in ["image", "audio", "video"]), "source_types": dict(source_types)},
        "graph_rag": {"activated": False, "note": "Primary graph execution is exposed in Hybrid Graph pipeline."},
        "adaptive_rag": {"activated": True, "note": "Recommendation layer compares pipeline evidence quality."},
    }
