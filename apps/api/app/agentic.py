from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel, Field

from .database import connect
from .database import data_dir
from .providers import ProviderUnavailable, deterministic_mode, provider_registry
from .rag import (
    RetrievedChunk,
    graph_diagnostics,
    grounding_report,
    keyword_retrieve,
    rerank_evidence,
    retrieve,
    synthesize_answer,
    tokenize,
    vector_retrieve,
)


AGENT_LIMITS = {
    "max_specialist_agents": 3,
    "max_agent_steps": 6,
    "max_agent_rounds": 2,
    "max_retrieval_calls": 3,
    "max_vlm_calls": 2,
    "max_vlm_retries_per_source": 1,
    "max_graph_agent_calls": 1,
    "max_critic_passes": 1,
    "max_total_tool_calls": 8,
    "max_query_variants": 3,
    "max_chunks_in_context": 8,
    "max_answer_revision_attempts": 1,
}

TOOL_NAMES = [
    "search_vector_store",
    "search_keyword_index",
    "rerank_evidence",
    "inspect_chunk",
    "inspect_visual_source",
    "refine_visual_question",
    "inspect_transcript",
    "lookup_entities",
    "evaluate_grounding",
    "final_answer",
]

VISUAL_TERMS = {"diagram", "image", "photo", "picture", "visual", "chart", "frame", "video", "screenshot"}
GRAPH_TERMS = {"connect", "connected", "relationship", "related", "depends", "dependency", "graph", "between"}
WEAK_VISUAL_PHRASES = {
    "image shows",
    "shows a diagram",
    "system diagram",
    "not sure",
    "unclear",
    "cannot determine",
    "can't determine",
    "no relevant",
}


class EvidenceItem(BaseModel):
    chunk_id: str | None = None
    source_id: str | None = None
    citation: str | None = None
    text: str
    score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidencePackage(BaseModel):
    agent_name: str
    task: str
    query_used: str
    evidence_items: list[EvidenceItem] = Field(default_factory=list)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    confidence: float = 0.0
    gaps: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    recommended_next_action: str


@dataclass
class AgentState:
    project_id: str
    question: str
    filters: dict[str, Any] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    tool_calls: int = 0
    retrieval_calls: int = 0
    vlm_calls: int = 0
    graph_calls: int = 0
    critic_passes: int = 0
    steps: int = 0
    specialists_used: list[str] = field(default_factory=list)
    evidence_packages: list[EvidencePackage] = field(default_factory=list)

    def step(self, name: str, detail: str, **extra: Any) -> None:
        self.steps += 1
        if self.steps > AGENT_LIMITS["max_agent_steps"]:
            self.warnings.append("agent_step_limit_reached")
            return
        event = {"step": name, "detail": detail}
        event.update(extra)
        self.trace.append(event)

    def tool(self, name: str, **args: Any) -> None:
        self.tool_calls += 1
        if self.tool_calls > AGENT_LIMITS["max_total_tool_calls"]:
            self.warnings.append("agent_tool_limit_reached")
            return
        self.trace.append({"step": "tool_call", "tool": name, "args": scrub_tool_args(args)})


def scrub_tool_args(args: dict[str, Any]) -> dict[str, Any]:
    scrubbed = dict(args)
    if "retrieved_chunks" in scrubbed:
        scrubbed["retrieved_chunks"] = f"{len(scrubbed['retrieved_chunks'])} chunks"
    if "evidence" in scrubbed:
        scrubbed["evidence"] = f"{len(scrubbed['evidence'])} items"
    return scrubbed


class ToolRegistry:
    def __init__(self, state: AgentState):
        self.state = state
        self._tools: dict[str, Callable[..., Any]] = {
            "search_vector_store": self.search_vector_store,
            "search_keyword_index": self.search_keyword_index,
            "rerank_evidence": self.rerank_tool,
            "inspect_chunk": self.inspect_chunk,
            "inspect_visual_source": self.inspect_visual_source,
            "refine_visual_question": self.refine_visual_question,
            "inspect_transcript": self.inspect_transcript,
            "lookup_entities": self.lookup_entities,
            "evaluate_grounding": self.evaluate_grounding,
            "final_answer": self.final_answer,
        }

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def call(self, name: str, **kwargs: Any) -> Any:
        if name not in self._tools:
            self.state.warnings.append("malformed_tool_call")
            raise ValueError(f"Tool is not registered: {name}")
        self.state.tool(name, **kwargs)
        return self._tools[name](**kwargs)

    def search_vector_store(self, query: str, filters: dict[str, Any] | None = None, top_k: int = 8) -> list[RetrievedChunk]:
        self.state.retrieval_calls += 1
        if self.state.retrieval_calls > AGENT_LIMITS["max_retrieval_calls"]:
            self.state.warnings.append("retrieval_call_limit_reached")
            return []
        return vector_retrieve(self.state.project_id, query, filters or self.state.filters, min(top_k, AGENT_LIMITS["max_chunks_in_context"]))

    def search_keyword_index(self, query: str, filters: dict[str, Any] | None = None, top_k: int = 8) -> list[RetrievedChunk]:
        self.state.retrieval_calls += 1
        if self.state.retrieval_calls > AGENT_LIMITS["max_retrieval_calls"]:
            self.state.warnings.append("retrieval_call_limit_reached")
            return []
        return keyword_retrieve(self.state.project_id, query, filters or self.state.filters, min(top_k, AGENT_LIMITS["max_chunks_in_context"]))

    def rerank_tool(self, question: str, retrieved_chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        return rerank_evidence(question, retrieved_chunks, top_k=AGENT_LIMITS["max_chunks_in_context"])

    def inspect_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        with connect() as conn:
            row = conn.execute("SELECT * FROM chunks WHERE id = ? AND project_id = ?", (chunk_id, self.state.project_id)).fetchone()
        return dict(row) if row else None

    def inspect_visual_source(self, source_id: str, frame_or_image_id: str | None, question_for_vlm: str) -> dict[str, Any]:
        self.state.vlm_calls += 1
        if self.state.vlm_calls > AGENT_LIMITS["max_vlm_calls"]:
            self.state.warnings.append("vlm_call_limit_reached")
            return {"available": False, "warning": "vlm_call_limit_reached", "text": ""}
        with connect() as conn:
            source = conn.execute("SELECT * FROM sources WHERE id = ? AND project_id = ?", (source_id, self.state.project_id)).fetchone()
            blocks = conn.execute(
                """
                SELECT id, block_type, text, frame_path, confidence, metadata_json
                FROM content_blocks
                WHERE project_id = ? AND source_id = ? AND block_type IN ('ocr', 'frame_ocr', 'caption', 'transcript')
                ORDER BY block_type
                """,
                (self.state.project_id, source_id),
            ).fetchall()
        derived_text = "\n".join(block["text"] for block in blocks)
        if not deterministic_mode() and source:
            image_path = None
            if source["source_type"] == "image":
                image_path = data_dir() / source["local_path"]
            else:
                for block in blocks:
                    if block["frame_path"]:
                        image_path = data_dir() / block["frame_path"]
                        break
            if image_path and image_path.exists():
                try:
                    vlm = provider_registry().vlm()
                    observation = vlm.inspect_image(source_id, frame_or_image_id, image_path, question_for_vlm)
                    observations = [
                        {
                            "question": question_for_vlm,
                            "text": observation.visual_answer,
                            "confidence": observation.confidence,
                            "provider": observation.provider,
                            "model": observation.model,
                        }
                    ]
                    selected = observation
                    weak = visual_answer_is_weak(question_for_vlm, observation.visual_answer, observation.confidence)
                    if weak and self.state.vlm_calls < AGENT_LIMITS["max_vlm_calls"]:
                        refined_question = self.refine_visual_question(question_for_vlm, observation.visual_answer, derived_text)
                        self.state.vlm_calls += 1
                        follow_up = vlm.inspect_image(source_id, frame_or_image_id, image_path, refined_question)
                        observations.append(
                            {
                                "question": refined_question,
                                "text": follow_up.visual_answer,
                                "confidence": follow_up.confidence,
                                "provider": follow_up.provider,
                                "model": follow_up.model,
                            }
                        )
                        if not visual_answer_is_weak(refined_question, follow_up.visual_answer, follow_up.confidence) or follow_up.confidence >= observation.confidence:
                            selected = follow_up
                        self.state.trace.append(
                            {
                                "step": "vlm_follow_up",
                                "detail": "Weak visual observation triggered one refined VLM question.",
                                "source_id": source_id,
                                "initial_question": question_for_vlm,
                                "refined_question": refined_question,
                                "provider": follow_up.provider,
                                "model": follow_up.model,
                            }
                        )
                    conflict = visual_conflicts_with_text(selected.visual_answer, derived_text)
                    if conflict:
                        self.state.warnings.append("visual_conflict_detected")
                    self.state.trace.append(
                        {
                            "step": "vlm_inspection",
                            "detail": "Recorded VLM visual inspection observation.",
                            "source_id": source_id,
                            "questions": [item["question"] for item in observations],
                            "provider": selected.provider,
                            "model": selected.model,
                            "conflict_detected": conflict,
                        }
                    )
                    return {
                        "available": True,
                        "warning": "",
                        "question_for_vlm": question_for_vlm,
                        "text": selected.visual_answer,
                        "source_id": source_id,
                        "frame_or_image_id": frame_or_image_id,
                        "provider": selected.provider,
                        "model": selected.model,
                        "confidence": selected.confidence,
                        "observations": observations,
                        "follow_up_performed": len(observations) > 1,
                        "conflict_detected": conflict,
                    }
                except ProviderUnavailable as exc:
                    self.state.warnings.append(f"vlm_unavailable:{exc}")
        if not blocks:
            return {"available": False, "warning": "unsupported_visual_evidence", "text": ""}
        self.state.warnings.append("vlm_unavailable_used_derived_visual_evidence")
        return {
            "available": False,
            "warning": "vlm_unavailable_used_derived_visual_evidence",
            "question_for_vlm": question_for_vlm,
            "text": derived_text,
            "source_id": source_id,
            "frame_or_image_id": frame_or_image_id,
        }

    def refine_visual_question(self, original_question: str, weak_visual_answer: str, evidence_context: str) -> str:
        return f"{original_question} Focus only on visible labels, arrows, entities, and relationships mentioned in: {evidence_context[:240]}"

    def inspect_transcript(self, source_id: str, timestamp_start: float | None = None, timestamp_end: float | None = None) -> list[dict[str, Any]]:
        with connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM content_blocks
                WHERE project_id = ? AND source_id = ? AND block_type = 'transcript'
                """,
                (self.state.project_id, source_id),
            ).fetchall()
        return [dict(row) for row in rows]

    def lookup_entities(self, entity_name: str) -> list[dict[str, Any]]:
        self.state.graph_calls += 1
        if self.state.graph_calls > AGENT_LIMITS["max_graph_agent_calls"]:
            self.state.warnings.append("graph_agent_call_limit_reached")
            return []
        like = f"%{entity_name.lower()}%"
        with connect() as conn:
            rows = conn.execute(
                """
                SELECT e.*, COUNT(r.id) AS relationship_count
                FROM entities e
                LEFT JOIN relationships r
                  ON r.from_entity_id = e.id OR r.to_entity_id = e.id
                WHERE e.project_id = ? AND e.normalized_name LIKE ?
                GROUP BY e.id
                ORDER BY relationship_count DESC
                LIMIT 8
                """,
                (self.state.project_id, like),
            ).fetchall()
        return [dict(row) for row in rows]

    def evaluate_grounding(self, answer: str, citations: list[dict[str, Any]], evidence: list[RetrievedChunk]) -> dict[str, Any]:
        self.state.critic_passes += 1
        if self.state.critic_passes > AGENT_LIMITS["max_critic_passes"]:
            self.state.warnings.append("critic_pass_limit_reached")
            return {"recommended_next_action": "finalize_with_warning", "unsupported_claims_count": 1}
        if not deterministic_mode():
            try:
                report = structured_critic_report(self.state.question, answer, citations, evidence)
                report["critic_provider"] = provider_registry().llm().provider
                report["critic_model"] = provider_registry().llm().model
                return report
            except Exception as exc:
                self.state.warnings.append(f"llm_critic_fallback:{exc}")
        report = grounding_report(self.state.question, evidence)
        if not citations or report["insufficient_evidence"]:
            return {**report, "recommended_next_action": "insufficient_evidence"}
        return {**report, "recommended_next_action": "finalize"}

    def final_answer(self, answer: str, citations: list[dict[str, Any]]) -> dict[str, Any]:
        return {"answer": answer, "citations": citations, "produced_by": "Main Orchestrator"}


def chunks_to_items(chunks: list[RetrievedChunk]) -> list[EvidenceItem]:
    return [
        EvidenceItem(
            chunk_id=chunk.id,
            source_id=chunk.source_id,
            citation=chunk.citation,
            text=chunk.text[:500],
            score=chunk.score,
            metadata=chunk.metadata,
        )
        for chunk in chunks
    ]


def citations_from_chunks(chunks: list[RetrievedChunk]) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": chunk.id,
            "source_id": chunk.source_id,
            "label": chunk.citation,
            "score": chunk.score,
            "metadata": chunk.metadata,
        }
        for chunk in chunks
    ]


def question_terms(question: str) -> set[str]:
    return {term.lower().strip("?,.") for term in question.split() if len(term.strip("?,.")) > 2}


def visual_answer_is_weak(question: str, answer: str, confidence: float) -> bool:
    normalized = " ".join(answer.lower().split())
    if confidence < 0.45:
        return True
    if len(tokenize(answer)) < 5:
        return True
    if any(phrase in normalized for phrase in WEAK_VISUAL_PHRASES):
        return True
    overlap = set(tokenize(question)).intersection(tokenize(answer))
    return bool(tokenize(question)) and len(overlap) == 0


def visual_conflicts_with_text(visual_answer: str, evidence_text: str) -> bool:
    visual_terms = {term for term in tokenize(visual_answer) if len(term) > 3}
    evidence_terms = {term for term in tokenize(evidence_text) if len(term) > 3}
    if len(visual_terms) < 3 or len(evidence_terms) < 3:
        return False
    return len(visual_terms.intersection(evidence_terms)) == 0


def structured_orchestrator_plan(question: str, available_tools: list[str]) -> dict[str, Any]:
    schema = {
        "actions": [{"tool": "string", "reason": "string"}],
        "rationale": "string",
    }
    prompt = (
        "You are the main Agentic RAG orchestrator. Choose only bounded tools from the allowed list. "
        "Specialist tools return evidence only; only the main orchestrator can produce the final answer.\n"
        f"Question: {question}\nAllowed tools: {', '.join(available_tools)}\n"
        "Return a compact JSON plan with actions in execution order."
    )
    plan = provider_registry().llm().generate_structured(prompt, schema)
    actions = []
    for action in plan.get("actions", []):
        tool = str(action.get("tool", "")).strip()
        if not tool:
            continue
        if tool not in available_tools:
            raise ValueError(f"Unsupported orchestrator tool: {tool}")
        actions.append({"tool": tool, "reason": str(action.get("reason", ""))})
    return {"actions": actions, "rationale": str(plan.get("rationale", ""))}


def structured_critic_report(question: str, answer: str, citations: list[dict[str, Any]], evidence: list[RetrievedChunk]) -> dict[str, Any]:
    schema = {
        "supported_claims": ["string"],
        "unsupported_claims": ["string"],
        "missing_evidence": ["string"],
        "conflicting_evidence": ["string"],
        "citation_quality": "string",
        "answer_sufficient": "boolean",
        "recommended_action": "finalize|revise|retrieve_more|ask_vlm_again|insufficient_evidence",
    }
    evidence_lines = "\n".join(f"- {chunk.text[:500]} [{chunk.citation}]" for chunk in evidence[:8])
    prompt = (
        "Evaluate whether this draft answer is grounded only in the evidence. "
        "Find unsupported claims, weak citations, missing evidence, and conflicts. "
        "Recommend exactly one action: finalize, revise, retrieve_more, ask_vlm_again, or insufficient_evidence.\n"
        f"Question: {question}\nAnswer: {answer}\nCitations: {citations}\nEvidence:\n{evidence_lines}"
    )
    report = provider_registry().llm().generate_structured(prompt, schema)
    action = str(report.get("recommended_action", "finalize"))
    allowed = {"finalize", "revise", "retrieve_more", "ask_vlm_again", "insufficient_evidence"}
    if action not in allowed:
        action = "insufficient_evidence"
    unsupported = report.get("unsupported_claims") or []
    missing = report.get("missing_evidence") or []
    conflicts = report.get("conflicting_evidence") or []
    sufficient = bool(report.get("answer_sufficient", action == "finalize"))
    return {
        "supported_claims": report.get("supported_claims") or [],
        "unsupported_claims": unsupported,
        "missing_evidence": missing,
        "conflicting_evidence": conflicts,
        "citation_quality": str(report.get("citation_quality", "unknown")),
        "answer_sufficient": sufficient,
        "recommended_next_action": action,
        "recommended_action": action,
        "unsupported_claims_count": len(unsupported),
        "insufficient_evidence": not sufficient or action == "insufficient_evidence",
        "grounding_score": 0.0 if not sufficient else 1.0,
    }


def should_use_visual_agent(project_id: str, question: str) -> bool:
    if question_terms(question).intersection(VISUAL_TERMS):
        return True
    with connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS count FROM sources WHERE project_id = ? AND source_type IN ('image', 'video')",
            (project_id,),
        ).fetchone()
    return row["count"] > 0 and bool(question_terms(question).intersection({"architecture", "screen", "label", "shown"}))


def should_use_graph_agent(question: str) -> bool:
    return bool(question_terms(question).intersection(GRAPH_TERMS))


def retrieval_agent(state: AgentState, registry: ToolRegistry) -> tuple[EvidencePackage, list[RetrievedChunk]]:
    state.specialists_used.append("Retrieval Agent")
    vector_hits = registry.call("search_vector_store", query=state.question, filters=state.filters, top_k=8)
    inspected_chunks = []
    for chunk in vector_hits[:3]:
        inspected = registry.call("inspect_chunk", chunk_id=chunk.id)
        if inspected:
            inspected_chunks.append({"chunk_id": chunk.id, "source_id": chunk.source_id, "citation": chunk.citation})
    state.trace.append(
        {
            "step": "inspect_retrieved_chunks",
            "detail": f"Inspected {len(inspected_chunks)} top vector hits before orchestration.",
            "inspected_chunks": inspected_chunks,
        }
    )
    reranked = registry.call("rerank_evidence", question=state.question, retrieved_chunks=vector_hits)
    package = EvidencePackage(
        agent_name="Retrieval Agent",
        task="Use bounded vector-store search, then inspect top chunks before handing evidence to the orchestrator.",
        query_used=state.question,
        evidence_items=chunks_to_items(reranked),
        citations=citations_from_chunks(reranked),
        confidence=max((chunk.score for chunk in reranked), default=0.0),
        gaps=[] if reranked else ["No grounded retrieval evidence found."],
        conflicts=[],
        recommended_next_action="finalize" if reranked else "insufficient_evidence",
    )
    return package, reranked


def visual_agent(state: AgentState, registry: ToolRegistry, retrieval_context: list[RetrievedChunk]) -> tuple[EvidencePackage, list[RetrievedChunk]]:
    state.specialists_used.append("Visual Agent")
    visual_chunks = retrieve(state.project_id, state.question, top_k=4, metadata_filters={**state.filters, "source_type": ["image", "video"]})
    visual_sources = []
    for chunk in visual_chunks:
        if chunk.source_id not in visual_sources:
            visual_sources.append(chunk.source_id)
    observations: list[EvidenceItem] = []
    for source_id in visual_sources[: AGENT_LIMITS["max_vlm_calls"]]:
        observation = registry.call(
            "inspect_visual_source",
            source_id=source_id,
            frame_or_image_id=None,
            question_for_vlm=state.question,
        )
        if observation.get("text"):
            observations.append(
                EvidenceItem(
                    source_id=source_id,
                    citation=f"visual-observation:{source_id}",
                    text=observation["text"],
                    score=0.5,
                    metadata={"warning": observation.get("warning"), "question_for_vlm": state.question},
                )
            )
    gaps = []
    if not visual_chunks and not observations:
        gaps.append("No supported visual evidence found.")
    if observations and "vlm_unavailable_used_derived_visual_evidence" in state.warnings:
        gaps.append("Real VLM unavailable; used derived OCR/caption/transcript evidence.")
    package = EvidencePackage(
        agent_name="Visual Agent",
        task="Inspect image or video evidence as a specialist tool.",
        query_used=state.question,
        evidence_items=[*chunks_to_items(visual_chunks), *observations],
        citations=[*citations_from_chunks(visual_chunks), *[item.model_dump() for item in observations]],
        confidence=max((chunk.score for chunk in visual_chunks), default=0.5 if observations else 0.0),
        gaps=gaps,
        conflicts=[],
        recommended_next_action="finalize" if visual_chunks or observations else "insufficient_evidence",
    )
    return package, visual_chunks


def graph_agent(state: AgentState, registry: ToolRegistry) -> EvidencePackage:
    state.specialists_used.append("Graph Agent")
    entities = []
    for term in list(question_terms(state.question))[:1]:
        entities.extend(registry.call("lookup_entities", entity_name=term))
    diagnostics = graph_diagnostics(state.project_id)
    items = [
        EvidenceItem(
            text=f"Entity {entity['name']} has {entity['relationship_count']} graph relationships.",
            score=min(1.0, 0.2 + entity["relationship_count"] * 0.1),
            metadata=entity,
        )
        for entity in entities[:8]
    ]
    return EvidencePackage(
        agent_name="Graph Agent",
        task="Check whether graph entities or relationships can support the answer.",
        query_used=state.question,
        evidence_items=items,
        citations=[],
        confidence=max((item.score for item in items), default=0.0),
        gaps=[] if items else ["No relevant graph entities found."],
        conflicts=[],
        recommended_next_action="finalize" if items else "continue_without_graph",
    ).model_copy(update={"citations": [{"graph_diagnostics": diagnostics}] if items else []})


def critic_agent(state: AgentState, registry: ToolRegistry, answer: str, citations: list[dict[str, Any]], evidence: list[RetrievedChunk]) -> EvidencePackage:
    state.specialists_used.append("Critic / Grounding Agent")
    report = registry.call("evaluate_grounding", answer=answer, citations=citations, evidence=evidence)
    gaps = []
    if report.get("insufficient_evidence"):
        gaps.append("Answer has insufficient source evidence.")
    if report.get("unsupported_claims_count", 0):
        gaps.append("Unsupported claims were detected or could not be ruled out.")
    return EvidencePackage(
        agent_name="Critic / Grounding Agent",
        task="Evaluate grounding, citation quality, and next action.",
        query_used=state.question,
        evidence_items=[],
        citations=citations,
        confidence=report.get("grounding_score", 0.0),
        gaps=gaps,
        conflicts=[],
        recommended_next_action=report["recommended_next_action"],
    )


def run_agentic_pipeline(
    project_id: str,
    question: str,
    filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    state = AgentState(project_id=project_id, question=question, filters=filters or {})
    registry = ToolRegistry(state)

    state.step(
        "orchestrator_plan",
        "Planned bounded specialist investigation.",
        tool_registry=registry.names,
        limits=AGENT_LIMITS,
    )
    llm_plan: dict[str, Any] | None = None
    planned_tools: set[str] = set()
    if not deterministic_mode():
        try:
            llm_plan = structured_orchestrator_plan(question, registry.names)
            planned_tools = {action["tool"] for action in llm_plan["actions"]}
            state.trace.append(
                {
                    "step": "llm_orchestrator_plan",
                    "detail": "Real LLM orchestrator selected bounded tool actions.",
                    "plan": llm_plan,
                }
            )
        except Exception as exc:
            state.warnings.append(f"llm_orchestrator_fallback:{exc}")

    packages: list[EvidencePackage] = []
    retrieval_package, retrieved_chunks = retrieval_agent(state, registry)
    packages.append(retrieval_package)
    state.step("dispatch_retrieval_agent", "Retrieval Agent returned an evidence package.", package=retrieval_package.model_dump())

    visual_chunks: list[RetrievedChunk] = []
    visual_requested = "inspect_visual_source" in planned_tools if llm_plan is not None else should_use_visual_agent(project_id, question)
    if len(packages) < AGENT_LIMITS["max_specialist_agents"] and visual_requested:
        visual_package, visual_chunks = visual_agent(state, registry, retrieved_chunks)
        packages.append(visual_package)
        state.step("dispatch_visual_agent_if_needed", "Visual Agent returned an evidence package.", package=visual_package.model_dump())

    graph_requested = "lookup_entities" in planned_tools if llm_plan is not None else should_use_graph_agent(question)
    if len(packages) < AGENT_LIMITS["max_specialist_agents"] and graph_requested:
        graph_package = graph_agent(state, registry)
        packages.append(graph_package)
        state.step("dispatch_graph_agent_if_needed", "Graph Agent returned an evidence package.", package=graph_package.model_dump())

    evidence = rerank_evidence(question, [*retrieved_chunks, *visual_chunks], top_k=AGENT_LIMITS["max_chunks_in_context"])
    answer, citations, answer_warnings, report = synthesize_answer(question, evidence, answer_mode="agentic")
    state.warnings.extend(warning for warning in answer_warnings if warning not in state.warnings)

    critic_package = critic_agent(state, registry, answer, citations, evidence)
    packages.append(critic_package)
    state.step("critic_grounding_check", "Critic checked evidence and citations.", package=critic_package.model_dump())

    if critic_package.recommended_next_action == "insufficient_evidence":
        answer = "I could not find enough evidence in the uploaded sources to answer this question."
        citations = []
        if "insufficient_evidence" not in state.warnings:
            state.warnings.append("insufficient_evidence")

    final = registry.call("final_answer", answer=answer, citations=citations)
    state.step("finalize_answer", "Main Orchestrator produced the final answer.", final_answer=final)

    state.evidence_packages = packages
    return {
        "pipeline_type": "agentic",
        "answer": final["answer"],
        "citations": final["citations"],
        "trace": state.trace,
        "metrics": {
            "citations_count": len(final["citations"]),
            "chunks_used": len(final["citations"]),
            "warnings_count": len(state.warnings),
            "retrieval_calls": state.retrieval_calls,
            "vlm_calls": state.vlm_calls,
            "graph_agent_calls": state.graph_calls,
            "critic_passes": state.critic_passes,
            "tool_calls": state.tool_calls,
            "agent_steps": min(state.steps, AGENT_LIMITS["max_agent_steps"]),
            "specialist_agents_used": len({name for name in state.specialists_used if name != "Critic / Grounding Agent"}),
            "grounding_score": report["grounding_score"],
            "unsupported_claims_count": report["unsupported_claims_count"],
            "insufficient_evidence_flag": "insufficient_evidence" in state.warnings,
            "evidence_packages_count": len(packages),
            "prompt_token_estimate": len(question_terms(question)) + sum(len(item.text.split()) for package in packages for item in package.evidence_items),
            "answer_length": len(final["answer"]),
            "source_coverage_count": len({citation.get("source_id") for citation in final["citations"] if citation.get("source_id")}),
            "collection_strategy": "bounded_agent_tool_collection",
            "data_collection_techniques": [
                "orchestrator_planned_tools",
                "vector_tool_search",
                "chunk_inspection",
                "specialist_evidence_packages",
                "critic_review",
            ],
        },
        "techniques": [
            "agentic_orchestrator",
            "llm_orchestrator" if llm_plan is not None else "rule_based_orchestrator",
            "bounded_tool_registry",
            "retrieval_agent",
            "visual_agent",
            "graph_agent",
            "grounding_critic",
            "llm_critic" if not deterministic_mode() else "rule_based_critic",
        ],
        "warnings": state.warnings,
    }
