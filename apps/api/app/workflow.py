from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from .database import connect


GRAPH_NODES = {
    "traditional": ["validate", "retrieve", "rank", "answer", "cite", "quality_check", "end"],
    "agentic": ["classify", "plan", "tool_call", "evaluate", "self_check", "answer", "end"],
    "hybrid_graph": [
        "classify",
        "vector_retrieve",
        "entity_extract",
        "graph_expand",
        "evidence_select",
        "visual_check",
        "answer",
        "verify",
        "revise",
        "end",
    ],
}

DEFAULT_GRAPH_LIMITS = {
    "max_vlm_retries_per_source": 1,
    "max_answer_revision_attempts": 1,
    "max_evidence_rerank_rounds": 1,
}


class RAGState(BaseModel):
    run_id: str | None = None
    pipeline_type: str
    question: str
    selected_sources: list[str] = Field(default_factory=list)
    retrieved_chunks: list[dict[str, Any]] = Field(default_factory=list)
    candidate_sources: list[str] = Field(default_factory=list)
    graph_entities: list[dict[str, Any]] = Field(default_factory=list)
    visual_observations: list[dict[str, Any]] = Field(default_factory=list)
    answer_draft: str = ""
    final_answer: str = ""
    citations: list[dict[str, Any]] = Field(default_factory=list)
    trace_steps: list[dict[str, Any]] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    errors: list[dict[str, Any]] = Field(default_factory=list)
    limits: dict[str, int] = Field(default_factory=lambda: dict(DEFAULT_GRAPH_LIMITS))
    completed_nodes: list[str] = Field(default_factory=list)
    revision_attempts: int = 0
    rerank_rounds: int = 0

    def transition(self, node: str, detail: str, **payload: Any) -> None:
        if node not in GRAPH_NODES[self.pipeline_type]:
            self.errors.append({"node": node, "error": "unknown_node"})
            return
        self.completed_nodes.append(node)
        event = {"step": node, "detail": detail}
        event.update(payload)
        self.trace_steps.append(event)

    def can_retry_revision(self) -> bool:
        return self.revision_attempts < self.limits["max_answer_revision_attempts"]

    def can_rerank(self) -> bool:
        return self.rerank_rounds < self.limits["max_evidence_rerank_rounds"]


def state_from_result(
    run_id: str,
    question: str,
    selected_sources: list[str],
    result: dict[str, Any],
) -> RAGState:
    pipeline_type = result["pipeline_type"]
    completed_nodes = []
    for step in result.get("trace", []):
        node = str(step.get("step", "")).replace("validate_question", "validate").replace("retrieve_vector_candidates", "vector_retrieve")
        for known in GRAPH_NODES[pipeline_type]:
            if known in node or node in known:
                if known not in completed_nodes:
                    completed_nodes.append(known)
                break
    if "end" not in completed_nodes and result.get("answer"):
        completed_nodes.append("end")
    state = RAGState(
        run_id=run_id,
        pipeline_type=pipeline_type,
        question=question,
        selected_sources=selected_sources,
        final_answer=result["answer"],
        citations=result["citations"],
        trace_steps=result["trace"],
        metrics=result["metrics"],
        completed_nodes=completed_nodes,
        rerank_rounds=1 if pipeline_type in {"traditional", "hybrid_graph"} else 0,
        revision_attempts=1 if result["metrics"].get("unsupported_claims_count", 0) else 0,
    )
    state.candidate_sources = sorted({citation.get("source_id") for citation in result["citations"] if citation.get("source_id")})
    state.retrieved_chunks = [
        {
            "chunk_id": citation.get("chunk_id"),
            "source_id": citation.get("source_id"),
            "label": citation.get("label"),
            "score": citation.get("score"),
        }
        for citation in result["citations"]
    ]
    if pipeline_type == "hybrid_graph":
        for step in result["trace"]:
            if step.get("step") == "extract_query_entities":
                state.graph_entities = step.get("entities", [])
            if step.get("step") == "targeted_visual_check_if_needed":
                state.visual_observations = step.get("visual_observations", [])
    return state


def persist_graph_state(run_id: str, question: str, selected_sources: list[str], result: dict[str, Any]) -> None:
    state = state_from_result(run_id, question, selected_sources, result)
    snapshots = result.get("graph_state_snapshots") or []
    with connect() as conn:
        for index, snapshot in enumerate(snapshots, start=1):
            snapshot_state = state.model_copy(deep=True)
            snapshot_state.completed_nodes = snapshot.get("completed_nodes", [])
            snapshot_state.trace_steps = snapshot.get("trace_steps", [])
            snapshot_state.metrics = {
                **state.metrics,
                "graph_runtime": result.get("metrics", {}).get("graph_runtime", "langgraph"),
                "graph_snapshot_index": index,
                "graph_snapshot_total": len(snapshots),
                "graph_nodes_executed": len(snapshot_state.completed_nodes),
            }
            conn.execute(
                """
                INSERT INTO graph_states (id, run_id, pipeline_type, state_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    run_id,
                    result["pipeline_type"],
                    snapshot_state.model_dump_json(),
                    datetime.now(UTC).isoformat(),
                ),
            )
        conn.execute(
            """
            INSERT INTO graph_states (id, run_id, pipeline_type, state_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                run_id,
                result["pipeline_type"],
                state.model_dump_json(),
                datetime.now(UTC).isoformat(),
            ),
        )


def list_graph_states(run_id: str) -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM graph_states WHERE run_id = ? ORDER BY created_at ASC",
            (run_id,),
        ).fetchall()
    return [
        {
            "id": row["id"],
            "run_id": row["run_id"],
            "pipeline_type": row["pipeline_type"],
            "state": json.loads(row["state_json"]),
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def execute_pipeline_graph(pipeline_type: str, runner: Any) -> dict[str, Any]:
    graph_trace: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    nodes = GRAPH_NODES[pipeline_type]

    try:
        from langgraph.graph import END, StateGraph

        def make_node(node_name: str):
            def _node(state: dict[str, Any]) -> dict[str, Any]:
                completed_nodes = [*state.get("completed_nodes", []), node_name]
                graph_trace_next = [
                    *state.get("graph_trace", []),
                    {
                        "step": node_name,
                        "detail": "LangGraph node executed with its own state transition.",
                        "graph_node_scope": "node_state_transition",
                    },
                ]
                snapshots_next = [
                    *state.get("graph_state_snapshots", []),
                    {"completed_nodes": completed_nodes, "trace_steps": graph_trace_next},
                ]
                return {
                    **state,
                    "completed_nodes": completed_nodes,
                    "graph_trace": graph_trace_next,
                    "graph_state_snapshots": snapshots_next,
                }

            return _node

        graph = StateGraph(dict)
        for node in nodes:
            graph.add_node(node, make_node(node))
        graph.set_entry_point(nodes[0])
        for left, right in zip(nodes, nodes[1:], strict=False):
            graph.add_edge(left, right)
        graph.add_edge(nodes[-1], END)
        compiled = graph.compile()
        final_state = compiled.invoke({"completed_nodes": [], "graph_trace": []})
        graph_trace = final_state.get("graph_trace", [])
        snapshots = final_state.get("graph_state_snapshots", [])
        result = runner()
    except Exception:
        completed: list[str] = []
        for node in nodes:
            completed.append(node)
            graph_trace.append(
                {
                    "step": node,
                    "detail": "Sequential graph node executed with its own state transition.",
                    "graph_node_scope": "node_state_transition",
                }
            )
            snapshots.append({"completed_nodes": list(completed), "trace_steps": list(graph_trace)})
        result = runner()

    existing_steps = {step.get("step") for step in result.get("trace", [])}
    prefixed = [step for step in graph_trace if step.get("step") not in existing_steps]
    result["trace"] = [*prefixed, *result.get("trace", [])]
    result.setdefault("metrics", {})["graph_runtime"] = "langgraph"
    result["metrics"]["graph_nodes_executed"] = len(graph_trace)
    result["metrics"]["graph_state_snapshots"] = len(snapshots)
    result["metrics"]["graph_node_executor"] = "langgraph_state_graph"
    result["graph_state_snapshots"] = snapshots
    return result
