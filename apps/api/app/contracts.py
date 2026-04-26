from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field, field_validator

from .providers import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OLLAMA_LLM_MODEL,
    DEFAULT_OLLAMA_VLM_MODEL,
    DEFAULT_TRANSCRIPTION_MODEL,
)


PROMPT_TEMPLATE_IDS = [
    "traditional_answer_v1",
    "agentic_orchestrator_v1",
    "retrieval_agent_v1",
    "visual_agent_v1",
    "graph_agent_v1",
    "grounding_critic_v1",
    "hybrid_answer_v1",
    "comparison_recommender_v1",
]


def load_prompt_registry() -> dict[str, str]:
    prompt_dir = Path(__file__).parent / "prompts"
    registry: dict[str, str] = {}
    for prompt_id in PROMPT_TEMPLATE_IDS:
        path = prompt_dir / f"{prompt_id}.txt"
        registry[prompt_id] = path.read_text(encoding="utf-8").strip() if path.exists() else prompt_id
    return registry


PROMPT_REGISTRY = load_prompt_registry()

PIPELINE_PROMPTS = {
    "traditional": {"answer": "traditional_answer_v1", "critic": "grounding_critic_v1"},
    "agentic": {
        "orchestrator": "agentic_orchestrator_v1",
        "retrieval": "retrieval_agent_v1",
        "visual": "visual_agent_v1",
        "graph": "graph_agent_v1",
        "critic": "grounding_critic_v1",
    },
    "hybrid_graph": {"answer": "hybrid_answer_v1", "graph": "graph_agent_v1", "critic": "grounding_critic_v1"},
}


class RunConfig(BaseModel):
    llm_provider: str = "ollama"
    llm_model: str = DEFAULT_OLLAMA_LLM_MODEL
    vlm_provider: str = "ollama"
    vlm_model: str = DEFAULT_OLLAMA_VLM_MODEL
    embedding_provider: str = "ollama"
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    transcription_provider: str = "faster-whisper"
    transcription_model: str = DEFAULT_TRANSCRIPTION_MODEL
    vector_db_provider: str = "chroma"
    chunk_size: int = 140
    chunk_overlap: int = 25
    top_k: int = 8
    reranker_enabled: bool = True
    graph_enabled: bool = True
    max_agent_steps: int = 6
    max_tool_calls: int = 8
    prompt_template_versions: dict[str, str] = Field(default_factory=lambda: dict(PROMPT_REGISTRY))
    knowledge_base_version_id: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class VisualObservation(BaseModel):
    source_id: str
    image_or_frame_id: str | None = None
    question: str
    answer: str
    visible_text: list[str] = Field(default_factory=list)
    described_objects: list[str] = Field(default_factory=list)
    relevant_regions: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    limitations: list[str] = Field(default_factory=list)
    needs_follow_up: bool = False


class GroundingReport(BaseModel):
    supported_claims: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    conflicting_evidence: list[str] = Field(default_factory=list)
    citation_quality: str
    answer_sufficient: bool
    recommended_action: str

    @field_validator("recommended_action")
    @classmethod
    def known_action(cls, value: str) -> str:
        allowed = {"finalize", "revise", "retrieve_more", "ask_vlm_again", "insufficient_evidence"}
        if value not in allowed:
            raise ValueError(f"recommended_action must be one of {sorted(allowed)}")
        return value


class LLMAdapter(Protocol):
    def generate(self, prompt: str) -> Any: ...
    def generate_structured(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]: ...


class VLMAdapter(Protocol):
    def inspect_image(self, source_id: str, frame_or_image_id: str | None, image_path: Any, question: str) -> Any: ...


class EmbeddingAdapter(Protocol):
    def embed_text(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


class RetrieverAdapter(Protocol):
    def vector(self, query: str, filters: dict[str, Any], top_k: int) -> list[Any]: ...
    def keyword(self, query: str, filters: dict[str, Any], top_k: int) -> list[Any]: ...
    def graph_expand(self, entity_name: str) -> list[Any]: ...


class EvaluatorAdapter(Protocol):
    def rerank(self, question: str, evidence: list[Any]) -> list[Any]: ...
    def evaluate_grounding(self, answer: str, citations: list[dict[str, Any]], evidence: list[Any]) -> GroundingReport: ...
    def evaluate_visual_observation(self, question: str, observation: VisualObservation) -> GroundingReport: ...


def default_run_config(knowledge_base_version_id: str | None) -> RunConfig:
    return RunConfig(knowledge_base_version_id=knowledge_base_version_id)


def prompt_versions_for_pipeline(pipeline_type: str) -> dict[str, str]:
    return PIPELINE_PROMPTS[pipeline_type]


def validate_answer_contract(answer: str, citations: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    if "Based on the uploaded evidence" in answer and not citations:
        warnings.append("answer_claims_without_citations")
    if "I could not find enough evidence" not in answer and not citations:
        warnings.append("missing_citations")
    return warnings
