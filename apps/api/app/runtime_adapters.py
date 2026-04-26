from __future__ import annotations

import os
import re
from collections import defaultdict
from typing import Any, Protocol

from .rag import cosine_similarity, deterministic_vector, tokenize


class VectorStoreAdapter(Protocol):
    provider: str

    def upsert(self, records: list[dict[str, Any]]) -> None:
        ...

    def query(self, text: str, top_k: int = 5) -> list[dict[str, Any]]:
        ...

    def delete(self, ids: list[str]) -> None:
        ...

    def delete_source(self, source_id: str) -> None:
        ...

    def delete_project(self) -> None:
        ...


def _collection_name(project_id: str, kb_version: str | None = None) -> str:
    suffix = kb_version or "active"
    safe = re.sub(r"[^A-Za-z0-9_-]+", "-", f"APRAG-Lab-{project_id}-{suffix}")[:60].strip("-")
    return safe or "APRAG-Lab-active"


def vector_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    return {
        key: value
        for key, value in metadata.items()
        if isinstance(value, str | int | float | bool) and value is not None
    }


class InMemoryVectorStore:
    provider = "chroma"

    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}

    def upsert(self, records: list[dict[str, Any]]) -> None:
        for record in records:
            stored = dict(record)
            stored.setdefault("vector", deterministic_vector(stored.get("text", "")))
            self.records[stored["id"]] = stored

    def query(self, text: str, top_k: int = 5) -> list[dict[str, Any]]:
        query_vector = deterministic_vector(text)
        ranked = []
        for record in self.records.values():
            score = cosine_similarity(query_vector, record["vector"])
            ranked.append({**record, "score": round(score, 4)})
        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked[:top_k]

    def delete(self, ids: list[str]) -> None:
        for record_id in ids:
            self.records.pop(record_id, None)

    def delete_source(self, source_id: str) -> None:
        for record_id, record in list(self.records.items()):
            if record.get("source_id") == source_id or record.get("metadata", {}).get("source_id") == source_id:
                self.records.pop(record_id, None)

    def delete_project(self) -> None:
        self.records.clear()


class ChromaVectorStoreAdapter(InMemoryVectorStore):
    provider = "chroma"

    def __init__(self, project_id: str | None = None, kb_version: str | None = None, embedding_fn: Any | None = None) -> None:
        super().__init__()
        self.project_id = project_id or "default"
        self.kb_version = kb_version
        self.embedding_fn = embedding_fn
        self.collection_name = _collection_name(self.project_id, self.kb_version)
        self.client = None
        self.collection = None
        try:
            import chromadb

            host = os.environ.get("CHROMA_HOST")
            if host:
                self.client = chromadb.HttpClient(host=host, port=int(os.environ.get("CHROMA_PORT", "8000")))
            else:
                from .database import data_dir

                self.client = chromadb.PersistentClient(path=str(data_dir() / "vector_store" / "chroma"))
            self.collection = self.client.get_or_create_collection(self.collection_name)
        except Exception:
            self.client = None
            self.collection = None

    def upsert(self, records: list[dict[str, Any]]) -> None:
        if not self.collection:
            return super().upsert(records)
        ids = [record["id"] for record in records]
        documents = [record.get("text", "") for record in records]
        metadatas = [
            vector_metadata(record.get("metadata", {}) | {"source_id": record.get("source_id", ""), "chunk_id": record.get("chunk_id", record["id"])})
            for record in records
        ]
        embeddings = [record.get("vector") or deterministic_vector(record.get("text", "")) for record in records]
        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

    def query(self, text: str, top_k: int = 5, vector: list[float] | None = None, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if not self.collection:
            return super().query(text, top_k)
        query_vector = vector or deterministic_vector(text)
        where = None
        if filters and filters.get("source_ids"):
            source_ids = filters["source_ids"]
            source_ids = source_ids if isinstance(source_ids, list) else [source_ids]
            where = {"source_id": {"$in": source_ids}}
        result = self.collection.query(query_embeddings=[query_vector], n_results=top_k, where=where)
        rows: list[dict[str, Any]] = []
        for index, record_id in enumerate(result.get("ids", [[]])[0]):
            metadata = result.get("metadatas", [[]])[0][index] or {}
            distance = result.get("distances", [[]])[0][index] if result.get("distances") else 0.0
            rows.append(
                {
                    "id": record_id,
                    "text": result.get("documents", [[]])[0][index],
                    "metadata": metadata,
                    "source_id": metadata.get("source_id", ""),
                    "chunk_id": metadata.get("chunk_id", record_id),
                    "score": round(1.0 / (1.0 + float(distance)), 4),
                }
            )
        return rows

    def delete(self, ids: list[str]) -> None:
        if not self.collection:
            return super().delete(ids)
        if ids:
            self.collection.delete(ids=ids)

    def delete_source(self, source_id: str) -> None:
        if not self.collection:
            return super().delete_source(source_id)
        self.collection.delete(where={"source_id": source_id})

    def delete_project(self) -> None:
        if not self.client:
            return super().delete_project()
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass


class QdrantVectorStoreAdapter(InMemoryVectorStore):
    provider = "qdrant"

    def __init__(self, url: str | None = None, project_id: str | None = None, kb_version: str | None = None) -> None:
        super().__init__()
        self.url = url or os.environ.get("QDRANT_URL", "http://qdrant:6333")
        self.project_id = project_id or "default"
        self.kb_version = kb_version
        self.collection_name = _collection_name(self.project_id, self.kb_version)
        self.client = None
        try:
            from qdrant_client import QdrantClient

            self.client = QdrantClient(url=self.url, timeout=3)
        except Exception:
            self.client = None

    def _ensure_collection(self, size: int) -> None:
        if not self.client:
            return
        from qdrant_client.http.models import Distance, VectorParams

        try:
            existing = [collection.name for collection in self.client.get_collections().collections]
            if self.collection_name not in existing:
                self.client.create_collection(self.collection_name, vectors_config=VectorParams(size=size, distance=Distance.COSINE))
        except Exception:
            self.client = None

    def upsert(self, records: list[dict[str, Any]]) -> None:
        if not self.client:
            return super().upsert(records)
        from qdrant_client.http.models import PointStruct

        if not records:
            return
        vectors = [record.get("vector") or deterministic_vector(record.get("text", "")) for record in records]
        self._ensure_collection(len(vectors[0]))
        if not self.client:
            return super().upsert(records)
        points = [
            PointStruct(
                id=record["id"],
                vector=vector,
                payload=vector_metadata(
                    record.get("metadata", {})
                    | {
                        "text": record.get("text", ""),
                        "source_id": record.get("source_id", ""),
                        "chunk_id": record.get("chunk_id", record["id"]),
                    }
                ),
            )
            for record, vector in zip(records, vectors, strict=True)
        ]
        try:
            self.client.upsert(collection_name=self.collection_name, points=points)
        except Exception:
            self.client = None
            super().upsert(records)

    def query(self, text: str, top_k: int = 5, vector: list[float] | None = None, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if not self.client:
            return super().query(text, top_k)
        from qdrant_client.http.models import FieldCondition, Filter, MatchAny

        query_filter = None
        if filters and filters.get("source_ids"):
            source_ids = filters["source_ids"]
            source_ids = source_ids if isinstance(source_ids, list) else [source_ids]
            query_filter = Filter(must=[FieldCondition(key="source_id", match=MatchAny(any=source_ids))])
        try:
            hits = self.client.search(
                collection_name=self.collection_name,
                query_vector=vector or deterministic_vector(text),
                query_filter=query_filter,
                limit=top_k,
            )
        except Exception:
            return []
        return [
            {
                "id": str(hit.id),
                "text": hit.payload.get("text", ""),
                "metadata": hit.payload,
                "source_id": hit.payload.get("source_id", ""),
                "chunk_id": hit.payload.get("chunk_id", str(hit.id)),
                "score": round(float(hit.score), 4),
            }
            for hit in hits
        ]

    def delete(self, ids: list[str]) -> None:
        if not self.client:
            return super().delete(ids)
        self.client.delete(collection_name=self.collection_name, points_selector=ids)

    def delete_source(self, source_id: str) -> None:
        if not self.client:
            return super().delete_source(source_id)
        from qdrant_client.http.models import FieldCondition, Filter, MatchValue

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(must=[FieldCondition(key="source_id", match=MatchValue(value=source_id))]),
        )

    def delete_project(self) -> None:
        if not self.client:
            return super().delete_project()
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass


def selected_vector_store(project_id: str, kb_version: str | None = None) -> VectorStoreAdapter:
    name = os.environ.get("APRAG_VECTOR_STORE", "chroma").strip().lower()
    if name == "qdrant":
        return QdrantVectorStoreAdapter(project_id=project_id, kb_version=kb_version)
    if name == "sqlite":
        return InMemoryVectorStore()
    return ChromaVectorStoreAdapter(project_id=project_id, kb_version=kb_version)


class MetadataRepository(Protocol):
    provider: str

    def create_project(self, project_id: str, name: str) -> dict[str, Any]:
        ...

    def get_project(self, project_id: str) -> dict[str, Any] | None:
        ...

    def list_projects(self) -> list[dict[str, Any]]:
        ...


class InMemoryMetadataRepository:
    provider = "sqlite"

    def __init__(self) -> None:
        self.projects: dict[str, dict[str, Any]] = {}

    def create_project(self, project_id: str, name: str) -> dict[str, Any]:
        self.projects[project_id] = {"id": project_id, "name": name}
        return self.projects[project_id]

    def get_project(self, project_id: str) -> dict[str, Any] | None:
        return self.projects.get(project_id)

    def list_projects(self) -> list[dict[str, Any]]:
        return list(self.projects.values())


class SQLiteMetadataRepository(InMemoryMetadataRepository):
    provider = "sqlite"


class GraphStoreAdapter(Protocol):
    provider: str

    def add_relationship(self, left: str, right: str, evidence: str) -> None:
        ...

    def expand(self, entity: str) -> list[dict[str, str]]:
        ...


class InMemoryGraphStore:
    provider = "sqlite_graph"

    def __init__(self) -> None:
        self.edges: dict[str, list[dict[str, str]]] = defaultdict(list)

    def add_relationship(self, left: str, right: str, evidence: str) -> None:
        edge = {"from": left, "to": right, "evidence": evidence}
        self.edges[left.lower()].append(edge)
        self.edges[right.lower()].append(edge)

    def expand(self, entity: str) -> list[dict[str, str]]:
        return self.edges.get(entity.lower(), [])


class SQLiteGraphStoreAdapter(InMemoryGraphStore):
    provider = "sqlite_graph"


class NetworkXGraphStoreAdapter(InMemoryGraphStore):
    provider = "networkx"

    def __init__(self) -> None:
        super().__init__()
        try:
            import networkx as nx
        except Exception:
            self.graph = None
        else:
            self.graph = nx.MultiGraph()

    def add_relationship(self, left: str, right: str, evidence: str) -> None:
        super().add_relationship(left, right, evidence)
        if self.graph is None:
            return
        self.graph.add_node(left.lower(), label=left)
        self.graph.add_node(right.lower(), label=right)
        self.graph.add_edge(left.lower(), right.lower(), evidence=evidence, relationship_type="co_occurs_with")

    def expand(self, entity: str) -> list[dict[str, str]]:
        key = entity.lower()
        if self.graph is None:
            return super().expand(entity)
        if key not in self.graph:
            return []
        rows = []
        for neighbor in self.graph.neighbors(key):
            for edge in self.graph.get_edge_data(key, neighbor).values():
                rows.append({"from": self.graph.nodes[key]["label"], "to": self.graph.nodes[neighbor]["label"], "evidence": edge.get("evidence", "")})
        return rows


def normalize_ocr_output(provider: str, text: str, confidence: float, regions: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {"provider": provider, "text": text, "confidence": confidence, "regions": regions or [], "warnings": []}


def normalize_transcript_output(provider: str, segments: list[dict[str, Any]]) -> dict[str, Any]:
    normalized = [
        {
            "text": segment.get("text", ""),
            "timestamp_start": float(segment.get("timestamp_start", segment.get("start", 0.0))),
            "timestamp_end": float(segment.get("timestamp_end", segment.get("end", 0.0))),
            "confidence": float(segment.get("confidence", 0.0)),
        }
        for segment in segments
    ]
    return {"provider": provider, "segments": normalized, "text": " ".join(segment["text"] for segment in normalized).strip()}


def adapter_status() -> dict[str, Any]:
    from .provider_extensions import extension_settings

    status = {
        "vector_stores": {
            "chroma": {"provider": "chroma", "available": True, "container": "api"},
            "qdrant": {"provider": "qdrant", "available": False, "container": "qdrant", "url": os.environ.get("QDRANT_URL", "http://qdrant:6333")},
        },
        "metadata_stores": {
            "sqlite": {"provider": "sqlite", "available": True, "container": "api", "required": True},
            "postgres": {"provider": "postgres", "available": False, "required": False, "removed_from_required_scope": True},
        },
        "graph_stores": {
            "sqlite_graph": {"provider": "sqlite_graph", "available": True, "container": "api"},
            "networkx": {"provider": "networkx", "available": True, "container": "api"},
        },
        "runtimes": {
            "ollama": {"provider": "ollama", "runtime": "host", "url": os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434")},
            "llama_cpp": {"provider": "llama.cpp", "container": "api-or-runtime-profile"},
        },
        "ocr": {
            "tesseract": {"provider": "tesseract", "runtime": os.environ.get("APRAG_MEDIA_RUNTIME", "container")},
            "easyocr": {"provider": "easyocr", "runtime": "unsupported-by-default"},
        },
        "transcription": {
            "whisper_cpp": {"provider": "whisper.cpp", "runtime": os.environ.get("APRAG_MEDIA_RUNTIME", "container")},
            "faster_whisper": {"provider": "faster-whisper", "runtime": os.environ.get("APRAG_MEDIA_RUNTIME", "container")},
        },
    }
    status["paid_cloud_extensions"] = extension_settings()["paid_cloud_extensions"]
    return status


def parity_fixture_records() -> list[dict[str, Any]]:
    return [
        {"id": "a", "text": "Authentication uses API Gateway and Token Store.", "metadata": {"source": "auth"}},
        {"id": "b", "text": "Latency risk belongs to deployment monitoring.", "metadata": {"source": "ops"}},
    ]


def equivalent_vector_results(left: VectorStoreAdapter, right: VectorStoreAdapter, query: str) -> bool:
    records = parity_fixture_records()
    left.upsert(records)
    right.upsert(records)
    return [item["id"] for item in left.query(query, top_k=2)] == [item["id"] for item in right.query(query, top_k=2)]


def equivalent_graph_results(left: GraphStoreAdapter, right: GraphStoreAdapter) -> bool:
    for store in [left, right]:
        store.add_relationship("API Gateway", "Token Store", "auth.txt#chunk-1")
    return left.expand("API Gateway") == right.expand("API Gateway")


def contains_query_terms(result: dict[str, Any], query: str) -> bool:
    return bool(set(tokenize(query)).intersection(tokenize(str(result))))
