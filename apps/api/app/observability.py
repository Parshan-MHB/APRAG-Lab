from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator

from .diagnostics import scrub, write_local_log

_INITIALIZED = False
_OTEL_AVAILABLE = False
_OTEL_ERROR = ""


def _enabled(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _safe_attribute(value: Any) -> str | int | float | bool:
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _otlp_endpoints() -> list[str]:
    raw = os.environ.get("APRAG_OTEL_EXPORTER_OTLP_ENDPOINTS", "")
    if raw:
        return [endpoint.strip() for endpoint in raw.split(",") if endpoint.strip()]
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    return [endpoint] if endpoint else []


def setup_observability(service_name: str) -> None:
    global _INITIALIZED, _OTEL_AVAILABLE, _OTEL_ERROR
    if _INITIALIZED:
        return
    _INITIALIZED = True
    if not _enabled(os.environ.get("APRAG_OBSERVABILITY_ENABLED"), default=True):
        write_local_log("observability_disabled", {"service": service_name})
        return
    endpoints = _otlp_endpoints()
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except Exception as exc:  # pragma: no cover - exercised in slim dependency environments
        _OTEL_ERROR = str(exc)
        write_local_log("observability_import_failed", {"service": service_name, "error": _OTEL_ERROR})
        return

    try:
        provider = TracerProvider(resource=Resource.create({"service.name": service_name, "service.namespace": "aprag-lab"}))
        for endpoint in endpoints:
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
        trace.set_tracer_provider(provider)
        _OTEL_AVAILABLE = True
        write_local_log("observability_started", {"service": service_name, "otlp_endpoints": endpoints})
        globals()["_FastAPIInstrumentor"] = FastAPIInstrumentor
    except Exception as exc:  # pragma: no cover - defensive against duplicate providers under reload
        _OTEL_ERROR = str(exc)
        write_local_log("observability_setup_failed", {"service": service_name, "error": _OTEL_ERROR})


def instrument_fastapi(app: Any) -> None:
    instrumentor = globals().get("_FastAPIInstrumentor")
    if instrumentor and _OTEL_AVAILABLE:
        try:
            instrumentor.instrument_app(app)
        except Exception as exc:  # pragma: no cover
            write_local_log("observability_fastapi_instrument_failed", {"error": str(exc)})


def observability_status() -> dict[str, Any]:
    endpoints = _otlp_endpoints()
    langsmith_enabled = _enabled(os.environ.get("LANGSMITH_TRACING")) or _enabled(os.environ.get("LANGCHAIN_TRACING_V2"))
    phoenix_endpoint = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "http://phoenix:6006/v1/traces")
    return {
        "enabled": _enabled(os.environ.get("APRAG_OBSERVABILITY_ENABLED"), default=True),
        "otel_available": _OTEL_AVAILABLE,
        "otel_error": _OTEL_ERROR,
        "otlp_endpoints": endpoints,
        "dashboards": [
            {
                "id": "jaeger",
                "label": "Jaeger Traces",
                "kind": "OpenTelemetry trace dashboard",
                "url": "http://localhost:16686",
                "queryable": True,
                "container": "jaeger",
            },
            {
                "id": "phoenix",
                "label": "Phoenix AI Observability",
                "kind": "LLM/RAG trace dashboard",
                "url": "http://localhost:6006",
                "queryable": True,
                "container": "phoenix",
            },
            {
                "id": "langsmith",
                "label": "LangSmith Project",
                "kind": "external LangGraph/LangChain trace dashboard",
                "url": os.environ.get("LANGSMITH_PROJECT_URL", "https://smith.langchain.com"),
                "queryable": bool(os.environ.get("LANGSMITH_API_KEY")),
                "external": True,
            },
        ],
        "langsmith": {
            "enabled": langsmith_enabled,
            "api_key_configured": bool(os.environ.get("LANGSMITH_API_KEY")),
            "project": os.environ.get("LANGSMITH_PROJECT") or os.environ.get("LANGCHAIN_PROJECT") or "APRAG-Lab",
            "endpoint": os.environ.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"),
        },
        "phoenix": {
            "enabled": any("phoenix" in endpoint for endpoint in endpoints) or bool(os.environ.get("PHOENIX_COLLECTOR_ENDPOINT")),
            "collector_endpoint": phoenix_endpoint,
        },
    }


@contextmanager
def observe_span(name: str, attributes: dict[str, Any] | None = None, payload: dict[str, Any] | None = None) -> Iterator[Any]:
    attributes = scrub(attributes or {})
    payload = scrub(payload or {})
    span = None
    span_context = None
    langsmith_context = None
    if _OTEL_AVAILABLE:
        try:
            from opentelemetry import trace

            span_context = trace.get_tracer("aprag-lab").start_as_current_span(name)
            span = span_context.__enter__()
            for key, value in attributes.items():
                span.set_attribute(key, _safe_attribute(value))
            if payload:
                span.add_event("payload", {key: _safe_attribute(value) for key, value in payload.items()})
        except Exception as exc:  # pragma: no cover
            write_local_log("observability_span_failed", {"name": name, "error": str(exc)})
            span = None
            span_context = None
    if _enabled(os.environ.get("LANGSMITH_TRACING")) or _enabled(os.environ.get("LANGCHAIN_TRACING_V2")):
        try:
            from langsmith import trace as langsmith_trace

            langsmith_context = langsmith_trace(name, inputs=payload or attributes, metadata=attributes)
            langsmith_context.__enter__()
        except Exception as exc:  # pragma: no cover
            write_local_log("langsmith_trace_failed", {"name": name, "error": str(exc)})
            langsmith_context = None
    try:
        yield span
    except Exception as exc:
        if span:
            span.record_exception(exc)
            span.set_attribute("error", True)
        raise
    finally:
        if langsmith_context:
            try:
                langsmith_context.__exit__(None, None, None)
            except Exception as exc:  # pragma: no cover
                write_local_log("langsmith_trace_close_failed", {"name": name, "error": str(exc)})
        if span:
            if span_context:
                span_context.__exit__(None, None, None)
            else:
                span.end()
