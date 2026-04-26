from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

EXTENSION_TYPES = ("llm", "vlm", "ocr", "transcription", "vector_database", "object_storage")


@dataclass
class PaidCloudProviderStub:
    provider: str
    provider_type: str
    requires_credentials: bool = True
    enabled: bool = False
    network_call: Callable[[dict[str, Any]], dict[str, Any]] | None = None

    def availability(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "provider_type": self.provider_type,
            "enabled": self.enabled,
            "available": False,
            "requires_credentials": self.requires_credentials,
            "selected_by_default": False,
            "sends_uploaded_data": False,
        }

    def execute(self, payload: dict[str, Any], explicit_user_enabled: bool = False, credentials: str | None = None) -> dict[str, Any]:
        if not explicit_user_enabled or not self.enabled:
            return {
                "provider": self.provider,
                "status": "disabled",
                "network_called": False,
                "reason": "Paid/cloud provider is not explicitly enabled.",
            }
        if self.requires_credentials and not credentials:
            return {
                "provider": self.provider,
                "status": "missing_credentials",
                "network_called": False,
                "reason": "Credentials are required only after explicit provider enablement.",
            }
        if not self.network_call:
            return {"provider": self.provider, "status": "stubbed", "network_called": False}
        result = self.network_call(payload)
        return {"provider": self.provider, "status": "called", "network_called": True, "result": result}


def extension_registry() -> dict[str, dict[str, Any]]:
    providers = {
        "openai_llm": PaidCloudProviderStub("openai", "llm"),
        "anthropic_llm": PaidCloudProviderStub("anthropic", "llm"),
        "cloud_vlm": PaidCloudProviderStub("cloud-vlm", "vlm"),
        "cloud_ocr": PaidCloudProviderStub("cloud-ocr", "ocr"),
        "cloud_transcription": PaidCloudProviderStub("cloud-transcription", "transcription"),
        "managed_vector_db": PaidCloudProviderStub("managed-vector-db", "vector_database"),
        "cloud_object_storage": PaidCloudProviderStub("cloud-object-storage", "object_storage"),
    }
    return {name: provider.availability() for name, provider in providers.items()}


def extension_settings() -> dict[str, Any]:
    return {
        "local_required_providers": {
            "llm": "ollama_or_deterministic",
            "vlm": "ollama_vlm_or_derived_visual_evidence",
            "ocr": "host_tesseract_via_host_media_runtime",
            "transcription": "host_faster_whisper_or_whisper_cpp_via_host_media_runtime",
            "vector_database": "chroma_or_qdrant_container",
            "object_storage": "local_project_artifacts",
        },
        "paid_cloud_extensions": extension_registry(),
        "defaults": {
            "paid_cloud_selected": False,
            "credentials_required_for_local_workflows": False,
            "send_uploaded_data_without_explicit_configuration": False,
        },
        "extension_types": list(EXTENSION_TYPES),
    }


def assert_local_first_provider_policy(settings: dict[str, Any] | None = None) -> list[str]:
    settings = settings or extension_settings()
    violations: list[str] = []
    defaults = settings["defaults"]
    if defaults.get("paid_cloud_selected"):
        violations.append("paid_cloud_selected")
    if defaults.get("credentials_required_for_local_workflows"):
        violations.append("credentials_required_for_local_workflows")
    if defaults.get("send_uploaded_data_without_explicit_configuration"):
        violations.append("send_uploaded_data_without_explicit_configuration")
    for name, provider in settings["paid_cloud_extensions"].items():
        if provider.get("enabled") or provider.get("selected_by_default") or provider.get("sends_uploaded_data"):
            violations.append(name)
    return violations
