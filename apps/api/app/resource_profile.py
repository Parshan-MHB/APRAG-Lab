from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

GIB = 1024**3


@dataclass(frozen=True)
class ModelProfile:
    name: str
    label: str
    llm_model: str
    vlm_model: str
    embedding_model: str
    models: tuple[str, ...]
    min_free_disk_bytes: int
    min_memory_bytes: int
    steady_state_disk_bytes: int
    description: str


MODEL_PROFILES: dict[str, ModelProfile] = {
    "lite": ModelProfile(
        name="lite",
        label="Lite",
        llm_model="qwen3:8b",
        vlm_model="qwen3-vl:4b",
        embedding_model="bge-m3",
        models=("qwen3:8b", "qwen3-vl:4b", "bge-m3"),
        min_free_disk_bytes=40 * GIB,
        min_memory_bytes=15 * GIB,
        steady_state_disk_bytes=10 * GIB,
        description="Smallest real local stack for first launch; accepts Docker Desktop's 16 GB setting after container overhead.",
    ),
    "standard": ModelProfile(
        name="standard",
        label="Standard",
        llm_model="qwen3:14b",
        vlm_model="qwen3-vl:8b",
        embedding_model="bge-m3",
        models=("qwen3:14b", "qwen3-vl:8b", "bge-m3"),
        min_free_disk_bytes=80 * GIB,
        min_memory_bytes=32 * GIB,
        steady_state_disk_bytes=25 * GIB,
        description="Recommended quality profile for repeated benchmark use.",
    ),
    "high_quality": ModelProfile(
        name="high_quality",
        label="High Quality",
        llm_model="qwen3:32b",
        vlm_model="qwen3-vl:8b",
        embedding_model="bge-m3",
        models=("qwen3:32b", "qwen3-vl:8b", "bge-m3"),
        min_free_disk_bytes=160 * GIB,
        min_memory_bytes=64 * GIB,
        steady_state_disk_bytes=50 * GIB,
        description="Largest local profile for strong workstations; not pulled by default on smaller machines.",
    ),
}

PROFILE_ORDER = ("lite", "standard", "high_quality")


def _read_int(path: Path) -> int | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not value or value == "max":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def host_memory_bytes() -> int | None:
    cgroup_limit = _read_int(Path("/sys/fs/cgroup/memory.max"))
    if cgroup_limit and cgroup_limit < 1_000_000_000_000_000:
        return cgroup_limit
    legacy_limit = _read_int(Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"))
    if legacy_limit and legacy_limit < 1_000_000_000_000_000:
        return legacy_limit
    if hasattr(os, "sysconf"):
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return int(pages * page_size)
        except (OSError, ValueError):
            return None
    return None


def free_disk_bytes(path: str | Path | None = None) -> int:
    target = Path(path or os.environ.get("RAGBENCH_RESOURCE_CHECK_PATH", "/"))
    if not target.exists():
        target = Path("/")
    return shutil.disk_usage(target).free


def available_resources() -> dict[str, Any]:
    disk = free_disk_bytes()
    memory = host_memory_bytes()
    return {
        "runtime": "host_ollama",
        "resource_check_scope": "Docker resources are checked for app storage only; Ollama model memory/disk are managed by the host Ollama runtime.",
        "free_disk_bytes": disk,
        "memory_bytes": memory,
        "free_disk_gib": round(disk / GIB, 2),
        "memory_gib": None if memory is None else round(memory / GIB, 2),
        "disk_check_path": os.environ.get("RAGBENCH_RESOURCE_CHECK_PATH", "/"),
    }


def profile_requirements(profile: ModelProfile) -> dict[str, Any]:
    return {
        "name": profile.name,
        "label": profile.label,
        "models": list(profile.models),
        "llm_model": profile.llm_model,
        "vlm_model": profile.vlm_model,
        "embedding_model": profile.embedding_model,
        "min_free_disk_bytes": profile.min_free_disk_bytes,
        "min_free_disk_gib": round(profile.min_free_disk_bytes / GIB, 2),
        "min_memory_bytes": profile.min_memory_bytes,
        "min_memory_gib": round(profile.min_memory_bytes / GIB, 2),
        "steady_state_disk_bytes": profile.steady_state_disk_bytes,
        "steady_state_disk_gib": round(profile.steady_state_disk_bytes / GIB, 2),
        "description": profile.description,
    }


def fits(profile: ModelProfile, resources: dict[str, Any]) -> bool:
    return True


def max_feasible_profile(resources: dict[str, Any] | None = None) -> str | None:
    requested = requested_profile_name()
    return requested if requested in MODEL_PROFILES else "lite"


def requested_profile_name() -> str:
    return os.environ.get("RAGBENCH_MODEL_PROFILE", "auto").strip().lower() or "auto"


def selected_profile_name(resources: dict[str, Any] | None = None) -> str:
    requested = requested_profile_name()
    if requested in MODEL_PROFILES:
        return requested
    return "lite"


def selected_profile(resources: dict[str, Any] | None = None) -> ModelProfile:
    return MODEL_PROFILES[selected_profile_name(resources)]


def configured_models_for_bootstrap(resources: dict[str, Any] | None = None) -> list[str]:
    configured = os.environ.get("RAGBENCH_OLLAMA_MODELS")
    if configured:
        return [model.strip() for model in configured.split(",") if model.strip()]
    return list(selected_profile(resources).models)


def preflight_report(resources: dict[str, Any] | None = None) -> dict[str, Any]:
    resources = resources or available_resources()
    requested = requested_profile_name()
    selected = selected_profile_name(resources)
    feasible = max_feasible_profile(resources)
    profile = MODEL_PROFILES[selected]
    warnings: list[str] = []
    if requested == "auto":
        warnings.append("Auto uses the lite host-Ollama profile; choose standard or high_quality only after confirming the laptop can run those models.")
    if resources.get("memory_bytes") is None:
        warnings.append("Container memory could not be detected; host Ollama model capacity is managed outside Docker.")
    return {
        "requested_profile": requested,
        "selected_profile": selected,
        "max_feasible_profile": feasible,
        "safe_to_pull": True,
        "runtime": "host_ollama",
        "resources": resources,
        "selected_requirements": profile_requirements(profile),
        "profiles": {name: profile_requirements(MODEL_PROFILES[name]) for name in PROFILE_ORDER},
        "models_to_pull": configured_models_for_bootstrap(resources),
        "pull_commands": [f"ollama pull {model}" for model in configured_models_for_bootstrap(resources)],
        "warnings": warnings,
        "dynamic_flow": {
            "first_launch": "Install Ollama on the host laptop, pull the selected profile models, then start Docker Compose for the app.",
            "upgrade_path": "Raise RAGBENCH_MODEL_PROFILE to standard or high_quality after confirming the host laptop can run those models.",
            "blocked_behavior": "Docker memory does not block host Ollama; missing models are reported with ollama pull remediation.",
        },
    }
