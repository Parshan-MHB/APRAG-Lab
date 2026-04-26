from __future__ import annotations

MVP_EXCLUSIONS = [
    "paid_api_required",
    "user_accounts",
    "team_collaboration",
    "cloud_storage",
    "cloud_vector_database",
    "heavy_evaluation_framework",
    "advanced_permission_system",
    "fine_tuning",
    "long_term_production_observability",
    "enterprise_rbac",
    "fully_automated_model_selection",
]

REQUIRED_DEFERRED_SCOPE = [
    "structured_rag",
    "long_context_rag",
    "desktop_app_wrapper",
    "browser_only_local_mode",
    "qdrant_adapter",
    "networkx_adapter",
    "paid_cloud_extension_interfaces",
]


def guardrail_status() -> dict[str, object]:
    return {
        "mvp_exclusions": {name: {"enabled": False, "reason": "Explicitly excluded from MVP product path."} for name in MVP_EXCLUSIONS},
        "required_deferred_scope": {name: {"tracked": True, "reason": "Required later scope from original plan."} for name in REQUIRED_DEFERRED_SCOPE},
        "credentials_required": False,
        "paid_or_cloud_default": False,
        "local_first_required": True,
    }


def assert_no_forbidden_defaults(config: dict[str, object]) -> list[str]:
    violations = []
    for key in MVP_EXCLUSIONS:
        if config.get(key):
            violations.append(key)
    if config.get("paid_or_cloud_default"):
        violations.append("paid_or_cloud_default")
    if config.get("credentials_required"):
        violations.append("credentials_required")
    return violations
