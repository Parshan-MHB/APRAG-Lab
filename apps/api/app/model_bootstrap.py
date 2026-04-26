from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from .resource_profile import configured_models_for_bootstrap, preflight_report


def _request(base_url: str, path: str, payload: dict | None = None, timeout: float = 30.0) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8") or "{}")


def wait_for_ollama(base_url: str, attempts: int = 60) -> None:
    for _ in range(attempts):
        try:
            _request(base_url, "/api/tags", timeout=3.0)
            return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2)
    raise RuntimeError(f"Ollama did not become reachable at {base_url}")


def required_models() -> list[str]:
    return configured_models_for_bootstrap()


def installed_models(base_url: str, timeout: float = 10.0) -> set[str]:
    body = _request(base_url, "/api/tags", timeout=timeout)
    return {model.get("name", "") for model in body.get("models", [])}


def pull_model(base_url: str, model: str) -> None:
    print(f"Pulling Ollama model: {model}", flush=True)
    _request(base_url, "/api/pull", {"name": model, "stream": False}, timeout=3600.0)


def main() -> None:
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    report = preflight_report()
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    wait_for_ollama(base_url)
    installed = installed_models(base_url)
    for model in report["models_to_pull"]:
        if model not in installed:
            pull_model(base_url, model)
    print("Ollama model bootstrap complete.", flush=True)


if __name__ == "__main__":
    main()
