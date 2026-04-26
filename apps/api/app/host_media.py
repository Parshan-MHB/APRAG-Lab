from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .database import data_dir
from .providers import ProviderUnavailable


def host_media_enabled() -> bool:
    return os.environ.get("RAGBENCH_MEDIA_RUNTIME", "container").strip().lower() == "host"


def host_media_base_url() -> str:
    return os.environ.get("RAGBENCH_HOST_MEDIA_BASE_URL", "http://host.docker.internal:8765").rstrip("/")


def relative_data_path(path: Path) -> str:
    try:
        return str(path.relative_to(data_dir()))
    except ValueError as exc:
        raise ProviderUnavailable(f"Path is outside DATA_DIR and cannot be sent to host media runtime: {path}") from exc


class HostMediaClient:
    provider = "host-media-runtime"

    def __init__(self, base_url: str | None = None, timeout: float | None = None):
        self.base_url = (base_url or host_media_base_url()).rstrip("/")
        self.timeout = timeout or float(os.environ.get("RAGBENCH_HOST_MEDIA_TIMEOUT", "900"))

    def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}{endpoint}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (TimeoutError, urllib.error.URLError) as exc:
            raise ProviderUnavailable(f"Host media runtime is unavailable at {self.base_url}: {exc}") from exc
        if not body.get("ok", True):
            raise ProviderUnavailable(str(body.get("error") or f"Host media runtime failed for {endpoint}"))
        return body

    def health(self) -> dict[str, Any]:
        try:
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=float(os.environ.get("RAGBENCH_HOST_MEDIA_HEALTH_TIMEOUT", "10"))) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            return {
                "ok": False,
                "runtime": "host",
                "base_url": self.base_url,
                "error": str(exc),
            }

    def probe_duration(self, path: Path) -> float | None:
        body = self._post("/probe", {"relative_path": relative_data_path(path)})
        duration = body.get("duration_seconds")
        return float(duration) if duration is not None else None

    def ocr(self, path: Path) -> dict[str, Any]:
        return self._post("/ocr", {"relative_path": relative_data_path(path)})

    def transcribe_audio(self, path: Path) -> dict[str, Any]:
        return self._post("/audio/transcribe", {"relative_path": relative_data_path(path)})

    def extract_audio(self, video_path: Path, output_path: Path) -> Path:
        body = self._post(
            "/video/extract-audio",
            {
                "relative_path": relative_data_path(video_path),
                "output_relative_path": relative_data_path(output_path),
            },
        )
        return data_dir() / body["output_relative_path"]

    def extract_frames(self, video_path: Path, frame_dir: Path, fps: float) -> list[Path]:
        body = self._post(
            "/video/extract-frames",
            {
                "relative_path": relative_data_path(video_path),
                "frame_dir_relative_path": relative_data_path(frame_dir),
                "fps": fps,
            },
        )
        return [data_dir() / relative for relative in body.get("frames", [])]
