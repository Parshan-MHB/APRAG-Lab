#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


def data_dir() -> Path:
    return Path(os.environ.get("APRAG_HOST_DATA_DIR", Path.cwd() / "data")).resolve()


def safe_path(relative_path: str) -> Path:
    root = data_dir()
    target = (root / relative_path).resolve()
    if root != target and root not in target.parents:
        raise RuntimeError(f"Path escapes APRAG_HOST_DATA_DIR: {relative_path}")
    return target


def command_available(name: str) -> bool:
    return shutil.which(name) is not None


def run(command: list[str], timeout: int = 900) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, capture_output=True, text=True, timeout=timeout)


def transcribe_with_faster_whisper(path: Path) -> dict[str, Any]:
    from faster_whisper import WhisperModel

    model_name = os.environ.get("APRAG_FASTER_WHISPER_MODEL", "large-v3-turbo")
    device = os.environ.get("APRAG_WHISPER_DEVICE", "cpu")
    compute_type = os.environ.get("APRAG_WHISPER_COMPUTE_TYPE", "int8")
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    segments, info = model.transcribe(str(path), vad_filter=True)
    return {
        "ok": True,
        "provider": "faster-whisper-host",
        "language": getattr(info, "language", None),
        "segments": [
            {
                "text": segment.text.strip(),
                "timestamp_start": float(segment.start),
                "timestamp_end": float(segment.end),
                "confidence": 1.0 - float(getattr(segment, "avg_logprob", 0.0) < -1.0),
            }
            for segment in segments
            if segment.text.strip()
        ],
    }


def transcribe_with_whisper_cpp(path: Path) -> dict[str, Any]:
    executable = shutil.which("whisper-cli") or shutil.which("main")
    if not executable:
        raise RuntimeError("Neither faster-whisper nor whisper.cpp is available on the host.")
    model_path = os.environ.get("WHISPER_CPP_MODEL", "")
    command = [executable, "-f", str(path), "-otxt"]
    if model_path:
        command.extend(["-m", model_path])
    process = run(command, timeout=900)
    text = process.stdout.strip()
    return {
        "ok": True,
        "provider": "whisper.cpp-host",
        "segments": [{"text": text, "timestamp_start": 0.0, "timestamp_end": 0.0, "confidence": 0.0}],
    }


def transcribe(path: Path) -> dict[str, Any]:
    try:
        return transcribe_with_faster_whisper(path)
    except Exception as faster_error:
        try:
            response = transcribe_with_whisper_cpp(path)
            response["fallback_warning"] = str(faster_error)
            return response
        except Exception as whisper_error:
            raise RuntimeError(f"Host transcription unavailable: faster-whisper={faster_error}; whisper.cpp={whisper_error}") from whisper_error


def health() -> dict[str, Any]:
    faster_whisper = True
    faster_warning = ""
    try:
        import faster_whisper as _faster_whisper  # noqa: F401
    except Exception as exc:
        faster_whisper = False
        faster_warning = str(exc)
    return {
        "ok": True,
        "runtime": "host",
        "data_dir": str(data_dir()),
        "ffmpeg": {"available": command_available("ffmpeg"), "path": shutil.which("ffmpeg") or ""},
        "ffprobe": {"available": command_available("ffprobe"), "path": shutil.which("ffprobe") or ""},
        "tesseract": {"available": command_available("tesseract"), "path": shutil.which("tesseract") or ""},
        "faster_whisper": {"available": faster_whisper, "warning": faster_warning},
        "whisper_cpp": {"available": command_available("whisper-cli") or command_available("main")},
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "APRAGLabHostMedia/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"host-media {self.address_string()} {fmt % args}", flush=True)

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _payload(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if not length:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json(200, health())
            return
        self._json(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:
        try:
            payload = self._payload()
            if self.path == "/probe":
                process = run(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-show_entries",
                        "format=duration",
                        "-of",
                        "default=noprint_wrappers=1:nokey=1",
                        str(safe_path(payload["relative_path"])),
                    ],
                    timeout=30,
                )
                value = process.stdout.strip()
                self._json(200, {"ok": True, "duration_seconds": float(value) if value else None})
                return
            if self.path == "/ocr":
                if not command_available("tesseract"):
                    raise RuntimeError("Host tesseract is not installed or not on PATH.")
                process = run(["tesseract", str(safe_path(payload["relative_path"])), "stdout"], timeout=120)
                self._json(200, {"ok": True, "provider": "host-tesseract", "model": "tesseract-cli", "text": process.stdout, "confidence": 0.8})
                return
            if self.path == "/audio/transcribe":
                self._json(200, transcribe(safe_path(payload["relative_path"])))
                return
            if self.path == "/video/extract-audio":
                if not command_available("ffmpeg"):
                    raise RuntimeError("Host ffmpeg is not installed or not on PATH.")
                source = safe_path(payload["relative_path"])
                target = safe_path(payload["output_relative_path"])
                target.parent.mkdir(parents=True, exist_ok=True)
                run(["ffmpeg", "-y", "-i", str(source), "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(target)], timeout=300)
                self._json(200, {"ok": True, "output_relative_path": payload["output_relative_path"]})
                return
            if self.path == "/video/extract-frames":
                if not command_available("ffmpeg"):
                    raise RuntimeError("Host ffmpeg is not installed or not on PATH.")
                source = safe_path(payload["relative_path"])
                frame_dir = safe_path(payload["frame_dir_relative_path"])
                frame_dir.mkdir(parents=True, exist_ok=True)
                for old_frame in frame_dir.glob("frame_*.jpg"):
                    old_frame.unlink()
                pattern = frame_dir / "frame_%04d.jpg"
                run(["ffmpeg", "-y", "-i", str(source), "-vf", f"fps={float(payload.get('fps', 0.2))}", str(pattern)], timeout=300)
                root = data_dir()
                frames = [str(path.relative_to(root)) for path in sorted(frame_dir.glob("frame_*.jpg"))]
                self._json(200, {"ok": True, "frames": frames})
                return
            self._json(404, {"ok": False, "error": "not found"})
        except Exception as exc:
            self._json(500, {"ok": False, "error": str(exc)})


def main() -> None:
    parser = argparse.ArgumentParser(description="Host media runtime for APRAG-Lab.")
    parser.add_argument("--host", default=os.environ.get("APRAG_HOST_MEDIA_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("APRAG_HOST_MEDIA_PORT", "8765")))
    args = parser.parse_args()
    print(f"APRAG-Lab host media runtime listening on http://{args.host}:{args.port}", flush=True)
    print(f"APRAG_HOST_DATA_DIR={data_dir()}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
