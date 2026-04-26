from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any

from .database import data_dir
from .host_media import HostMediaClient, host_media_enabled
from .providers import ProviderUnavailable, TesseractOCRAdapter, WhisperCppAdapter, provider_registry
from .runtime_adapters import normalize_transcript_output


def deterministic_media_enabled() -> bool:
    return os.environ.get("APRAG_PROVIDER_MODE", "real").strip().lower() == "deterministic"


def media_duration_seconds(path: Path) -> float | None:
    if host_media_enabled():
        return HostMediaClient().probe_duration(path)
    if not shutil.which("ffprobe"):
        raise ProviderUnavailable("FFprobe is unavailable in the container.")
    process = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    value = process.stdout.strip()
    return float(value) if value else None


def _derived_dir(project_root: Path, name: str) -> Path:
    path = project_root / "sources" / "derived" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _relative(path: Path) -> str:
    return str(path.relative_to(data_dir()))


def _vlm_caption_block(
    image_path: Path,
    filename: str,
    project_root: Path,
    source_id: str,
    question: str,
    frame_or_image_id: str | None = None,
    timestamp_start: float | None = None,
    timestamp_end: float | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        observation = provider_registry().vlm().inspect_image(source_id, frame_or_image_id, image_path, question)
    except Exception as exc:
        return None, f"vlm_caption_failed:{image_path.name}:{exc}"
    caption_text = observation.visual_answer.strip()
    if not caption_text:
        try:
            retry_observation = provider_registry().vlm().inspect_image(
                source_id,
                frame_or_image_id,
                image_path,
                "Describe the visible scene in one concise factual sentence. Do not rely on OCR and do not return an empty response.",
            )
            caption_text = retry_observation.visual_answer.strip()
            if caption_text:
                observation = retry_observation
        except Exception as exc:
            return None, f"vlm_caption_empty:{image_path.name}:retry_failed:{exc}"
    if not caption_text:
        return None, f"vlm_caption_empty:{image_path.name}"
    caption_dir = _derived_dir(project_root, "captions")
    suffix = frame_or_image_id or source_id
    caption_path = caption_dir / f"{suffix}.txt"
    caption_path.write_text(caption_text, encoding="utf-8")
    metadata = {
        "derived_path": _relative(caption_path),
        "filename": filename,
        "provider": observation.provider,
        "model": observation.model,
        "question_for_vlm": question,
        "source_reference": frame_or_image_id or source_id,
        "visual_caption": True,
    }
    block: dict[str, Any] = {
        "block_type": "caption",
        "text": caption_text,
        "confidence": observation.confidence,
        "metadata": metadata,
    }
    if timestamp_start is not None:
        block["timestamp_start"] = timestamp_start
    if timestamp_end is not None:
        block["timestamp_end"] = timestamp_end
    if frame_or_image_id:
        block["frame_path"] = _relative(image_path)
        metadata["frame_id"] = frame_or_image_id
    return block, None


def image_blocks(path: Path, filename: str, project_root: Path, source_id: str) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    if deterministic_media_enabled():
        ocr_path = _derived_dir(project_root, "ocr") / f"{source_id}.txt"
        caption_path = _derived_dir(project_root, "captions") / f"{source_id}.txt"
        ocr_text = f"Deterministic OCR placeholder for image '{filename}'."
        caption = f"Deterministic caption placeholder for image '{filename}'."
        ocr_path.write_text(ocr_text, encoding="utf-8")
        caption_path.write_text(caption, encoding="utf-8")
        return [
            {"block_type": "ocr", "text": ocr_text, "confidence": 0.5, "metadata": {"derived_path": _relative(ocr_path), "filename": filename}},
            {"block_type": "caption", "text": caption, "confidence": 0.5, "metadata": {"derived_path": _relative(caption_path), "filename": filename}},
        ], warnings

    if host_media_enabled():
        host_result = HostMediaClient().ocr(path)
        result_text = host_result.get("text", "")
        result_provider = host_result.get("provider", "host-tesseract")
        result_model = host_result.get("model", "tesseract-cli")
        result_confidence = float(host_result.get("confidence", 0.8))
    else:
        ocr = provider_registry().ocr()
        result = ocr.extract_text(path)
        result_text = result.text
        result_provider = result.provider
        result_model = result.model
        result_confidence = float(result.metadata.get("confidence", 0.8)) if result.metadata else 0.8
    ocr_path = _derived_dir(project_root, "ocr") / f"{source_id}.txt"
    ocr_path.write_text(result_text, encoding="utf-8")
    blocks = [
        {
            "block_type": "ocr",
            "text": result_text,
            "confidence": result_confidence,
            "metadata": {"derived_path": _relative(ocr_path), "filename": filename, "provider": result_provider, "model": result_model},
        }
    ]
    caption_block, warning = _vlm_caption_block(
        path,
        filename,
        project_root,
        source_id,
        "Create a concise factual visual caption. Describe the visible scene, objects, people, entities, diagrams, UI states, visible text if any, and limitations.",
        frame_or_image_id=source_id,
    )
    if caption_block:
        blocks.append(caption_block)
    if warning:
        warnings.append(warning)
    return blocks, warnings


def transcribe_audio(path: Path, project_root: Path, source_id: str) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    if deterministic_media_enabled():
        transcript_path = _derived_dir(project_root, "transcripts") / f"{source_id}.txt"
        transcript = f"Deterministic transcript placeholder for audio '{path.name}'."
        transcript_path.write_text(transcript, encoding="utf-8")
        return [
            {
                "block_type": "transcript",
                "text": transcript,
                "timestamp_start": 0.0,
                "timestamp_end": 0.0,
                "confidence": 0.5,
                "metadata": {"derived_path": _relative(transcript_path), "filename": path.name},
            }
        ], warnings

    if host_media_enabled():
        transcript = HostMediaClient().transcribe_audio(path)
    else:
        try:
            transcript = provider_registry().transcription().transcribe(path)
        except ProviderUnavailable as exc:
            warnings.append(str(exc))
            transcript = WhisperCppAdapter().transcribe(path)
    normalized = normalize_transcript_output(transcript["provider"], transcript.get("segments", []))
    transcript_path = _derived_dir(project_root, "transcripts") / f"{source_id}.txt"
    transcript_path.write_text(normalized["text"], encoding="utf-8")
    return [
        {
            "block_type": "transcript",
            "text": segment["text"],
            "timestamp_start": segment["timestamp_start"],
            "timestamp_end": segment["timestamp_end"],
            "confidence": segment["confidence"],
            "metadata": {"derived_path": _relative(transcript_path), "filename": path.name, "provider": transcript["provider"]},
        }
        for segment in normalized["segments"]
        if segment["text"].strip()
    ], warnings


def extract_audio(video_path: Path, project_root: Path, source_id: str) -> Path:
    target = _derived_dir(project_root, "audio") / f"{source_id}.wav"
    if host_media_enabled():
        return HostMediaClient().extract_audio(video_path, target)
    if not shutil.which("ffmpeg"):
        raise ProviderUnavailable("FFmpeg is unavailable in the container.")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(target)],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    return target


def extract_frames(video_path: Path, project_root: Path, source_id: str, fps: float = 0.2) -> list[Path]:
    frame_dir = _derived_dir(project_root, "frames") / source_id
    frame_dir.mkdir(parents=True, exist_ok=True)
    if host_media_enabled():
        return HostMediaClient().extract_frames(video_path, frame_dir, fps)
    if not shutil.which("ffmpeg"):
        raise ProviderUnavailable("FFmpeg is unavailable in the container.")
    pattern = frame_dir / "frame_%04d.jpg"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path), "-vf", f"fps={fps}", str(pattern)],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    return sorted(frame_dir.glob("frame_*.jpg"))


def video_blocks(path: Path, filename: str, project_root: Path, source_id: str) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    if deterministic_media_enabled():
        transcript_path = _derived_dir(project_root, "transcripts") / f"{source_id}.txt"
        frame_path = _derived_dir(project_root, "frames") / f"{source_id}_frame_0001.txt"
        transcript = f"Deterministic transcript placeholder for video '{filename}'."
        frame_caption = f"Deterministic frame caption placeholder for video '{filename}'."
        transcript_path.write_text(transcript, encoding="utf-8")
        frame_path.write_text(frame_caption, encoding="utf-8")
        return [
            {"block_type": "transcript", "text": transcript, "timestamp_start": 0.0, "timestamp_end": 0.0, "confidence": 0.5, "metadata": {"derived_path": _relative(transcript_path), "filename": filename}},
            {"block_type": "caption", "text": frame_caption, "timestamp_start": 0.0, "timestamp_end": 0.0, "frame_path": _relative(frame_path), "confidence": 0.5, "metadata": {"filename": filename}},
        ], warnings

    blocks: list[dict[str, Any]] = []
    try:
        audio = extract_audio(path, project_root, source_id)
        audio_blocks, audio_warnings = transcribe_audio(audio, project_root, source_id)
        warnings.extend(audio_warnings)
        blocks.extend(audio_blocks)
    except Exception as exc:
        warnings.append(f"video_audio_failed: {exc}")
    try:
        frames = extract_frames(path, project_root, source_id)
        host_media = HostMediaClient() if host_media_enabled() else None
        ocr = None if host_media else TesseractOCRAdapter()
        for index, frame in enumerate(frames[:30], start=1):
            timestamp_start = float(index - 1)
            timestamp_end = float(index)
            try:
                if host_media:
                    host_result = host_media.ocr(frame)
                    result_text = host_result.get("text", "")
                    result_provider = host_result.get("provider", "host-tesseract")
                    result_model = host_result.get("model", "tesseract-cli")
                else:
                    result = ocr.extract_text(frame)
                    result_text = result.text
                    result_provider = result.provider
                    result_model = result.model
                if result_text.strip():
                    blocks.append(
                        {
                            "block_type": "frame_ocr",
                            "text": result_text,
                            "timestamp_start": timestamp_start,
                            "timestamp_end": timestamp_end,
                            "frame_path": _relative(frame),
                            "confidence": 0.8,
                            "metadata": {"filename": filename, "provider": result_provider, "model": result_model},
                        }
                    )
            except Exception as exc:
                warnings.append(f"frame_ocr_failed:{frame.name}:{exc}")
            caption_block, warning = _vlm_caption_block(
                frame,
                filename,
                project_root,
                source_id,
                "Create a concise factual caption for this video frame. Include visible text, UI state, objects, and limitations.",
                frame_or_image_id=f"{source_id}_frame_{index:04d}",
                timestamp_start=timestamp_start,
                timestamp_end=timestamp_end,
            )
            if caption_block:
                blocks.append(caption_block)
            if warning:
                warnings.append(warning)
    except Exception as exc:
        warnings.append(f"video_frame_failed: {exc}")
    return blocks, warnings
