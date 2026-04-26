from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import urllib.error
import urllib.request
import base64
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel

from .resource_profile import selected_profile

_DEFAULT_PROFILE = selected_profile()
DEFAULT_OLLAMA_LLM_MODEL = os.environ.get("RAGBENCH_DEFAULT_OLLAMA_LLM", _DEFAULT_PROFILE.llm_model)
DEFAULT_OLLAMA_VLM_MODEL = os.environ.get("RAGBENCH_DEFAULT_OLLAMA_VLM", _DEFAULT_PROFILE.vlm_model)
DEFAULT_EMBEDDING_MODEL = os.environ.get("RAGBENCH_DEFAULT_EMBEDDINGS", _DEFAULT_PROFILE.embedding_model)
DEFAULT_TRANSCRIPTION_MODEL = os.environ.get("RAGBENCH_DEFAULT_TRANSCRIPTION", "faster-whisper:large-v3-turbo")


def ollama_model_installed(requested: str, installed: list[str] | set[str]) -> bool:
    installed_names = {name for name in installed if name}
    if requested in installed_names:
        return True
    if ":" not in requested and f"{requested}:latest" in installed_names:
        return True
    return False


class ProviderUnavailable(RuntimeError):
    pass


class ProviderTimeout(RuntimeError):
    pass


class VisualObservation(BaseModel):
    source_id: str
    frame_or_image_id: str | None = None
    question: str
    visual_answer: str
    confidence: float
    provider: str
    model: str
    warnings: list[str] = []


class ProviderResult(BaseModel):
    provider: str
    model: str
    text: str
    warnings: list[str] = []
    metadata: dict[str, Any] = {}


def default_model_guidance() -> dict[str, Any]:
    return {
        "ollama": {
            "llm_model": DEFAULT_OLLAMA_LLM_MODEL,
            "vlm_model": DEFAULT_OLLAMA_VLM_MODEL,
            "setup": [
                "Docker Compose runs the app; real model mode requires Ollama running on the host laptop.",
                f"To enable local LLM calls, run: ollama pull {DEFAULT_OLLAMA_LLM_MODEL}",
                f"To enable local VLM calls, run: ollama pull {DEFAULT_OLLAMA_VLM_MODEL}",
            ],
        },
        "embeddings": {
            "default": DEFAULT_EMBEDDING_MODEL,
            "choices": ["ollama:bge-m3", "deterministic_lexical", "ollama:nomic-embed-text"],
        },
        "transcription": {
            "default": DEFAULT_TRANSCRIPTION_MODEL,
            "fallback": "whisper.cpp:large-v3-turbo",
        },
        "document_parser": {
            "default": "fallback_pypdf_python_docx",
            "docling": "preferred when installed and configured",
        },
    }


def http_json(url: str, payload: dict[str, Any] | None = None, timeout: float = 10.0) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except TimeoutError as exc:
        raise ProviderTimeout(f"Provider request timed out: {url}") from exc
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, TimeoutError):
            raise ProviderTimeout(f"Provider request timed out: {url}") from exc
        raise ProviderUnavailable(f"Provider is unavailable: {url}") from exc


def ollama_generation_payload(model: str, prompt: str, **extra: Any) -> dict[str, Any]:
    payload = {"model": model, "prompt": prompt, "stream": False, **extra}
    disable_thinking = os.environ.get("RAGBENCH_OLLAMA_THINKING", "0").strip().lower() not in {"1", "true", "yes", "on"}
    if disable_thinking:
        payload["think"] = False
    return payload


def ollama_generation_options(prefix: str = "RAGBENCH_OLLAMA") -> dict[str, Any]:
    return {
        "temperature": float(os.environ.get(f"{prefix}_TEMPERATURE", os.environ.get("RAGBENCH_OLLAMA_TEMPERATURE", "0"))),
        "num_predict": max(32, int(os.environ.get(f"{prefix}_NUM_PREDICT", os.environ.get("RAGBENCH_OLLAMA_NUM_PREDICT", "512")))),
    }


class OllamaAdapter:
    provider = "ollama"

    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_LLM_MODEL,
        base_url: str | None = None,
        timeout: float = 10.0,
        transport: Callable[[str, dict[str, Any] | None, float], dict[str, Any]] = http_json,
    ):
        self.model = model
        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL") or "http://host.docker.internal:11434").rstrip("/")
        self.timeout = timeout
        self.transport = transport

    def validate_model(self) -> dict[str, Any]:
        tags = self.transport(f"{self.base_url}/api/tags", None, self.timeout)
        installed = [model.get("name") for model in tags.get("models", [])]
        if not ollama_model_installed(self.model, installed):
            return {
                "available": False,
                "model": self.model,
                "warning": f"Ollama model '{self.model}' is not installed.",
                "setup": f"ollama pull {self.model}",
            }
        return {"available": True, "model": self.model}

    def generate(self, prompt: str, options: dict[str, Any] | None = None) -> ProviderResult:
        validation = self.validate_model()
        if not validation["available"]:
            raise ProviderUnavailable(validation["warning"])
        payload = ollama_generation_payload(self.model, prompt)
        payload["options"] = {**ollama_generation_options(), **(options or {})}
        body = self.transport(
            f"{self.base_url}/api/generate",
            payload,
            self.timeout,
        )
        text = body.get("response", "")
        warnings = []
        if not text and body.get("thinking"):
            warnings.append("Ollama returned only thinking content; enable final-answer generation or increase model budget.")
        return ProviderResult(provider=self.provider, model=self.model, text=text, warnings=warnings, metadata={"done_reason": body.get("done_reason")})

    def generate_structured(self, prompt: str, schema: dict[str, Any], options: dict[str, Any] | None = None) -> dict[str, Any]:
        result = self.generate(f"{prompt}\nReturn JSON matching this schema:\n{json.dumps(schema)}", options=options)
        try:
            return json.loads(result.text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", result.text, flags=re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise ProviderUnavailable("Ollama did not return valid JSON for structured generation.")


class LlamaCppAdapter:
    provider = "llama.cpp"

    def __init__(self, endpoint: str | None = None, model: str = "local-llamacpp", timeout: float = 10.0):
        self.endpoint = endpoint or os.environ.get("LLAMACPP_BASE_URL")
        self.model = model
        self.timeout = timeout

    def availability(self) -> dict[str, Any]:
        if not self.endpoint:
            return {"available": False, "provider": self.provider, "model": self.model, "warning": "llama.cpp endpoint is not configured."}
        return {"available": True, "provider": self.provider, "model": self.model, "endpoint": self.endpoint}

    def generate(self, prompt: str) -> ProviderResult:
        if not self.endpoint:
            raise ProviderUnavailable("llama.cpp endpoint is not configured.")
        body = http_json(f"{self.endpoint.rstrip('/')}/completion", {"prompt": prompt}, self.timeout)
        return ProviderResult(provider=self.provider, model=self.model, text=body.get("content", ""))


class OllamaVLMAdapter(OllamaAdapter):
    provider = "ollama_vlm"

    def __init__(self, model: str = DEFAULT_OLLAMA_VLM_MODEL, **kwargs: Any):
        super().__init__(model=model, **kwargs)

    def inspect_image(self, source_id: str, frame_or_image_id: str | None, image_path: Path, question: str) -> VisualObservation:
        validation = self.validate_model()
        if not validation["available"]:
            raise ProviderUnavailable(validation["warning"])
        if not image_path.exists():
            raise ProviderUnavailable(f"Image artifact does not exist: {image_path}")
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        body = self.transport(
            f"{self.base_url}/api/generate",
            {
                **ollama_generation_payload(self.model, f"Question: {question}", images=[encoded]),
                "options": ollama_generation_options("RAGBENCH_VLM"),
            },
            self.timeout,
        )
        result = ProviderResult(provider=self.provider, model=self.model, text=body.get("response", ""))
        return VisualObservation(
            source_id=source_id,
            frame_or_image_id=frame_or_image_id,
            question=question,
            visual_answer=result.text,
            confidence=0.6,
            provider=self.provider,
            model=self.model,
        )


class DeterministicEmbeddingAdapter:
    provider = "deterministic"
    model = "deterministic_lexical"

    def embed_text(self, text: str) -> list[float]:
        from .rag import deterministic_vector

        return deterministic_vector(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]


class OllamaEmbeddingAdapter(OllamaAdapter):
    provider = "ollama_embeddings"

    def __init__(self, model: str = DEFAULT_EMBEDDING_MODEL, **kwargs: Any):
        super().__init__(model=model, **kwargs)

    def embed_text(self, text: str) -> list[float]:
        validation = self.validate_model()
        if not validation["available"]:
            raise ProviderUnavailable(validation["warning"])
        body = self.transport(f"{self.base_url}/api/embeddings", {"model": self.model, "prompt": text}, self.timeout)
        return body.get("embedding", [])

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]


class TesseractOCRAdapter:
    provider = "tesseract"

    def availability(self) -> dict[str, Any]:
        path = shutil.which("tesseract")
        return {
            "available": bool(path),
            "provider": self.provider,
            "path": path or "",
            "warning": "" if path else "Tesseract executable is not available in the container.",
        }

    def extract_text(self, image_path: Path, timeout: float = 20.0) -> ProviderResult:
        status = self.availability()
        if not status["available"]:
            raise ProviderUnavailable(status["warning"])
        try:
            process = subprocess.run(
                ["tesseract", str(image_path), "stdout"],
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise ProviderTimeout("Tesseract OCR timed out.") from exc
        return ProviderResult(provider=self.provider, model="tesseract-cli", text=process.stdout)


class EasyOCRAdapter:
    provider = "easyocr"

    def availability(self) -> dict[str, Any]:
        try:
            import easyocr  # noqa: F401
        except Exception:
            return {"available": False, "provider": self.provider, "warning": "EasyOCR Python package is not installed."}
        return {"available": True, "provider": self.provider}

    def extract_text(self, image_path: Path) -> ProviderResult:
        if not self.availability()["available"]:
            raise ProviderUnavailable("EasyOCR Python package is not installed.")
        import easyocr

        reader = easyocr.Reader(["en"], gpu=os.environ.get("RAGBENCH_EASYOCR_GPU", "0") == "1")
        rows = reader.readtext(str(image_path), detail=1)
        text = "\n".join(str(row[1]) for row in rows)
        confidences = [float(row[2]) for row in rows if len(row) > 2]
        confidence = sum(confidences) / len(confidences) if confidences else 0.0
        return ProviderResult(provider=self.provider, model="easyocr-en", text=text, metadata={"confidence": confidence})


class WhisperCppAdapter:
    provider = "whisper.cpp"

    def availability(self) -> dict[str, Any]:
        path = shutil.which("whisper-cli") or shutil.which("main")
        return {
            "available": bool(path),
            "provider": self.provider,
            "path": path or "",
            "warning": "" if path else "whisper.cpp executable is not available in the container.",
        }

    def transcribe(self, audio_path: Path) -> dict[str, Any]:
        if not self.availability()["available"]:
            raise ProviderUnavailable("whisper.cpp executable is not available in the container.")
        executable = shutil.which("whisper-cli") or shutil.which("main")
        model_path = os.environ.get("WHISPER_CPP_MODEL", "")
        command = [executable, "-f", str(audio_path), "-otxt"]
        if model_path:
            command.extend(["-m", model_path])
        process = subprocess.run(command, check=True, capture_output=True, text=True, timeout=900)
        text = process.stdout.strip()
        return {"segments": [{"text": text, "timestamp_start": 0.0, "timestamp_end": 0.0, "confidence": 0.0}], "provider": self.provider, "source": str(audio_path)}


class FasterWhisperAdapter:
    provider = "faster-whisper"

    def availability(self) -> dict[str, Any]:
        try:
            import faster_whisper  # noqa: F401
        except Exception:
            return {"available": False, "provider": self.provider, "warning": "faster-whisper Python package is not installed."}
        return {"available": True, "provider": self.provider}

    def transcribe(self, audio_path: Path) -> dict[str, Any]:
        if not self.availability()["available"]:
            raise ProviderUnavailable("faster-whisper Python package is not installed.")
        from faster_whisper import WhisperModel

        model_name = os.environ.get("RAGBENCH_FASTER_WHISPER_MODEL", "large-v3-turbo")
        device = os.environ.get("RAGBENCH_WHISPER_DEVICE", "cpu")
        compute_type = os.environ.get("RAGBENCH_WHISPER_COMPUTE_TYPE", "int8")
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
        segments, info = model.transcribe(str(audio_path), vad_filter=True)
        normalized = []
        for segment in segments:
            normalized.append(
                {
                    "text": segment.text.strip(),
                    "timestamp_start": float(segment.start),
                    "timestamp_end": float(segment.end),
                    "confidence": 1.0 - float(getattr(segment, "avg_logprob", 0.0) < -1.0),
                }
            )
        return {"segments": normalized, "provider": self.provider, "source": str(audio_path), "language": getattr(info, "language", None)}


def provider_mode() -> str:
    return os.environ.get("RAGBENCH_PROVIDER_MODE", "real").strip().lower()


def deterministic_mode() -> bool:
    return provider_mode() == "deterministic"


class ProviderRegistry:
    def __init__(self, mode: str | None = None):
        self.mode = (mode or provider_mode()).lower()

    @property
    def deterministic(self) -> bool:
        return self.mode == "deterministic"

    def llm(self) -> OllamaAdapter:
        if self.deterministic:
            raise ProviderUnavailable("Deterministic mode does not expose a real LLM adapter.")
        return OllamaAdapter(model=os.environ.get("RAGBENCH_DEFAULT_OLLAMA_LLM", DEFAULT_OLLAMA_LLM_MODEL), timeout=float(os.environ.get("RAGBENCH_LLM_TIMEOUT", "180")))

    def vlm(self) -> OllamaVLMAdapter:
        if self.deterministic:
            raise ProviderUnavailable("Deterministic mode does not expose a real VLM adapter.")
        return OllamaVLMAdapter(model=os.environ.get("RAGBENCH_DEFAULT_OLLAMA_VLM", DEFAULT_OLLAMA_VLM_MODEL), timeout=float(os.environ.get("RAGBENCH_VLM_TIMEOUT", "240")))

    def embeddings(self) -> DeterministicEmbeddingAdapter | OllamaEmbeddingAdapter:
        if self.deterministic:
            return DeterministicEmbeddingAdapter()
        return OllamaEmbeddingAdapter(model=os.environ.get("RAGBENCH_DEFAULT_EMBEDDINGS", DEFAULT_EMBEDDING_MODEL), timeout=float(os.environ.get("RAGBENCH_EMBEDDING_TIMEOUT", "60")))

    def ocr(self) -> TesseractOCRAdapter | EasyOCRAdapter:
        if os.environ.get("RAGBENCH_OCR_PROVIDER", "tesseract") == "easyocr":
            return EasyOCRAdapter()
        return TesseractOCRAdapter()

    def transcription(self) -> FasterWhisperAdapter | WhisperCppAdapter:
        if os.environ.get("RAGBENCH_TRANSCRIPTION_PROVIDER", "faster-whisper") == "whisper.cpp":
            return WhisperCppAdapter()
        return FasterWhisperAdapter()


def provider_registry() -> ProviderRegistry:
    return ProviderRegistry()


def choose_document_parser(filename: str, prefer_docling: bool | None = None) -> dict[str, Any]:
    suffix = Path(filename).suffix.lower()
    use_docling = prefer_docling if prefer_docling is not None else os.environ.get("RAGBENCH_DOC_PARSER") == "docling"
    docling_available = shutil.which("docling") is not None
    if use_docling and docling_available:
        return {"parser": "docling", "available": True, "reason": "Docling requested and executable is available."}
    if use_docling and not docling_available:
        return {
            "parser": "fallback",
            "available": True,
            "warning": "Docling requested but unavailable; using deterministic parser fallback.",
        }
    if suffix == ".pdf":
        return {"parser": "pypdf", "available": True}
    if suffix == ".docx":
        return {"parser": "python-docx", "available": True}
    return {"parser": "plain-text", "available": True}


def provider_status() -> dict[str, Any]:
    from .provider_health import provider_health

    health = provider_health()
    ollama = health["ollama"]
    return {
        "mode": health["mode"],
        "ready": health["ready"],
        "defaults": default_model_guidance(),
        "llm_adapters": {
            "ollama": {
                "provider": "ollama",
                "model": DEFAULT_OLLAMA_LLM_MODEL,
                "available": bool(ollama.get("available") and ollama.get("models", {}).get(DEFAULT_OLLAMA_LLM_MODEL, {}).get("installed")),
                "setup": f"ollama pull {DEFAULT_OLLAMA_LLM_MODEL}",
            },
            "llama_cpp": LlamaCppAdapter().availability(),
        },
        "vlm_adapters": {
            "ollama": {
                "provider": "ollama_vlm",
                "model": DEFAULT_OLLAMA_VLM_MODEL,
                "available": bool(ollama.get("available") and ollama.get("models", {}).get(DEFAULT_OLLAMA_VLM_MODEL, {}).get("installed")),
                "setup": f"ollama pull {DEFAULT_OLLAMA_VLM_MODEL}",
            },
        },
        "embedding_adapters": {
            "deterministic": {"provider": "deterministic", "model": "deterministic_lexical", "available": health["deterministic_fallback"]},
            "ollama": {
                "provider": "ollama_embeddings",
                "model": DEFAULT_EMBEDDING_MODEL,
                "available": bool(ollama.get("available") and ollama.get("models", {}).get(DEFAULT_EMBEDDING_MODEL, {}).get("installed")),
                "setup": f"ollama pull {DEFAULT_EMBEDDING_MODEL}",
            },
        },
        "ocr_adapters": {
            "tesseract": TesseractOCRAdapter().availability(),
            "easyocr": EasyOCRAdapter().availability(),
        },
        "transcription_adapters": {
            "whisper_cpp": WhisperCppAdapter().availability(),
            "faster_whisper": FasterWhisperAdapter().availability(),
        },
        "parser_decisions": {
            "pdf": choose_document_parser("sample.pdf"),
            "docx": choose_document_parser("sample.docx"),
            "docling_preferred": choose_document_parser("sample.pdf", prefer_docling=True),
        },
    }
