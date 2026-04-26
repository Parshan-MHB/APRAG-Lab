from __future__ import annotations

import math
import json
import shutil
import struct
import subprocess
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "sample-data" / "manual-test-suite"


def write_text_files() -> None:
    (OUT / "01_ragbench_requirements.txt").write_text(
        "\n".join(
            [
                "RAGBench manual test source.",
                "Decision: the product uses host Ollama for real LLM, VLM, and embedding benchmarks.",
                "The default models are qwen3:8b for LLM, qwen3-vl:4b for VLM, and bge-m3 for embeddings.",
                "The system must compare Traditional RAG, Agentic RAG, and Hybrid Graph RAG on the same knowledge base.",
                "Every factual answer must include citations from uploaded evidence.",
                "Uploads support text, PDF, DOCX, images, audio, and video.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (OUT / "02_architecture_notes.md").write_text(
        "\n".join(
            [
                "# Architecture Notes",
                "",
                "The API, web app, queue worker, SQLite, Chroma, and Qdrant run in Docker.",
                "Ollama, OCR, FFmpeg media extraction, and transcription run on the host laptop.",
                "Ollama runs on the host laptop at http://host.docker.internal:11434.",
                "SQLite is the required metadata store for this product scope.",
                "The graph layer stores entities such as API, Web, Worker, Chroma, Qdrant, SQLite, and Ollama.",
                "Hybrid Graph RAG should connect retrieved chunks with graph relationships before answering.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (OUT / "03_architecture_notes.markdown").write_text(
        "\n".join(
            [
                "# Markdown Extension Variant",
                "",
                "This file verifies the `.markdown` upload path.",
                "The benchmark lab should treat this as text evidence and preserve citations.",
                "Follow-up questions should reuse the active knowledge base version.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (OUT / "04_metrics_seed.csv").write_text(
        "\n".join(
            [
                "pipeline,expected_strength,risk",
                "traditional,fast grounded answers,misses multi-hop graph links",
                "agentic,tool trace and source inspection,slower model calls",
                "hybrid_graph,entity relationships and vector recall,depends on graph quality",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (OUT / "05_provider_settings.json").write_text(
        json.dumps(
            {
                "llm_model": "qwen3:8b",
                "vlm_model": "qwen3-vl:4b",
                "embedding_model": "bge-m3",
                "vector_stores": ["chroma", "qdrant"],
                "metadata_store": "sqlite",
                "runtime": "host_ollama_with_dockerized_app",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def write_pdf() -> None:
    lines = [
        "RAGBench Release Notes",
        "Release goal: verify upload, retrieval, citations, exports, and run history.",
        "Risk: local models may be unavailable or slow on some laptops.",
        "Mitigation: Settings shows host Ollama status and exact ollama pull commands.",
        "Acceptance: previous benchmark runs reopen without reprocessing uploaded files.",
    ]
    stream = "BT /F1 14 Tf 72 740 Td "
    stream += " T* ".join(f"({pdf_escape(line)}) Tj" for line in lines)
    stream += " ET"
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        f"<< /Length {len(stream.encode('utf-8'))} >>\nstream\n{stream}\nendstream",
    ]
    chunks = ["%PDF-1.4\n"]
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(sum(len(chunk.encode("utf-8")) for chunk in chunks))
        chunks.append(f"{index} 0 obj\n{obj}\nendobj\n")
    xref_offset = sum(len(chunk.encode("utf-8")) for chunk in chunks)
    chunks.append(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n")
    chunks.extend(f"{offset:010d} 00000 n \n" for offset in offsets[1:])
    chunks.append(f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n")
    (OUT / "06_release_notes.pdf").write_bytes("".join(chunks).encode("utf-8"))


def write_docx() -> None:
    document_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>Meeting Decisions</w:t></w:r></w:p>
    <w:p><w:r><w:t>The team decided to keep source evidence as the authority for follow-up questions.</w:t></w:r></w:p>
    <w:p><w:r><w:t>The benchmark comparison should explain why Traditional, Agentic, or Hybrid Graph is recommended.</w:t></w:r></w:p>
    <w:p><w:r><w:t>The Settings page must show host Ollama, model availability, vector store choice, and provider mode.</w:t></w:r></w:p>
  </w:body>
</w:document>
"""
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>
"""
    rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>
"""
    with zipfile.ZipFile(OUT / "07_meeting_decisions.docx", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", rels)
        archive.writestr("word/document.xml", document_xml)


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def drawtext_filter(lines: list[tuple[str, int, int, int, str]]) -> str:
    filters = []
    for index, (text, x, y, size, color) in enumerate(lines):
        text_path = OUT / f"_drawtext_{index}.txt"
        text_path.write_text(text, encoding="utf-8")
        filters.append(
            f"drawtext=fontfile='/System/Library/Fonts/Supplemental/Arial.ttf':"
            f"textfile='{text_path}':x={x}:y={y}:fontsize={size}:fontcolor={color}"
        )
    return ",".join(filters)


def ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg")


def say_path() -> str | None:
    return shutil.which("say")


def write_wav_fallback(path: Path) -> None:
    sample_rate = 16000
    seconds = 3
    frames = []
    for index in range(sample_rate * seconds):
        value = int(12000 * math.sin(2 * math.pi * 440 * index / sample_rate))
        frames.append(struct.pack("<h", value))
    data = b"".join(frames)
    header = (
        b"RIFF"
        + struct.pack("<I", 36 + len(data))
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
        + b"data"
        + struct.pack("<I", len(data))
    )
    path.write_bytes(header + data)


def write_media_files() -> None:
    ffmpeg = ffmpeg_path()
    image = OUT / "08_architecture_diagram.png"
    audio = OUT / "12_meeting_audio.wav"
    video = OUT / "16_product_demo_video.mp4"
    if ffmpeg:
        run(
            [
                ffmpeg,
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=white:s=1280x720:d=1",
                "-vf",
                drawtext_filter(
                    [
                        ("RAGBench Architecture", 60, 70, 48, "black"),
                        ("Web -> API -> Queue Worker", 60, 180, 36, "navy"),
                        ("Host Ollama: qwen3 8b, qwen3-vl 4b, bge-m3", 60, 270, 34, "darkgreen"),
                        ("Vector DB: Chroma or Qdrant | Metadata: SQLite", 60, 360, 34, "purple"),
                    ]
                ),
                "-frames:v",
                "1",
                "-update",
                "1",
                str(image),
            ]
        )
        run([ffmpeg, "-y", "-i", str(image), str(OUT / "09_architecture_diagram.jpg")])
        run([ffmpeg, "-y", "-i", str(image), str(OUT / "10_architecture_diagram.webp")])
        run([ffmpeg, "-y", "-i", str(image), str(OUT / "11_architecture_diagram.gif")])
        if say_path():
            aiff = OUT / "_meeting_audio.aiff"
            run(
                [
                    "say",
                    "-o",
                    str(aiff),
                    "Meeting decision: use host Ollama for real benchmark models. Keep SQLite metadata and cite uploaded evidence.",
                ]
            )
            run([ffmpeg, "-y", "-i", str(aiff), "-ar", "16000", "-ac", "1", str(audio)])
            aiff.unlink(missing_ok=True)
        else:
            write_wav_fallback(audio)
        run([ffmpeg, "-y", "-i", str(audio), str(OUT / "13_meeting_audio.mp3")])
        run([ffmpeg, "-y", "-i", str(audio), str(OUT / "14_meeting_audio.m4a")])
        run([ffmpeg, "-y", "-i", str(audio), str(OUT / "15_meeting_audio.ogg")])
        run(
            [
                ffmpeg,
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=white:s=1280x720:d=5",
                "-i",
                str(audio),
                "-vf",
                drawtext_filter(
                    [
                        ("Product Demo Video", 60, 70, 50, "black"),
                        ("Visible UI issue: Settings button overlaps the chart legend", 60, 190, 36, "red"),
                        ("Expected answer cites timestamp and frame evidence", 60, 300, 34, "navy"),
                    ]
                ),
                "-shortest",
                "-pix_fmt",
                "yuv420p",
                str(video),
            ]
        )
        run([ffmpeg, "-y", "-i", str(video), "-c", "copy", str(OUT / "17_product_demo_video.mov")])
        run([ffmpeg, "-y", "-i", str(video), "-c", "copy", str(OUT / "18_product_demo_video.mkv")])
        run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(video),
                "-c:v",
                "libvpx-vp9",
                "-b:v",
                "0",
                "-crf",
                "36",
                "-c:a",
                "libopus",
                str(OUT / "19_product_demo_video.webm"),
            ]
        )
    else:
        write_wav_fallback(audio)
        image.write_text("PNG unavailable because ffmpeg is not installed.\n", encoding="utf-8")
        video.write_text("MP4 unavailable because ffmpeg is not installed.\n", encoding="utf-8")


def write_readme() -> None:
    (OUT / "README.md").write_text(
        "\n".join(
            [
                "# RAGBench Manual Test Suite",
                "",
                "Use these files for real browser testing and demos.",
                "",
                "Files:",
                "- `01_ragbench_requirements.txt`: text source for core product requirements.",
                "- `02_architecture_notes.md`: Markdown source for architecture and graph entities.",
                "- `03_architecture_notes.markdown`: `.markdown` variant for text routing.",
                "- `04_metrics_seed.csv`: CSV variant for text routing.",
                "- `05_provider_settings.json`: JSON variant for text routing.",
                "- `06_release_notes.pdf`: PDF source with release, risk, and acceptance facts.",
                "- `07_meeting_decisions.docx`: DOCX source with meeting decisions.",
                "- `08_architecture_diagram.png`, `09_architecture_diagram.jpg`, `10_architecture_diagram.webp`, `11_architecture_diagram.gif`: image variants for OCR and VLM inspection.",
                "- `12_meeting_audio.wav`, `13_meeting_audio.mp3`, `14_meeting_audio.m4a`, `15_meeting_audio.ogg`: audio variants for transcription and timestamp citations.",
                "- `16_product_demo_video.mp4`, `17_product_demo_video.mov`, `18_product_demo_video.mkv`, `19_product_demo_video.webm`: video variants for transcript, OCR, frame, and VLM paths.",
                "",
                "Suggested benchmark questions:",
                "1. Which model runtime did the team choose and why?",
                "2. What are the default LLM, VLM, and embedding models?",
                "3. Which components run in Docker and which component runs on the host?",
                "4. What decision was made in the meeting audio?",
                "5. What UI issue is visible in the product demo video?",
                "6. Which pipeline should be recommended when graph relationships connect API, Worker, SQLite, and Ollama evidence?",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for existing in OUT.iterdir():
        if existing.is_file():
            existing.unlink()
    write_text_files()
    write_pdf()
    write_docx()
    write_media_files()
    write_readme()
    for temp in OUT.glob("_drawtext_*.txt"):
        temp.unlink(missing_ok=True)
    print(f"Wrote manual test dataset to {OUT}")


if __name__ == "__main__":
    main()
