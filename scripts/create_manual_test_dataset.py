from __future__ import annotations

import csv
import shutil
import subprocess
import wave
from pathlib import Path
from urllib.request import urlretrieve


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "sample-data" / "manual-test-suite"

IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8b/Nurse_administers_a_vaccine.jpg/1280px-Nurse_administers_a_vaccine.jpg"
IMAGE_SOURCE = "https://commons.wikimedia.org/wiki/File:Nurse_administers_a_vaccine.jpg"

AUDIO_SCRIPT = (
    "Dispatch memo for incident INC-1043. Lakeside Clinic vaccine freezer A entered safe mode "
    "after a condenser fan retry storm. Priya Shah applied the cold chain safe profile and restarted "
    "the condenser fan service. Harbor Market stayed within range and should receive a preventive "
    "notice, not an SLA credit."
)

PDF_LINES = [
    "Northstar Appliances - March 2026 Incident Review",
    "",
    "Scenario",
    "Northstar Appliances runs connected refrigeration systems for grocery and healthcare customers.",
    "In March 2026, the team investigated compressor shutdowns after an NS-900 firmware rollout.",
    "The same customer accounts appear in the image, audio memo, CSV tickets, and this PDF review.",
    "",
    "Key Entities",
    "Lakeside Clinic, account C-104, site SEA-17, operated vaccine freezer A.",
    "Harbor Market, account C-118, site PDX-04, operated aisle freezer 3.",
    "Pine Ridge Foods, account C-122, site BOI-02, operated controller 7.",
    "Firmware under review was NS-900 version 4.8.2.",
    "Replacement firmware approved for rollout was NS-900 version 4.8.3.",
    "Priya Shah owned the field runbook update.",
    "Marco Diaz owned the firmware rollout.",
    "Elena Brooks owned customer communication.",
    "",
    "Timeline And Root Cause",
    "On 2026-03-04 at 02:14 local time, Lakeside Clinic opened ticket INC-1043.",
    "Vaccine freezer A entered safe mode after three compressor restart attempts.",
    "The case was severity P1 because temperature-sensitive medical inventory was at risk.",
    "Logs showed firmware 4.8.2 retried the condenser fan check too aggressively after a transient sensor fault.",
    "The retry storm increased controller CPU load, delayed telemetry, and triggered safe mode for 47 minutes.",
    "The immediate workaround was cold_chain_safe plus a condenser fan service restart.",
    "CSV metrics for INC-1043 were severity P1, 184000 dollars of inventory at risk, and 47 downtime minutes.",
    "",
    "Review Board Decisions",
    "The board decided not to roll back every customer to firmware 4.7.9.",
    "Rollback would disable telemetry compression required by service analytics.",
    "Instead, firmware 4.8.3 rolls out first to healthcare cold-chain accounts, then grocery accounts.",
    "Lakeside Clinic qualifies for an SLA credit and a compliance incident summary.",
    "Harbor Market receives a preventive maintenance notice, but no SLA credit, because cooling stayed in range.",
    "Pine Ridge Foods stays in the pilot group because its delayed defrost was a local night schedule error.",
]

TICKET_ROWS = [
    {
        "ticket_id": "INC-1043",
        "date": "2026-03-04",
        "customer": "Lakeside Clinic",
        "account_id": "C-104",
        "site_id": "SEA-17",
        "asset": "vaccine freezer A",
        "firmware": "NS-900 4.8.2",
        "severity": "P1",
        "symptom": "safe mode after three compressor restart attempts",
        "cooling_within_range": "no",
        "inventory_at_risk_usd": "184000",
        "downtime_minutes": "47",
        "root_cause": "condenser fan retry storm after transient sensor fault",
        "mitigation": "cold_chain_safe profile and staged firmware 4.8.3 rollout",
        "sla_credit": "yes",
        "owner": "Priya Shah",
    },
    {
        "ticket_id": "INC-1051",
        "date": "2026-03-05",
        "customer": "Harbor Market",
        "account_id": "C-118",
        "site_id": "PDX-04",
        "asset": "aisle freezer 3",
        "firmware": "NS-900 4.8.2",
        "severity": "P2",
        "symptom": "intermittent alarm noise",
        "cooling_within_range": "yes",
        "inventory_at_risk_usd": "0",
        "downtime_minutes": "0",
        "root_cause": "same firmware retry warning but no safe-mode transition",
        "mitigation": "preventive maintenance notice and firmware 4.8.3 maintenance window",
        "sla_credit": "no",
        "owner": "Elena Brooks",
    },
    {
        "ticket_id": "INC-1062",
        "date": "2026-03-08",
        "customer": "Pine Ridge Foods",
        "account_id": "C-122",
        "site_id": "BOI-02",
        "asset": "controller 7",
        "firmware": "NS-900 4.8.3",
        "severity": "P3",
        "symptom": "delayed defrost cycle",
        "cooling_within_range": "yes",
        "inventory_at_risk_usd": "0",
        "downtime_minutes": "0",
        "root_cause": "local night schedule misconfiguration",
        "mitigation": "correct night schedule and keep site in pilot group",
        "sla_credit": "no",
        "owner": "Marco Diaz",
    },
    {
        "ticket_id": "INC-1069",
        "date": "2026-03-10",
        "customer": "Lakeside Clinic",
        "account_id": "C-104",
        "site_id": "SEA-17",
        "asset": "backup freezer B",
        "firmware": "NS-900 4.8.3",
        "severity": "P3",
        "symptom": "post-mitigation telemetry delay check",
        "cooling_within_range": "yes",
        "inventory_at_risk_usd": "0",
        "downtime_minutes": "0",
        "root_cause": "verification event after firmware 4.8.3 pilot",
        "mitigation": "monitor telemetry delay for 72 hours",
        "sla_credit": "no",
        "owner": "Priya Shah",
    },
]


def escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def write_pdf(path: Path, lines: list[str]) -> None:
    text_commands = ["BT", "/F1 11 Tf", "50 760 Td", "14 TL"]
    for line in lines:
        text_commands.append(f"({escape_pdf_text(line)}) Tj")
        text_commands.append("T*")
    text_commands.append("ET")
    stream = "\n".join(text_commands).encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
    ]

    content = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(content))
        content.extend(f"{index} 0 obj\n".encode("ascii"))
        content.extend(obj)
        content.extend(b"\nendobj\n")
    xref_offset = len(content)
    content.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    content.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        content.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    content.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    path.write_bytes(bytes(content))


def write_csv(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(TICKET_ROWS[0].keys()))
        writer.writeheader()
        writer.writerows(TICKET_ROWS)


def write_silent_fallback_wav(path: Path, seconds: int = 2) -> None:
    sample_rate = 16000
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * sample_rate * seconds)


def write_audio(path: Path) -> None:
    tmp_aiff = path.with_suffix(".aiff")
    try:
        subprocess.run(["say", "-o", str(tmp_aiff), AUDIO_SCRIPT], check=True)
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(tmp_aiff), "-ar", "16000", "-ac", "1", str(path)],
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        write_silent_fallback_wav(path)
    finally:
        tmp_aiff.unlink(missing_ok=True)


def write_image(path: Path) -> None:
    try:
        urlretrieve(IMAGE_URL, path)
    except Exception:
        cached = Path("/tmp/nurse_administers_vaccine.jpg")
        if cached.exists():
            shutil.copyfile(cached, path)
        else:
            raise


def write_readme() -> None:
    (OUT / "README.md").write_text(
        f"""# Northstar Multimodal Test Scenario

This sample is designed for realistic multimodal RAG benchmark testing. It uses one image, one audio memo, one CSV, and one PDF. The evidence is connected across customers, ticket IDs, owners, firmware versions, incident metrics, SLA decisions, and mitigation actions.

Files:

- `01_vaccine_administration_event.jpg`: real-world clinical vaccine administration photo with no added text overlay. Source: {IMAGE_SOURCE}
- `02_lakeside_dispatch_memo.wav`: spoken dispatch memo for INC-1043 and Priya Shah's field action.
- `03_service_tickets.csv`: structured ticket metrics, severity, downtime, inventory risk, owners, and SLA flags.
- `04_incident_review.pdf`: incident narrative, root cause, rollback decision, rollout plan, and customer outcomes.

Suggested benchmark questions:

1. What caused the Lakeside Clinic outage, and which CSV ticket metrics prove it was the highest-risk case?
2. What does the vaccination image show, and how does it relate to the Lakeside vaccine freezer incident?
3. What did the audio dispatch memo say Priya Shah did for INC-1043?
4. Which customers were on firmware 4.8.2, and why did only Lakeside qualify for an SLA credit?
5. What did the review board decide about rollback versus staged firmware 4.8.3 rollout?
6. Was Pine Ridge Foods part of the firmware defect, or was it a different issue?
""",
        encoding="utf-8",
    )


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    write_image(OUT / "01_vaccine_administration_event.jpg")
    write_audio(OUT / "02_lakeside_dispatch_memo.wav")
    write_csv(OUT / "03_service_tickets.csv")
    write_pdf(OUT / "04_incident_review.pdf", PDF_LINES)
    write_readme()
    print(f"Wrote multimodal benchmark scenario to {OUT}")


if __name__ == "__main__":
    main()
