from __future__ import annotations

import csv
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "sample-data" / "manual-test-suite"


INCIDENT_BRIEF = """# Northstar Field Service Case File

## Scenario

Northstar Appliances runs connected refrigeration systems for grocery and healthcare customers in the Pacific Northwest. In March 2026 the company investigated repeated compressor shutdowns after a firmware rollout to the NS-900 controller. The same customer accounts appear in support tickets, service metrics, and the operations review so benchmark questions require cross-source reasoning rather than single-snippet lookup.

## Key Entities

- Customer: Lakeside Clinic, account C-104, site SEA-17.
- Customer: Harbor Market, account C-118, site PDX-04.
- Customer: Pine Ridge Foods, account C-122, site BOI-02.
- Controller firmware under review: NS-900 version 4.8.2.
- Replacement firmware approved for staged rollout: NS-900 version 4.8.3.
- On-call service lead: Priya Shah.
- Firmware owner: Marco Diaz.
- Customer success owner: Elena Brooks.

## Incident Timeline

On 2026-03-04 at 02:14 local time, Lakeside Clinic reported that vaccine freezer A entered safe mode after three compressor restart attempts. Ticket INC-1043 was opened with severity P1 because the freezer held temperature-sensitive inventory. Logs showed firmware 4.8.2 retried the condenser fan check too aggressively after a transient sensor fault. The retry storm increased controller CPU load, delayed telemetry, and triggered safe mode. The immediate workaround was to pin the controller to profile cold_chain_safe and restart the condenser fan service.

On 2026-03-05, Harbor Market opened ticket INC-1051 for intermittent alarm noise on aisle freezer 3. The unit also ran firmware 4.8.2, but the logs showed no safe-mode transition and no inventory loss. The issue was classified as P2 because it was customer-visible but did not break cooling. Harbor Market needs proactive communication, not an SLA credit.

On 2026-03-08, Pine Ridge Foods opened ticket INC-1062 for a delayed defrost cycle on controller 7. That site was already on firmware 4.8.3 from a pilot group. The root cause was a misconfigured night schedule, not the firmware retry defect.

## Operations Review

The review board decided not to roll back all customers to firmware 4.7.9 because rollback would disable new telemetry compression needed by the service analytics program. Instead, Marco Diaz will ship firmware 4.8.3 to cold-chain healthcare accounts first, then grocery accounts. Priya Shah owns the field runbook update. Elena Brooks owns customer communication for Lakeside Clinic and Harbor Market.

Lakeside Clinic qualifies for an SLA credit because the P1 outage risked regulated medical inventory for 47 minutes and required manual intervention. Harbor Market does not qualify for an SLA credit because cooling stayed within range. Pine Ridge Foods does not qualify because the issue was a local schedule configuration error.

## Benchmark-Relevant Facts

- The highest business risk is Lakeside Clinic because medical inventory was at risk and a P1 ticket was opened.
- Ticket INC-1043 recorded 184000 dollars of inventory at risk and 47 minutes of downtime.
- The technical root cause for Lakeside is the NS-900 4.8.2 condenser fan retry storm after a transient sensor fault.
- The approved mitigation is staged firmware 4.8.3 rollout plus the cold_chain_safe profile for healthcare accounts.
- A good answer should connect incident notes with ticket metrics and cite both the case file and the CSV evidence.
"""


REVIEW_NOTES = """# March 2026 Service Review Notes

## Meeting Summary

The service review meeting on 2026-03-11 compared the March incident tickets with fleet metrics. The team agreed that the problem was not a general refrigeration failure. It was a firmware-specific control-loop defect affecting NS-900 version 4.8.2 when condenser fan sensor data briefly dropped out.

Priya Shah reported that field technicians could apply the cold_chain_safe profile in under 12 minutes. Marco Diaz confirmed that firmware 4.8.3 changes the retry policy from three immediate retries to one retry followed by a 90-second cooldown. Elena Brooks requested separate customer messaging for healthcare customers and grocery customers.

## Decisions

1. Do not perform a broad rollback to firmware 4.7.9.
2. Deploy firmware 4.8.3 first to healthcare cold-chain sites, starting with Lakeside Clinic SEA-17.
3. Send Harbor Market a preventive notice and maintenance window, but no SLA credit.
4. Issue Lakeside Clinic an SLA credit and provide a compliance incident summary.
5. Keep Pine Ridge Foods in the pilot group and correct its night schedule.

## Open Actions

| Owner | Action | Due Date | Related Evidence |
| --- | --- | --- | --- |
| Priya Shah | Publish updated cold_chain_safe runbook | 2026-03-13 | INC-1043 |
| Marco Diaz | Release firmware 4.8.3 staged rollout package | 2026-03-14 | INC-1043, INC-1051 |
| Elena Brooks | Send Lakeside Clinic incident summary and credit memo | 2026-03-15 | INC-1043 |
| Elena Brooks | Send Harbor Market preventive maintenance notice | 2026-03-15 | INC-1051 |

## Risks To Monitor

The review board marked three watch items. First, field teams must verify that telemetry delay clears after 4.8.3. Second, customer success must avoid promising SLA credits to sites that did not lose cooling. Third, analytics must separate firmware defects from local schedule configuration issues, because Pine Ridge Foods looked similar at first but had a different root cause.
"""


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


def write_csv(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(TICKET_ROWS[0].keys()))
        writer.writeheader()
        writer.writerows(TICKET_ROWS)


def write_readme() -> None:
    (OUT / "README.md").write_text(
        """# Northstar Field Service Scenario

This sample is designed for realistic RAG benchmark testing, not file-extension coverage. It uses only Markdown and CSV, but the evidence is connected across customer accounts, tickets, owners, firmware versions, mitigation decisions, and financial/SLA outcomes.

Files:

- `01_northstar_incident_brief.md`: case narrative, timeline, entities, root cause, and decisions.
- `02_service_tickets.csv`: structured ticket metrics, severity, downtime, risk, owners, and SLA flags.
- `03_service_review_notes.md`: meeting decisions, owners, due dates, and risk interpretation.

Suggested benchmark questions:

1. What caused the Lakeside Clinic outage, and which ticket metrics prove it was the highest-risk case?
2. Which customers were affected by firmware 4.8.2, and why did only one qualify for an SLA credit?
3. What did the review board decide about rollback versus staged firmware 4.8.3 rollout?
4. Compare Lakeside Clinic, Harbor Market, and Pine Ridge Foods by root cause, severity, and mitigation.
5. Which owner is responsible for each follow-up action, and what evidence connects the owner to the ticket?
6. Was Pine Ridge Foods part of the firmware defect, or was it a different issue?
""",
        encoding="utf-8",
    )


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "01_northstar_incident_brief.md").write_text(INCIDENT_BRIEF, encoding="utf-8")
    write_csv(OUT / "02_service_tickets.csv")
    (OUT / "03_service_review_notes.md").write_text(REVIEW_NOTES, encoding="utf-8")
    write_readme()
    print(f"Wrote realistic benchmark scenario to {OUT}")


if __name__ == "__main__":
    main()
