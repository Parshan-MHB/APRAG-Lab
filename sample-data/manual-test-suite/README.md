# Northstar Field Service Scenario

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
