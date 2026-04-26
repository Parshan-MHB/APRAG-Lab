# Northstar Multimodal Test Scenario

This sample is designed for realistic multimodal RAG benchmark testing. It uses one image, one audio memo, one CSV, and one PDF. The evidence is connected across customers, ticket IDs, owners, firmware versions, incident metrics, SLA decisions, and mitigation actions.

Files:

- `01_vaccine_administration_event.jpg`: real-world clinical vaccine administration photo with no added text overlay. Source: https://commons.wikimedia.org/wiki/File:Nurse_administers_a_vaccine.jpg
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
