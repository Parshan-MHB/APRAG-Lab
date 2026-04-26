# Northstar Field Service Case File

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
