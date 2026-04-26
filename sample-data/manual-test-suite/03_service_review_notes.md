# March 2026 Service Review Notes

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
