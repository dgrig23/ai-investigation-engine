---
title: Postmortem PM-2041 (draft v0.9): Deposit processing delays (INC-2041)
doc_type: postmortem
date: 2026-03-16
author: Ana Lomidze
status: draft
---

## Summary

Status: DRAFT v0.9 - numbers not yet confirmed.

On 2026-03-12 deposits on the Acme Play platform were delayed and partly failed. Impact started at 14:05 UTC, when payment-gateway began timing out on deposit requests, and the incident was resolved at 15:52 UTC, a total duration of 102 minutes. Monitoring detected the problem at 14:20 UTC when the PaymentGatewayTimeoutRate alert fired. At the peak, 18% of deposit attempts failed. In total 41,300 deposit transactions were delayed and 2,140 deposits failed and had to be retried by players.

The payment service provider PayFlex was healthy during the whole incident; its status page and our account manager confirmed there was no PayFlex outage. The initial suspicion of a PayFlex outage was wrong.

## Affected services

payment-gateway: connection pool to PayFlex exhausted, deposit requests timed out.

fraud-scoring: model v3.2 responded slowly (p99 latency about 450 ms instead of 80 ms), which was the trigger.

wallet-service: deposits were credited late because payment confirmations arrived late.

ledger-service and withdrawals were not affected.

## Root cause

The root cause was connection pool exhaustion in payment-gateway, created by two changes that were each low risk on their own.

CHG-1042 (deployed 2026-03-09) reduced the PayFlex connection pool max size in payment-gateway from 200 to 50 to lower per-connection fees.

CHG-1051 (deployed 2026-03-12 at 09:15 UTC) released fraud model v3.2 in fraud-scoring. The new device-fingerprint features increased p99 scoring latency from 80 ms to about 450 ms.

payment-gateway calls fraud-scoring synchronously while it holds a PayFlex pool connection, so every deposit held a connection roughly six times longer than before. With only 50 connections, the pool was exhausted when evening traffic grew, requests queued and timed out, and mobile clients retried immediately, which amplified the load (a retry storm).

## Timeline (UTC)

09:15 CHG-1051 deploys fraud model v3.2.

14:05 First payment-gateway timeouts on deposit requests. Impact began.

14:20 PaymentGatewayTimeoutRate alert fires; Nika Kapanadze (on-call) starts the incident.

14:25 PayFlex status page checked: all systems operational.

14:38 Pool usage observed at 50 of 50 connections on all payment-gateway pods.

14:52 Maya Chen notices fraud-scoring p99 latency of 470 ms since the morning deploy.

15:20 Decision to roll back fraud-scoring to v3.1.

15:31 Rollback completed (CHG-1053).

15:40 Pool usage drops to 22 of 50; timeouts stop. Mitigated.

15:52 Incident resolved.

## Resolution and follow-up

fraud-scoring was rolled back to v3.1 (CHG-1053). On 2026-03-13 the pool size was restored to 200 (CHG-1055).

The financial impact of the incident is still being assessed by the Finance team and is not part of this document.

Action items: AI-2041-1 load-test fraud-scoring releases together with payment-gateway (owner: Maya Chen). AI-2041-2 alert when pool saturation exceeds 80% (owner: Tom Weber). AI-2041-3 change advisory board must require a dependency review for connection pool changes (owner: Ana Lomidze). AI-2041-4 add exponential backoff with jitter to mobile client retries (owner: David Okafor).
