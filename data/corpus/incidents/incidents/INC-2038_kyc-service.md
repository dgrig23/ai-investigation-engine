---
title: Incident report INC-2038: KYC document upload failures
doc_type: incident_report
date: 2026-03-03
severity: SEV-2
---

## Summary

KYC document upload failures (SEV-2). On 2026-03-03 the incident started at 07:45 UTC and was resolved at 08:37 UTC, a duration of 52 minutes. Impact: new players could not upload identity documents.

## Affected services

kyc-service (owner: Compliance Engineering).

## Root cause

The root cause was an expired TLS certificate on the kyc-service document storage endpoint.

## Follow-up

Action items are tracked in the reliability review.
