---
title: Incident report INC-2044: Nightly regulatory report delayed
doc_type: incident_report
date: 2026-03-15
severity: SEV-4
---

## Summary

Nightly regulatory report delayed (SEV-4). On 2026-03-15 the incident started at 02:00 UTC and was resolved at 04:10 UTC, a duration of 130 minutes. Impact: the regulatory report was submitted two hours late (still within the deadline).

## Affected services

reporting-service (owner: Data team).

## Root cause

The root cause was the nightly Spark job ran out of memory after data growth.

## Follow-up

Action items are tracked in the reliability review.
