---
title: Postmortem PM-2047: Withdrawal processing stalled (INC-2047)
doc_type: postmortem
date: 2026-03-24
author: Levan Abashidze
---

# Postmortem PM-2047: Withdrawal processing stalled (INC-2047)

## Summary

From 22:04 to 23:41 UTC on 2026-03-18 (97 minutes) withdrawals were stalled. 6,870 withdrawal requests were delayed; none were lost. Root cause: the ledger-service migration CHG-1047 built an index on ledger_entries without CONCURRENTLY and locked the table for writes.

## Why it happened

The migration was approved by the change advisory board on 2026-03-06 with the condition to run it in a low-traffic window. 22:00 UTC is a low-traffic window for deposits, but it is the daily peak for withdrawals because the withdrawal batch from the bonus-engine cashback programme runs at 22:00. The migration tool's review checklist did not flag the missing CONCURRENTLY keyword.

## Affected services

ledger-service, wallet-service and payment-gateway (payouts). Deposits were credited, but with delayed ledger entries.

## Action items

AI-2047-1 lint migrations for CREATE INDEX without CONCURRENTLY (owner: Levan Abashidze). AI-2047-2 document the withdrawal peak at 22:00 UTC in the change calendar (owner: Ana Lomidze).
