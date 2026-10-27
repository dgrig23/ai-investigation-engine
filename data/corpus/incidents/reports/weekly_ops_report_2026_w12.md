---
title: Weekly operations report week 12
doc_type: ops_report
date: 2026-03-23
author: Giorgi Tsiklauri
---

# Weekly operations report, week 12 (2026-03-16 to 2026-03-22)

## Incidents

- INC-2047 on 2026-03-18: Withdrawal processing stalled, total duration of 97 minutes.

## Changes

- CHG-1036 (wallet-service): Add metrics for balance cache hit rate (follow-up)
- CHG-1047 (ledger-service): Add index idx_entries_account_created on ledger_entries (online migration)
- CHG-1037 (wallet-service): Upgrade JVM from 17 to 21 (follow-up)
- CHG-1058 (ledger-service): Rebuild idx_entries_account_created with CREATE INDEX CONCURRENTLY after INC-2047
- CHG-1038 (ledger-service): Add read replica for reporting queries (follow-up)

## Availability

Platform availability this week: 99.71%.
