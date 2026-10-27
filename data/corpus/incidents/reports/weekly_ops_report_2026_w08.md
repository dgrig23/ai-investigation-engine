---
title: Weekly operations report week 8
doc_type: ops_report
date: 2026-02-23
author: Giorgi Tsiklauri
---

# Weekly operations report, week 8 (2026-02-16 to 2026-02-22)

## Incidents

- INC-2033 on 2026-02-17: Duplicate free-spin bonuses, total duration of 38 minutes.

## Changes

- CHG-1021 (api-gateway): Raise rate limit for partner API keys from 300 to 500 requests per minute (follow-up)
- CHG-1031 (bonus-engine): Change idempotency key format for bonus credit requests
- CHG-1022 (api-gateway): Rotate TLS certificates for the public edge (follow-up)
- CHG-1023 (auth-service): Enable WebAuthn as an optional second factor (follow-up)
- CHG-1024 (auth-service): Shorten idle session timeout from 60 to 30 minutes (follow-up)

## Availability

Platform availability this week: 99.90%.
