---
title: Weekly operations report week 11
doc_type: ops_report
date: 2026-03-16
author: Giorgi Tsiklauri
---

# Weekly operations report, week 11 (2026-03-09 to 2026-03-15)

## Incidents

- INC-2044 on 2026-03-15: Nightly regulatory report delayed, total duration of 130 minutes.
- INC-2041 on 2026-03-12: Deposit processing delays, total duration of 45 minutes. Numbers are preliminary.

## Changes

- CHG-1032 (notification-service): Add Georgian language templates for deposit receipts (follow-up)
- CHG-1042 (payment-gateway): Reduce PayFlex connection pool max size from 200 to 50 to lower per-connection PSP fees
- CHG-1033 (notification-service): Move SMS sending to a second provider for redundancy (follow-up)
- CHG-1051 (fraud-scoring): Deploy fraud model v3.2 with new device-fingerprint features
- CHG-1053 (fraud-scoring): Roll back fraud model to v3.1 during INC-2041
- CHG-1055 (payment-gateway): Restore PayFlex connection pool max size to 200 after INC-2041
- CHG-1034 (reporting-service): Increase Spark executor memory for nightly regulatory report (follow-up)
- CHG-1035 (reporting-service): Add daily responsible-gaming limits report (follow-up)

## Availability

Platform availability this week: 99.71%.
