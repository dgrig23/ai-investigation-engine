---
title: Incident report INC-2041: Deposits delayed
doc_type: incident_report
date: 2026-03-12
author: Nika Kapanadze
severity: SEV-1
status: initial
---

# Incident report INC-2041: Deposits delayed

## Overview

Written by the on-call engineer at 16:30 UTC on 2026-03-12, shortly after the incident was closed. Details may change after the postmortem.

Impact began at 14:20 UTC and the incident lasted approximately 60 minutes. Players saw deposits stuck in "pending" and some deposits failed.

## Suspected cause

Root cause: suspected outage at our payment provider PayFlex, because payment-gateway reported timeouts on PayFlex calls. The status page was updated to say a third-party provider was involved.

## Affected services

payment-gateway (deposit requests timing out).

## Actions taken

The team investigated PayFlex connectivity and later rolled back a fraud-scoring release as a precaution. Deposits recovered afterwards. A full postmortem (PM-2041) will follow.
