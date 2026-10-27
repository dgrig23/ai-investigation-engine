---
title: Change record CHG-1031: Change idempotency key format for bonus credit requests
doc_type: change_record
date: 2026-02-16
author: Sara Klein
service: bonus-engine
---

# Change record CHG-1031

## Details

Service: bonus-engine. Type: normal change. Author: Sara Klein. Risk assessment: medium. Implemented on 2026-02-16 at 10:00 UTC.

## Description

Change idempotency key format for bonus credit requests. The bonus credit idempotency key now includes the campaign version. Retries generated during deploy used the old format.
