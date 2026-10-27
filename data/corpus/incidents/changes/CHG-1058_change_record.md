---
title: Change record CHG-1058: Rebuild idx_entries_account_created with CREATE INDEX CONCURRENTLY after INC-2047
doc_type: change_record
date: 2026-03-20
author: Levan Abashidze
service: ledger-service
---

# Change record CHG-1058

## Details

Service: ledger-service. Type: normal change. Author: Levan Abashidze. Risk assessment: medium. Implemented on 2026-03-20 at 21:00 UTC.

## Description

Rebuild idx_entries_account_created with CREATE INDEX CONCURRENTLY after INC-2047. Follow-up to INC-2047: rebuild the index with CREATE INDEX CONCURRENTLY, which does not block writes.
