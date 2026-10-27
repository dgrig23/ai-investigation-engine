---
title: Change record CHG-1047: Add index idx_entries_account_created on ledger_entries (online migration)
doc_type: change_record
date: 2026-03-18
author: Levan Abashidze
service: ledger-service
---

# Change record CHG-1047

## Details

Service: ledger-service. Type: normal change. Author: Levan Abashidze. Risk assessment: medium. Implemented on 2026-03-18 at 22:00 UTC.

## Description

Add index idx_entries_account_created on ledger_entries (online migration). Migration adds idx_entries_account_created on ledger_entries (about 480 million rows) to speed up account statement queries. Planned for a low-traffic window at 22:00 UTC.
