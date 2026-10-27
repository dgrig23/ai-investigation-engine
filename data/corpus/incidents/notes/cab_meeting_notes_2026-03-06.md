---
title: Change advisory board (CAB) notes 2026-03-06
doc_type: notes
date: 2026-03-06
author: Ana Lomidze
---

# Change advisory board notes, 6 March 2026

Attendees: Ana Lomidze (chair), Tom Weber, Levan Abashidze, Giorgi Tsiklauri.

## CHG-1042: reduce PayFlex connection pool to 50

Presented by Tom Weber. PayFlex charges per persistent connection; reducing the pool from 200 to 50 saves about EUR 1,800 per month. Current peak usage was measured at 35 connections. Assessed as low risk. Approved by Ana Lomidze. No dependency review was done.

## CHG-1047: index on ledger_entries

Presented by Levan Abashidze. Approved on condition that it runs in a low-traffic window.

## Other

The fraud-scoring v3.2 model release (later CHG-1051) was not on the agenda; model releases are standard changes and do not require CAB approval.
