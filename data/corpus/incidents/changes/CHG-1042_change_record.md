---
title: Change record CHG-1042: Reduce PayFlex connection pool max size from 200 to 50 to lower per-connection PSP fees
doc_type: change_record
date: 2026-03-09
author: Tom Weber
service: payment-gateway
---

# Change record CHG-1042

## Details

Service: payment-gateway. Type: normal change. Author: Tom Weber. Risk assessment: low. Implemented on 2026-03-09 at 10:30 UTC.

## Description

Reduce PayFlex connection pool max size from 200 to 50 to lower per-connection PSP fees. Motivation: PayFlex charges a fee per persistent connection. Peak usage in February was 35 connections, so 50 leaves headroom. Rollback: set PAYFLEX_POOL_MAX back to 200. Approved by the CAB on 2026-03-06.
