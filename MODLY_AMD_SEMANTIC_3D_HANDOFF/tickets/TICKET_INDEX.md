# Modly AMD-Native Semantic 3D — Ticket Index

Status: audited ticket set generated from `MODLY_AMD_SEMANTIC_3D_SPEC_AUDITED.md`; three consecutive clean audit passes achieved.

## Frontier at start

- 01 — Structured Asset headless round-trip
- 02 — AMD Runtime proof through Modly

## Dependency order

01 → establishes Structured Asset/extension contract.
02 → establishes AMD Runtime independently.
03 → geometry generation, blocked by 01+02.
04 → native-3D part segmentation, blocked by 01+02.
05 → part semantics, blocked by 01+02+04.
06 → material-region segmentation, blocked by 01+02.
07 → material identity, blocked by 01+02+06.
08 → PBR recovery, blocked by 01+02+06.
09 → fusion/corrections, blocked by 05+06+07+08.
10 → caching/resume, blocked by 01+03+04+08+09.
11 → export/validation, blocked by 01+07+08+09+10.
12 → full RX 7900 GRE POC acceptance, blocked by 03–11 as listed in the ticket.
13 → interactive workflow polish, blocked by 12.
