# 10 — Content-addressed resumability and targeted re-run

**What to build:** Make expensive semantic-3D stages durable and resumable so a headless workflow can reuse valid intermediate artifacts, re-run only a selected stage plus true dependents, and invalidate caches correctly when topology, adapter/weights versions, parameters, or semantically relevant inputs change.

**Blocked by:** 01 — Structured Asset headless round-trip; 03 — Select and integrate the reference geometry generator; 04 — Native-3D part segmentation on 16 GB AMD; 08 — PBR property recovery with synthetic validation; 09 — Evidence fusion, conflicts, and user corrections.

**Status:** ready-for-agent

- [ ] Every major stage emits a durable artifact/result with a digest and a cache identity derived from semantically relevant input digests, adapter revision, weights digest, parameters, schema version, and other runtime factors that can change output semantics.
- [ ] Interrupting a workflow after a completed stage and restarting reuses valid prior artifacts instead of recomputing from the beginning.
- [ ] Selecting one stage for re-run invalidates/recomputes only that stage and dependency descendants; unrelated branches remain reusable.
- [ ] A topology-changing geometry/segmentation result invalidates topology-bound part/material mappings, assertions, and affected corrections according to the contract.
- [ ] A semantic-label-only change does not invalidate geometry, part segmentation, or independent PBR data unless an explicit dependency declares otherwise.
- [ ] Failed/partial stage outputs are never promoted to valid cache entries.
- [ ] Cache reuse is observable in headless diagnostics and preserves original provenance while recording the new workflow/run identity.
