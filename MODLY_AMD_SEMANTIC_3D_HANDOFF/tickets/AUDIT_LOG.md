# Modly AMD-Native Semantic 3D — Ticket Audit Log

**Source:** `MODLY_AMD_SEMANTIC_3D_SPEC_AUDITED.md` revision 2.0  
**Final ticket count:** 13  
**Stopping rule:** stop only after three consecutive audit passes produce no ticket or dependency edits.  
**Result:** satisfied.

## Correction loop 1 — Spec/testing coverage

Findings corrected before any clean streak was counted:

- Added visible trusted/pinned reference-adapter requirements.
- Expanded AMD Runtime failure diagnostics and stale-worker cleanup behavior.
- Added explicit single-image inferred-vs-observed geometry provenance.
- Added segmentation overlap policy, declared ground-truth metric, and unsupported-geometry failure behavior.
- Added fixture/local-ontology semantics evaluation requirements.
- Added numeric PBR validation thresholds and novel-light validation ownership.
- Added complete glTF coordinate, color-space, tangent-normal, bump/normal distinction, and non-glTF source-basis round-trip requirements.
- Expanded the final AMD compatibility report to cover host RAM, first/warm latency, MIGraphX correctness, ROCm fallback reasons, HIP/native work, CPU fallback, known-good quality comparison, and unresolved blockers.

## Correction loop 2 — Dependency/seam integrity

Findings corrected:

- Removed the false dependency from material-region segmentation to part segmentation; the two capabilities are now independently implementable as required by the spec.
- Moved cross-capability part/material relationship proofs to the evidence/fusion integration ticket.
- Added AMD Runtime blockers to semantic and material-classification adapters.
- Added predeclared quality/correctness thresholds and selection rubrics to probe-driven generator, part-segmentation, material-region, and PBR tickets.

## Correction loop 3 — Index consistency

Finding corrected:

- Updated the ticket index after the blocker graph changed so the documented dependency order matches the ticket files.

## Correction loop 4 — Fixture ownership and measurable acceptance

Findings corrected:

- Assigned the complete golden-fixture suite to the RX 7900 GRE headless POC acceptance ticket.
- Added predeclared semantic coverage/correctness thresholds.
- Added predeclared material-classification coverage/correctness thresholds.
- Required final adapter acceptance thresholds to be declared before the final run rather than judged visually after the fact.

## Clean pass 1 — Structure and dependency graph

**Result: CLEAN — no edits.**

Checked:

- 13 numbered one-file-per-ticket artifacts.
- Every ticket uses the ready-for-agent local-ticket contract.
- Exactly two initial frontier tickets: Structured Asset contract and AMD Runtime.
- No missing blockers, forward blockers, or dependency cycles.
- Dependency index matches ticket files.
- Tickets remain bounded enough for fresh-context implementation.

## Clean pass 2 — Spec traceability

**Result: CLEAN — no edits.**

Checked:

- All implementation-decision families have ticket ownership.
- All 14 POC acceptance gates have explicit final acceptance coverage.
- Testing contracts cover generator provenance, segmentation metrics, semantic ontology/unknown behavior, material-region relationships, material classification, PBR ground truth, evidence fusion, correction behavior, cache invalidation, export interchange, and AMD compatibility reporting.
- The full golden-fixture set has a single explicit owner.
- Out-of-scope decisions are not accidentally promoted into mandatory implementation work.

## Clean pass 3 — Agent implementation readiness

**Result: CLEAN — no edits.**

Checked:

- Probe-driven tickets declare gates before model selection.
- Backend/fallback policy is owned by the AMD Runtime ticket.
- Shared HIP primitives are not prematurely introduced before a justified second consumer.
- Named multi-view ports are not assumed to exist for the POC.
- Confidence arbitration, correction precedence, cache invalidation, glTF export, cloud-free operation, and viewer strategy are explicit rather than left for agents to invent.
- High-risk tickets define failure behavior.
- No ticket contains stale-prone implementation file paths or code snippets.
- Ticket size/acceptance surfaces remain suitable for one fresh implementation context.

## Final dependency frontier

Tickets **01** and **02** can start immediately and in parallel. Once both land, geometry generation and native-3D part segmentation can proceed independently; material-region work can also proceed without waiting for semantic-part segmentation, preserving the spec's separation of concerns.