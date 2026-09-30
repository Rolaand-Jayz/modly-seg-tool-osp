# 09 — Evidence fusion, conflicts, and user corrections

**What to build:** Combine segmentation, semantics, material identity, and PBR assertions into a deterministic Structured Asset view that preserves conflicts, refuses invalid cross-model confidence comparisons, and lets user-confirmed corrections supersede inference while remaining topology-bound.

**Blocked by:** 05 — Open-vocabulary part semantic identification; 06 — Material-region segmentation independent of parts; 07 — Material identity classification; 08 — PBR property recovery with synthetic validation.

**Status:** ready-for-agent

- [ ] The fusion logic represents compatible assertions together and records explicit conflict/ambiguous states when evidence disagrees.
- [ ] Integration fixtures prove that one semantic part can contain multiple material regions and that the same material identity can occur across multiple semantic parts without merging those parts.
- [ ] Raw confidence numbers from different adapters are never directly ranked as commensurate unless an explicit normalization/calibration rule exists.
- [ ] Deterministic rules may use within-adapter ranking, evidence kind, provenance quality, geometric consistency, and explicit calibration without hiding discarded candidates.
- [ ] A user-confirmed correction supersedes model inference for the same valid target while preserving the displaced model assertions in provenance/history.
- [ ] A topology-changing upstream result invalidates or marks affected corrections pending-remap/orphaned rather than attaching them to a different region.
- [ ] Re-running one inference adapter does not erase unrelated user corrections or assertions from independent capability families.
- [ ] Fusion/correction behavior is deterministic for identical inputs, versions, parameters, and seeds.
