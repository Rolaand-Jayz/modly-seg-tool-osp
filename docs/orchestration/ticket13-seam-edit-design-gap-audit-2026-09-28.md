# Ticket 13 seam-edit design-gap audit — 2026-09-28

## Scope and evidence boundary

Source-only contract audit of the final audited spec, Tickets 09 and 13, and the current Structured Asset schema. No code, fixtures, target hardware, or model behavior was evaluated. No implementation, test, acceptance, or learning result is claimed. The workspace did not expose Git metadata (`git rev-parse HEAD` failed: not a Git repository), so source identity is recorded with SHA-256 digests.

## Existing contracts

- The canonical asset includes topology revision, part/material mappings, assertions, provenance, user corrections, validation, and stage artifacts; the sidecar carries run history (§4, spec lines 128–130).
- Geometry mappings can target mesh/primitive/face/vertex/UV-region elements and must be bound to topology revision. After topology changes, downstream mappings require proven correspondence or invalidation/orphaning (§5–6, spec lines 132–140).
- User corrections supersede competing inference for the same valid target; invalid targets become orphaned or pending-remap (§17, spec lines 192–194). Ticket 09 also requires retaining displaced assertions in provenance/history, preserving unrelated corrections on reruns, and deterministic fusion (Ticket 09 lines 9–16).
- Ticket 13 retains the existing Electron/Three.js viewer, calls for topology-bound region selection/highlighting and assertion/evidence/provenance display, and supports corrections with visible precedence (Ticket 13 lines 3, 10–15). It does not specify interactive seam/vertex dragging, edit history, undo/redo, or model training.
- `TopologyMapping` stores revision/state/element type/element IDs (`api/schemas/structured_asset.py:227–239`). `UserCorrection` stores correction ID/property/value/target/status and `user-confirmed` evidence kind (`:387–395`). Asset validation checks mapping revisions, element ID bounds, unique IDs, and correction state (`:486–519`). These types do not encode edit operation, before/after delta, author/time, sequence, inverse operation, or model-update provenance.

## Design implications for seam adjustment

Treat a drag that changes only the part partition as an editable segmentation correction: record the affected face or boundary-element identities, old and new region assignment, asset/topology revision, and an ordered, reversible edit command. Keep the original model segmentation/assertions available and record the user-confirmed correction separately. Undo/redo should replay/invert these commands without discarding the audit trail. A boundary vertex handle may be the UI affordance, but its meaning must resolve to concrete topology-bound elements; a naked screen-space coordinate is not a durable correction target.

If a drag modifies mesh positions or connectivity, it is a geometry edit, distinct from relabeling the segmentation partition. Connectivity/topology changes require revision advancement and proven remap or invalidation/orphaning of affected part, material, correction, and derived mappings. For partition-only edits on unchanged topology, retain the topology revision but recompute or explicitly mark downstream semantic/material/PBR bindings affected by changed region membership. These are design implications from the existing contract, not already implemented behavior.

## Learning and data-use boundary

Persisting/reapplying a correction for the same asset is correction storage, not evidence that an AI model learned. A learning claim would require an actual training or model-update path, explicit data-use consent/purpose, reproducible training provenance, a new immutable model/weights identity, and held-out car segmentation evaluation showing an improvement. The cited spec and tickets provide no correction-training consent/privacy policy or learning acceptance criterion. The reference workflow is required to remain locally executable and cloud is optional (§34, spec lines 282–284); this does not itself define consent for any future training use.

## Source anchors and SHA-256

Digests are for the exact files inspected in this audit:

| File | SHA-256 | Relevant lines |
|---|---|---|
| `MODLY_AMD_SEMANTIC_3D_HANDOFF/spec/FINAL_AUDITED_SPEC.md` | `8d623f502303e14c311117248ddb041af99b3414491b5d004548ec49ecd51870` | 128–140, 192–194, 224–226, 282–284, 312–316, 358–364 |
| `MODLY_AMD_SEMANTIC_3D_HANDOFF/tickets/issues/09-evidence-fusion-and-corrections.md` | `fef8e774aa0e4e170894c17d9240c55fb9c2fc8d77955c067236df02b63cba85` | 1–16 |
| `MODLY_AMD_SEMANTIC_3D_HANDOFF/tickets/issues/13-interactive-modly-workflow-polish.md` | `21093f6c1a976ac15c6bf7cdb2fbe3fa4900de630ed39601f109fcdf29eacd83` | 1–16 |
| `api/schemas/structured_asset.py` | `e58cca36194b763dbd0731b03fddb82fabbcf133bad330ed02d16416e00be81e` | 227–239, 387–395, 486–519 |

No acceptance gates or ticket statuses were changed by this audit.
