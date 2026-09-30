# 13 — Interactive Modly workflow and correction UX

**What to build:** Expose the proven semantic-3D capabilities in Modly's existing visual workflow and Three.js-based viewer so users can inspect structure/materials/provenance, see ambiguity, correct assertions or mappings where supported, and trigger targeted re-runs without replacing the POC renderer.

**Blocked by:** 12 — RX 7900 GRE headless POC acceptance.

**Status:** ready-for-agent

- [ ] Existing Modly workflow nodes expose capability-oriented controls rather than vendor-specific implementation concepts wherever the capability contract can hide them.
- [ ] The viewer can select/highlight topology-bound part and material regions and display semantic/material/PBR assertions plus evidence/confidence/provenance.
- [ ] Unknown, ambiguous, conflicting, orphaned, and pending-remap states are visually distinguishable and never displayed as authoritative certainty.
- [ ] Users can apply supported corrections and see correction provenance/precedence without losing the underlying model assertions.
- [ ] Users can request a targeted stage re-run and see which downstream stages/artifacts will be invalidated before execution.
- [ ] Backend, stage latency, and peak VRAM telemetry are inspectable without requiring command-line log parsing.
- [ ] The implementation preserves Modly's existing Electron/Three.js viewer; Vulkan is introduced only for a separately justified native/offscreen need, not as a prerequisite for this ticket.
- [ ] Headless workflows remain fully functional without the interactive UI.

## User-directed car seam-correction requirement (2026-09-28)

The user requires a car asset in the review test and an interactive segmentation
adjustment before accepting the result. On the car mesh, the existing Three.js
viewer must let a reviewer grab a point on a part boundary and drag it along the
surface to the intended location, preview the affected face-to-region changes,
then confirm or cancel. The edit is a segmentation-membership correction; it
must not move geometry vertices unless a separate geometry-edit action is
explicitly chosen. A confirmed edit records the exact before/after membership,
affected boundary/faces, asset and topology revision, sequence, and user
provenance; it supports undo/redo and preserves the original model assertions.
Affected semantic/material/PBR bindings are recomputed or marked for review.
The workflow does not compare face indices across separately generated car
meshes or require their geometry to match the source image.

Corrections should be retained in a form that can support a future model
learning experiment. Saving or reusing a correction for one asset alone must
not be presented as model training. Any actual adapter training/update must be
explicitly opted into, produce a new pinned model identity with provenance, and
show improvement on held-out car segmentation examples before the UI claims the
model learned. The current ticket dependency on Ticket 12 remains unchanged.
The source-backed schema/UI gap review is in
`docs/orchestration/ticket13-seam-edit-design-gap-audit-2026-09-28.md`.

The currently available car test asset is the explicitly synthetic development
fixture at `api/runtime/adapters/parts/fixtures/car-development/`. It has
topology-bound per-face labels but is a simple overlapping-primitives mesh,
not a realistic automotive benchmark or generated model result. It has not yet
been rendered or loaded into the Modly viewer. The viewer interaction remains
blocked until Ticket 12 passes; this fixture is preparation only and does not
change that dependency.
