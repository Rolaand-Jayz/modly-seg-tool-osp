# Ticket 07 / Ticket 08 rights blocker disposition — 2026-09-25

## Decision

Neither branch can advance to weights/packages or fixture evaluation from the current workdrive evidence. Do not weaken thresholds or infer rights. The shortest technical lead is Ticket 07 DMS46, but its pinned Apple license requires a separate `ACKNOWLEDGEMENTS` file that is absent at the pinned revision and unavailable from the checked upstream paths. Ticket 08 SuperMat has an immutable official SD 2.1 checkpoint file identity, but not the complete immutable Diffusers component snapshot expected by SuperMat's loader; its exact linked CreativeML Open RAIL++-M terms were inaccessible in the recorded primary-source check, and project use scope has not been confirmed.

## Already-audited evidence

- Ticket 07: `api/runtime/adapters/material-identity/evidence/DMS46_LICENSE_REVIEW_2026-09-25.md` and `TICKET07_FASTPATH_DECISION_2026-09-25.md` establish the missing Apple acknowledgement/checkpoint-rights question and say not to load DMS46 until resolved.
- Ticket 08: `api/runtime/adapters/pbr/evidence/supermat-official-assets-license-refresh-2026-09-25.md` and `supermat-official-base-provenance-2026-09-25.md` establish the official checkpoint pointer but not the required Diffusers component tree, readable exact terms, or intended-use clearance. Do not convert the checkpoint or substitute a mirror.

No weights/packages were accessed; no fixture truth was read; no thresholds or ticket status were changed.

## Exact information needed to unblock

1. **Ticket 07 / DMS46:** Provide the official `ACKNOWLEDGEMENTS` file applicable to Apple source revision `a379a63e9435e32134a465eb31ecb0aefebed985`, plus an official statement that the pinned `LICENSE.txt` covers the DMS46 checkpoint. Confirm intended use (personal/noncommercial or commercial, and whether model artifacts will be distributed) so the grant can be checked against the actual project use.
2. **Ticket 08 / SuperMat:** Provide an authorized official source for the complete immutable Diffusers-format SD 2.1 component snapshot and the exact CreativeML Open RAIL++-M terms applicable to it. Confirm intended use and whether model artifacts will be distributed.

Until one path is cleared with those primary-source and scope details, both tickets remain blocked at rights/provenance; no next acquisition or evaluation prerequisite is permitted.
