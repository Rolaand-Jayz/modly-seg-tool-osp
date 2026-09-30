# Modly AMD Semantic 3D — Agent Instructions

## Authority and scope

- `MODLY_AMD_SEMANTIC_3D_HANDOFF/spec/FINAL_AUDITED_SPEC.md` is the final product and engineering authority.
- `MODLY_AMD_SEMANTIC_3D_HANDOFF/tickets/TICKET_INDEX.md` defines the ticket dependency graph. Work only tickets whose blockers have passed acceptance.
- Ticket details live in `MODLY_AMD_SEMANTIC_3D_HANDOFF/tickets/issues/`.
- Preserve Modly as the control plane and use its existing extension, workflow, headless execution, project, and viewer seams. Make only additive host changes needed by the audited contracts.
- Resolve ticket/spec conflicts in favor of `FINAL_AUDITED_SPEC.md`; record and correct the ticket conflict before implementation.

## Implementation rules

- Do not weaken acceptance criteria, substitute a simpler architecture, claim unverified completion, or add stubs, placeholders, TODO-backed behavior, or fake success paths.
- Keep geometry segmentation, part semantics, material-region segmentation, material identity, and PBR estimation as distinct replaceable capabilities.
- Keep topology mappings bound to topology revisions. Preserve evidence kind, confidence state, provenance, corrections, and intermediate artifacts explicitly.
- For AMD inference, prefer Torch-MIGraphX for dense compatible PyTorch regions; use explicit PyTorch ROCm fallback when justified; use HIP for specialized GPU operations where appropriate. Vulkan is optional and must have a concrete need.
- The RX 7900 GRE 16 GB POC constraint is a hard acceptance gate. Missing hardware or runtime evidence must be reported as a blocker, never inferred from code inspection.
- Pin adapter and weight identities. Treat downloaded extensions as executable code.
- Preserve legacy Modly generation compatibility.

## Execution and validation

- Start with ready frontier Tickets 01 and 02; continue only after their acceptance criteria pass, following the dependency graph for all later work.
- Parallelize only independent, dependency-ready tickets and keep file ownership disjoint.
- Add or run tests where required by the ticket acceptance criteria. Report exact commands, outcomes, and environment limits.
- At each phase boundary, update `docs/orchestration/progress-state.md` with completed tickets, evidence, blockers, and next dependency-ready work.
- Keep implementation status separate from acceptance status. The project is complete only when Ticket 12's RX 7900 GRE headless POC gate and Ticket 13's dependent workflow UX criteria pass.

## Baseline provenance

The Modly source baseline was retrieved from `https://github.com/lightningpixel/modly` at commit `1476fd0b1c19c9ab177c1ca3ee4d1842119e9f65` on 2026-09-24 because the initial workspace contained only the audited handoff package. Preserve the handoff files as supplied.
