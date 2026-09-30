# Modly AMD Semantic 3D — Authoritative Handoff

This archive is the complete audited planning package for the Modly AMD-native semantic/material-aware 3D pipeline.

## Authority order

1. `spec/FINAL_AUDITED_SPEC.md` — authoritative product and engineering specification.
2. `tickets/TICKET_INDEX.md` — authoritative implementation order and dependency frontier.
3. `tickets/issues/*.md` — agent-ready tracer-bullet implementation tickets.
4. Audit logs document how the spec and ticket set were hardened; they are evidence, not competing requirements.

If a ticket conflicts with the final audited spec, the final audited spec wins. Correct the ticket before implementation rather than weakening the spec.

## Execution entry point

Start with the current dependency frontier:

- Ticket 01 — Structured Asset headless round-trip
- Ticket 02 — AMD Runtime proof

These can proceed independently/in parallel. After each ticket lands and its acceptance criteria pass, work only tickets whose blockers are satisfied. Follow `tickets/TICKET_INDEX.md` rather than simply incrementing ticket numbers.

## Required implementation discipline

- Preserve Modly as the control plane and deepen its existing extension/workflow seams.
- Prefer Torch-MIGraphX for supported dense PyTorch inference regions.
- Use PyTorch ROCm fallback only where needed and record why.
- Use HIP/native AMD implementations for irregular/custom 3D operations where appropriate.
- Keep third-party engines replaceable behind Modly capability contracts.
- Do not silently weaken acceptance criteria to make a ticket pass.
- Preserve structured provenance, confidence, topology/version relationships, reproducibility, and targeted invalidation as specified.
- The RX 7900 GRE 16 GB acceptance target is a hard POC constraint.

## Completion definition

The project is not complete because all tickets contain code. It is complete when each ticket's acceptance criteria pass and the final POC acceptance ticket verifies the end-to-end system against the canonical fixtures and hardware constraints.

## Current implementation status (2026-09-25)

Ticket 05 remains unaccepted and its frozen evaluation contract and gates remain unchanged. Per the user's latest direction, Decider 2B Vision is the only selectable and registered vision resolver. The other resolver choices are removed; the existing GPT-6 Luna implementation source is untouched. Decider remains experimental until its model-parity, warning, and frozen semantic quality gates pass. Its local provider/model, evidence, and confidence state are preserved in provenance. The earlier Qwen2-VL candidate is superseded by the user's model constraint and must not be used for Ticket 05.
