# Modly AMD-Native Semantic 3D

An AMD-first semantic 3D reconstruction and authoring pipeline built around **Modly** as the control plane.

The project extends Modly's workflow, process-extension, asset, and headless execution surfaces to turn source imagery or existing 3D assets into structured, editable assets with explicit geometry, semantic parts, material regions, material identity, PBR properties, confidence state, provenance, corrections, and GLB/glTF export.

## Goals

- Preserve Modly's existing extension/workflow architecture rather than replace it.
- Run the reference pipeline on AMD Radeon hardware without CUDA or NVIDIA-only runtime dependencies.
- Prefer Torch-MIGraphX where it is correct and useful, with PyTorch ROCm and HIP fallbacks at tested seams.
- Keep geometry, segmentation, semantics, materials, PBR data, provenance, corrections, and intermediate artifacts independently inspectable and replaceable.
- Fail visibly when quality, compatibility, provenance, or hardware acceptance criteria are not met.

## Target platform

The primary proof-of-concept target is an **AMD Radeon RX 7900 GRE (gfx1100, 16 GB VRAM)** on Linux.

## Reference workflow

1. Acquire source observations or import an existing mesh.
2. Generate geometry when no mesh is supplied.
3. Segment geometry into candidate object parts.
4. Assign semantic identities to part segments.
5. Segment surface regions by material.
6. Classify material identity independently of PBR recovery.
7. Estimate PBR properties while preserving uncertainty and provenance.
8. Fuse compatible evidence without collapsing incomparable confidence signals.
9. Validate geometry, mappings, materials, and metadata.
10. Support targeted corrections and targeted stage re-runs.
11. Export GLB/glTF plus a versioned Structured Asset sidecar.

## Engineering contract

The implementation is governed by the audited specification and dependency-ordered ticket set under `MODLY_AMD_SEMANTIC_3D_HANDOFF/`.

Key rules:

- `FINAL_AUDITED_SPEC.md` is the final architecture authority.
- Only dependency-ready tickets advance.
- Acceptance criteria are not weakened to make a stage pass.
- Model and adapter identity, inputs, parameters, backend, device, runtime, and generated artifacts must remain traceable.
- The Structured Asset contract is the durable interchange boundary between stages.

## Status

**Active development.** The project is not yet release-ready. The audited handoff defines 13 implementation tickets and an end-to-end RX 7900 GRE acceptance gate.

The public repository is being restored around the audited project authority while implementation work continues.

## Upstream

Modly: https://github.com/lightningpixel/modly

This project builds on Modly's host architecture and adds the AMD-native semantic 3D pipeline described in the audited specification.
