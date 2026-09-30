# Modly AMD-Native Semantic 3D Spec — Audit Log

**Audited artifact:** `MODLY_AMD_SEMANTIC_3D_SPEC_AUDITED.md`  
**Audit date:** 2026-09-24  
**Stopping rule:** Stop only after three consecutive audit loops produce no gaps, errors, contradictions, or required edits.

## Loop 1 — Architecture and domain-model audit

**Result:** Issues found; spec edited.

Material corrections:

- Separated P3-SAM from X-Part. P3-SAM is the native-3D segmentation candidate; X-Part is optional downstream part generation/decomposition and is not required for segmentation.
- Separated material-region segmentation, semantic material classification, and PBR estimation into independent capabilities.
- Corrected Material Anything's role to PBR generation/estimation/refinement rather than assuming it performs material-region segmentation or semantic material classification.
- Replaced mandatory numeric confidence with an explicit confidence-state model that can represent native/calibrated/derived/uncalibrated/unknown confidence without fabricating a score.
- Added evidence-kind semantics: observed, deterministic-derived, model-inferred, and user-confirmed.
- Added topology-revision identity and explicit geometry mappings for part/material regions.
- Added deterministic invalidation/remap behavior when topology changes.
- Added correction orphaning instead of silently attaching corrections to changed geometry.
- Added reproducible provenance requirements including adapter revision, model/weight digest, inputs, parameters, backend, device, and runtime identity.
- Corrected the viewer architecture: Modly's existing Electron/Three.js viewer remains the POC viewer; Vulkan is optional for native/offscreen work rather than a forced UI rewrite.
- Defined AMD-native operationally and clarified that PyTorch's `torch.cuda` API name is acceptable when the actual backend is ROCm.
- Removed any assumption that Torch-MIGraphX automatically provides ideal mixed-backend graph partition fallback; fallback is explicit at tested module seams.
- Added a hard 16 GB target-hardware gate for heavy adapters, especially P3-SAM.
- Added version-pinned isolated runtime environments, content-addressed resumability, extension/weight pinning, and executable-extension trust visibility.
- Added a minimal-host-change rule because current Modly contracts may require additive structured-artifact/richer-I/O support.

## Loop 2 — Specification completeness audit

**Result:** Issues found; spec edited.

Material corrections:

- Defined glTF 2.0 as the POC interchange convention: right-handed coordinates, +Y up, +Z forward, meters, and metallic-roughness PBR export semantics.
- Added explicit color-space/normal-vs-bump handling so material maps cannot silently change meaning across adapters/export.
- Defined semantic and material labels as open-vocabulary assertions with optional normalization/ontology references rather than imposing an invented universal taxonomy.
- Added coordinate/unit/PBR round-trip tests and label-normalization tests.
- Added probe-driven candidate guidance for material-region segmentation and semantic/material identity resolution.

## Loop 3 — Internal consistency and traceability

**Result:** CLEAN — no edit.

Checks included:

- all 63 user stories sequential and structurally intact;
- all 36 implementation decisions sequential and intact;
- no contradictory legacy statements regarding X-Part, Material Anything, Vulkan, ONNX, confidence, MIGraphX fallback, or host changes;
- cross-cutting concepts present across requirements/implementation/testing/acceptance;
- no acceptance-vs-out-of-scope collisions found.

## Loop 4 — Current upstream factual verification

**Result:** CLEAN — no edit.

Verified against current upstream/project documentation:

- Modly external model/process extensions and canonical headless execution surfaces;
- Modly named multi-image port limitation/design status;
- P3-SAM/X-Part roles and current light X-Part release;
- Material Anything's documented material outputs and Blender-based preprocessing/rendering;
- Torch-MIGraphX support through PyTorch `torch.compile`;
- glTF 2.0 coordinate, unit, and metallic-roughness conventions.

## Loop 5 — Implementation-readiness audit

**Result:** CLEAN — no edit.

Checks included:

- no TBD/TODO/FIXME or hidden placeholder decisions;
- host contract, domain model, testing strategy, hardware gate, POC gate, risks, and success metric all present;
- tracer sequence 0–7 complete and ordered;
- no premature hard-lock of generator, segmenter, or PBR adapter before hardware probes;
- acceptance gate includes artifact output, provenance, AMD telemetry, Torch-MIGraphX proof, 16 GB viability, and legacy Modly regression coverage;
- failure, correction, cache invalidation, and unknown/ambiguous behavior are specified.

## Stop Condition

Loops 3, 4, and 5 were consecutive clean audits and made no changes to the specification. The requested stopping condition is satisfied.
