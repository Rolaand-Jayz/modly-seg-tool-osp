# 01 — Structured Asset headless round-trip

**What to build:** Make Modly accept an existing mesh through its existing headless workflow/extension seam, represent it as a versioned Structured Asset, persist its source observation/geometry references and topology metadata, and round-trip it through a no-op processing stage without breaking existing Modly generation workflows. Use the smallest additive host/manifest change necessary; if the current contract already carries the required artifact reference cleanly, do not add a new host abstraction.

**Blocked by:** None — can start immediately.

**Status:** accepted after §15 revalidation (2026-09-27).

- [x] A canonical headless workflow can ingest an existing GLB/glTF mesh and create a versioned Structured Asset sidecar that references the geometry by immutable artifact identity/digest.
- [x] The Structured Asset records schema identity/version, geometry digest, topology revision, coordinate basis/handedness, units, transforms, UV/texture-space convention when known, source observation references, provenance, validation state, and stage-artifact references.
- [x] Geometry-region mappings in the schema are explicitly topology-bound and have a defined invalid/orphaned state after topology changes.
- [x] Assertion records distinguish evidence kind from confidence state and permit an explicit unknown/uncalibrated confidence state without fabricating a numeric score.
- [x] Provenance can record adapter revision, upstream/model/weights identity and digest, runtime/backend, input digests, parameters, seed, device, source observations, and stage/run identity.
- [x] Extension/capability metadata can describe Structured Asset inputs/outputs without coupling callers to a named upstream model.
- [x] Reference adapters are distinguishable from arbitrary/unpinned third-party extensions, and reference workflows pin adapter revisions and model-weight identities rather than following mutable upstream tags.
- [x] Existing single-image/image-to-mesh extension behavior remains compatible and covered by regression tests.
- [x] The implementation does not require stable named multi-image ports; single-observation and imported-mesh paths work with current Modly contracts.
- [x] A malformed schema version, missing geometry artifact, incompatible artifact type, or invalid topology mapping fails before expensive GPU work with a bounded stage-specific diagnostic.
- [x] Source-observation records preserve known camera pose/intrinsics, exposure, white balance, lighting, and capture-order metadata with provenance bound to the immutable observation artifact; absent metadata remains absent, and known plus absent metadata survive the headless no-op round-trip unchanged. Validation: `test_structured_assets.py` and `test_headless_process.py`, including a real worker subprocess over seeded legacy and enriched sidecar observations. Lighting details retain directional, chromaticity, and environment-map identity data in digest-bound additional metadata; optional absent subfields are omitted in normal serialization.

Revalidation evidence (2026-09-27): Structured Asset plus headless process suites passed 38/38; combined compatibility suite across Structured Asset, headless process, Ticket 05 semantic node, Ticket 06 material regions, Ticket 07 material identity, and Ticket 08 PBR projection passed 89 tests with 3 environment-gated skips. The headless subprocess round-trip verifies an unchanged legacy observation and a digest-bound enriched observation including capture metadata, omitted optional values, directional/chromaticity/environment-map lighting data, and capture order. Existing `ArtifactReference` instances remain accepted for source observations. No metadata is guessed or synthesized.
