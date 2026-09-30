# 11 — Structured export, validation, and compatibility report

**What to build:** Export a completed Structured Asset as independently valid GLB/glTF plus a versioned semantic sidecar, using glTF 2.0 coordinate/material conventions, and generate a machine-readable compatibility report that records adapter/runtime/backend/hardware results for the run.

**Blocked by:** 01 — Structured Asset headless round-trip; 07 — Material identity classification; 08 — PBR property recovery with synthetic validation; 09 — Evidence fusion, conflicts, and user corrections; 10 — Content-addressed resumability and targeted re-run.

**Status:** ready-for-agent

- [ ] GLB/glTF export is independently loadable/valid without the sidecar and normalizes to the declared glTF 2.0 interchange conventions: right-handed coordinates, +Y up, +Z forward, meters, valid transforms/UV references, and metallic-roughness PBR.
- [ ] Export/round-trip fixtures preserve glTF color-space interpretation, metallic/roughness values, and tangent-space normal conventions; normal and bump remain distinct representations.
- [ ] A known non-glTF source-basis fixture proves conversion is reproducible and the source/native basis plus conversion are preserved in provenance.
- [ ] The sidecar is versioned and contains structured hierarchy, topology revision/mappings, semantic/material/PBR assertions, evidence/confidence state, provenance, corrections, validation state, and intermediate artifact references needed by Modly.
- [ ] Unsupported PBR channels remain absent/unknown rather than being synthesized during export.
- [ ] Export preserves stable IDs only where deterministic correspondence supports them.
- [ ] An independent validator catches broken references, invalid geometry mappings, incompatible schema versions, and malformed export before the result is marked complete.
- [ ] The compatibility report records each heavy stage's adapter/version, selected backend, compile/fallback outcome, runtime versions, device, peak VRAM, latency, low-memory mode, and materially significant CPU fallback.
- [ ] Export and validation are deterministic for identical accepted inputs except for explicitly declared nondeterministic metadata.
