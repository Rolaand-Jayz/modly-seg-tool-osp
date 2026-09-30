# 02 — AMD Runtime proof through Modly

**What to build:** Make a Modly processing run execute representative dense PyTorch inference through a shared AMD Runtime that prefers Torch-MIGraphX where it passes correctness/usefulness gates, deliberately falls back to PyTorch ROCm at explicit module seams, records backend/memory/latency telemetry, and releases GPU resources predictably between sequential heavy runs.

**Blocked by:** None — can start immediately.

**Status:** acceptance-passed (2026-09-24)

- [x] On `gfx1100`, at least one representative dense PyTorch module executes successfully through Torch-MIGraphX using the supported PyTorch integration path and matches the PyTorch ROCm reference within a declared numerical tolerance.
- [x] A deliberately unsupported or rejected candidate module runs through explicit PyTorch ROCm fallback without assuming automatic ideal graph partitioning.
- [x] The runtime records chosen backend, compile/lower outcome, fallback reason, device identity, runtime versions, stage latency, and peak VRAM for each major execution region.
- [x] CPU fallback is allowed only when required or measured useful and is visible when it materially affects latency.
- [x] Runtime compatibility is version-pinned across ROCm, PyTorch, MIGraphX, Torch-MIGraphX, native extensions, Python packages, and model weights used by the fixture.
- [x] A supported isolated/persistent adapter environment can be created without mutating Modly's global runtime assumptions.
- [x] Sequential heavy-load fixtures release enough model/runtime state that the next fixture executes without cumulative VRAM leakage causing failure.
- [x] Backend failure produces a bounded diagnostic containing stage/module, adapter revision, backend, model/weights identity where available, input artifact identity, error class, actionable compatibility information, and a detailed-log location rather than an opaque whole-workflow crash.
- [x] A failed or repaired worker exits/restarts cleanly and cannot remain alive holding stale model/GPU state.
- [x] The reference path has no NVIDIA driver, CUDA Toolkit, CUDA runtime library, or NVIDIA-only binary dependency.

Evidence: [`api/runtime/amd/evidence`](../../../api/runtime/amd/evidence) records the accepted 31-repetition FP16 MLP (3.14× warm speedup at `atol=rtol=0.005`), the rejected-candidate PyTorch ROCm fallback, three repeated MIGraphX runs, sequential memory release, package/native identities and no-NVIDIA linkage check. Full API tests pass 148/148 in the project Python 3.12 environment. The build recipe is source-pinned but not byte-for-byte reproducible because transitive AMD wheel/archive and OS repository content hashes are not fully locked; the tested OCI image digest and observed Python, OS, native-library and fixture-weight identities are recorded in `api/runtime/amd/LOCK.md` and the evidence artifacts.
