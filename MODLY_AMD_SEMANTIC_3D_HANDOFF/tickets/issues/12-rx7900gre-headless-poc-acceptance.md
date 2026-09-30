# 12 — RX 7900 GRE headless POC acceptance

**What to build:** Run the canonical semantic-3D workflow end-to-end on the RX 7900 GRE and turn the specification's proof-of-concept gate into an automated acceptance suite. The result must prove the product contracts and AMD execution path without relying on custom UI or any mandatory cloud service.

**Blocked by:** 03 — Select and integrate the reference geometry generator; 04 — Native-3D part segmentation on 16 GB AMD; 05 — Open-vocabulary part semantic identification; 06 — Material-region segmentation independent of parts; 07 — Material identity classification; 08 — PBR property recovery with synthetic validation; 09 — Evidence fusion, conflicts, and user corrections; 10 — Content-addressed resumability and targeted re-run; 11 — Structured export, validation, and compatibility report.

**Status:** ready-for-agent

- [ ] A canonical Modly headless workflow accepts at least one source observation or existing mesh and completes geometry ingestion/generation, native-3D part segmentation, useful semantic labeling with unknown support, independent material-region segmentation, material identity classification, supported PBR recovery, evidence/fusion, and structured export.
- [ ] The canonical golden-fixture suite is versioned and owned by this acceptance ticket: a simple single-material object; a multi-part single-material object; a single-part multi-material object; an object containing plastic, rubber, metal, and painted surfaces; ambiguous part-boundary and material-boundary cases; a low-confidence/unknown semantic case; an imported mesh with known part/material truth; a synthetic asset with known PBR ground truth; a topology-changing re-run fixture; and, when named observation ports are available, a multi-view fixture with known camera metadata.
- [ ] Each reference adapter's correctness/quality thresholds are declared before final acceptance runs, and the acceptance report records pass/fail against those thresholds rather than relying on visual judgment alone.
- [ ] Every accepted reference adapter completes within 16 GB VRAM or its explicitly supported low-memory/offload mode, and sequential stage execution does not accumulate enough VRAM leakage to break a later stage.
- [ ] At least one representative dense neural module executes through Torch-MIGraphX successfully; modules that remain on PyTorch ROCm have recorded compatibility/performance reasons.
- [ ] The canonical run requires no NVIDIA driver, CUDA Toolkit/runtime, NVIDIA-only binary dependency, hosted inference API, or mandatory cloud account.
- [ ] Every heavy stage reports backend, runtime/device, peak VRAM, latency, adapter/weights versions/digests, and failure/fallback status.
- [ ] Targeted re-run/caching, topology invalidation, unknown/ambiguous confidence, conflict preservation, and user-correction precedence are exercised by acceptance fixtures.
- [ ] Final GLB/glTF passes independent validation and the versioned sidecar passes schema/reference validation.
- [ ] Existing legacy Modly generation behavior covered by the project regression suite remains green.
- [ ] A single generated compatibility report identifies any reference capability still using a provisional workaround or material CPU fallback so POC success cannot hide technical debt; for each reference adapter it includes hardware/software versions, peak VRAM and host RAM, first-run and warm latency, Torch-MIGraphX acceptance/correctness tolerance, PyTorch ROCm regions and reasons, custom HIP/native work, material CPU fallbacks, output-quality comparison to a known-good path where available, and unresolved blockers.
- [ ] Every reference adapter used by the canonical acceptance workflow is pinned/trusted by explicit revision and weight identity; no mutable upstream `main` or mutable weight tag is silently used.
