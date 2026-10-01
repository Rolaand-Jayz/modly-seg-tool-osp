# 04 — Native-3D part segmentation on 16 GB AMD

**What to build:** Take a canonical mesh and produce topology-bound candidate part segments through a replaceable `Segment Parts` adapter on the RX 7900 GRE. Probe P3-SAM first, but select an alternate native-3D segmenter if P3-SAM cannot satisfy the AMD, 16 GB, or quality gates. Keep X-Part separate and optional.

**Blocked by:** 01 — Structured Asset headless round-trip; 02 — AMD Runtime proof through Modly.

**Status:** open; the latest frozen-fixture quality gate fails. Runs 3 and 4 produced bitwise-identical canonical labels on the same input, allowing the one-time frozen score. Macro-IoU was **0.50** (required >=0.90), raw model coverage was **0.50**, completed coverage was 1.0, and overlap was zero; completion cannot substitute for model coverage or semantic quality. A separate registered-input diagnostic produced a complete partition on a different run but does not replace the frozen fixture gates. The current path remains unaccepted until unchanged quality, coverage, overlap, repeatability, topology, provenance, confidence, and RX 7900 GRE resource criteria pass. PartField remains held before model load pending project-use scope clarification and is not used by the selected path.

### Synthetic-car AMD development run (2026-09-28)

The procedural 656-face car completed the RX 7900 GRE workflow. Raw model
coverage was only 28/656 faces (27 with ID 10 and one with ID 34); 628 were
unassigned. Completion filled the partition, leaving 655 faces in one
canonical region and one in another. The opt-in trace ran but was partial at
its 48 KiB cap. The face-label truth file was not read and no quality score
was calculated. This single sample is development evidence only and does not
pass Ticket 04. Details, hashes, preview, and run artifacts:
[`TICKET04_CAR_DEVELOPMENT_TRACE_2026-09-28.md`](../../../api/runtime/adapters/parts/evidence/TICKET04_CAR_DEVELOPMENT_TRACE_2026-09-28.md).

### Fresh native part-segmenter source screen (2026-09-28)

SegviGen's official inference path remeshes TRELLIS.2 output instead of
providing face labels for the caller's unchanged topology, and its documented
minimum is an NVIDIA GPU with 24 GB VRAM. S²AM3D segments point clouds but
does not publish a complete face-transfer contract for Modly's caller mesh;
its setup requires CUDA-specific dependencies and has no documented AMD
route. Neither clears the topology, AMD, rights/provenance, and quality
prerequisites. No code/weights, fixture, truth, or GPU was accessed. Ticket 04
remains open and gates are unchanged. Evidence:
[`TICKET04_FRESH_NATIVE_PART_SEGMENTER_SOURCE_SCREEN_2026-09-28.md`](../../../api/runtime/adapters/parts/evidence/TICKET04_FRESH_NATIVE_PART_SEGMENTER_SOURCE_SCREEN_2026-09-28.md).

### DirectPart official-source screen (2026-09-28)

DirectPart is a new multi-view semantic part-segmentation lead with a published
car category, but its official code back-projects SAM3 masks to a separate
ShapeNet point cloud and exports point labels; it does not preserve labels on
Modly's original caller mesh or topology revision. Its benchmark is
point-cloud evidence only. The documented setup is CUDA 12.6+ with a gated
SAM3 checkpoint whose terms/identity and full DirectPart code/weight set are
not resolved; no AMD/RX 7900 GRE <=16 GB path is documented. It was rejected
before acquisition or evaluation. No code, weights, fixtures, or truth were
accessed. Full gate findings and official links:
[`TICKET04_DIRECTPART_OFFICIAL_SOURCE_SCREEN_2026-09-28.md`](../../../api/runtime/adapters/parts/evidence/TICKET04_DIRECTPART_OFFICIAL_SOURCE_SCREEN_2026-09-28.md).

The view-2 follow-up is recorded in `api/runtime/adapters/parts/evidence/TICKET04_FINITE_RETRY_VIEW2_SEQUENCE_DIAGNOSTIC_2026-09-28.md`. A refreshed per-call capture later failed on view 1: its first cached features and one permitted retry were entirely non-finite. The diagnostic records the matching locked view input and stops before view 1 proposals or any final labels. Evidence and report hash are in that same diagnostic note. This keeps Ticket 04 open; no acceptance gate changed. One isolated view-2 setup recovered after one retry; an ordered cache-only run had a successful retry on view 0 and finite first passes on views 1 and 2. Later locked diagnostics used the actual upstream seed loop, including proposals and video propagation for earlier views, then stopped at view 2 image setup or failed earlier when a retry remained non-finite. Proposal counts varied. Some runs wrote partial per-view files; the observed face array contains only sentinel `999` and provides no usable segmentation. No truth was accessed and no retry policy was qualified. Keep the candidate opt-in and all Ticket 04 gates open.

The 2026-09-28 run 3/run 4 pair used matching input, topology, adapter revision, model/dependency locks, seed, and render identities. Although raw upstream label IDs differed (29 versus 27), both raw arrays had 768 assigned faces and 768 sentinel faces; the same recorded completion policy yielded the same one-region canonical partition. The canonical label arrays are bitwise identical, so the frozen repeatability prerequisite for scoring is now met. The scorer was invoked once against the authored two-part fixture: macro-IoU **0.50** (required >=0.90), coverage **1.0**, overlap **0**. The selected path fails quality and Ticket 04 remains open. Raw 50% model coverage remains separately visible; completion does not count as model segmentation quality. Full details and identities: `api/runtime/adapters/parts/evidence/TICKET04_FROZEN_FIXTURE_SCORE_RUN3_RUN4_2026-09-28.md`.

A read-only proposal-to-label audit found 41–43 accepted/registered 2D proposals per run, but saved per-view labels show only the later six views contributing face labels, and those reduce to one class on half the faces. The pinned projection/voting path can explain how proposals fail to become distinct 3D regions, but available logs do not reveal the per-proposal survival and lifting outcomes needed to identify the exact loss point. Completion cannot recover a second class that is absent from raw labels. Evidence: `api/runtime/adapters/parts/evidence/TICKET04_RUN3_RUN4_PROPOSAL_TO_LABEL_BOUNDARY_AUDIT_2026-09-28.md`. No code or threshold changed.

The 2026-09-27 larger-mesh panel includes two successful same-code Flamingo repeats (46,180 faces), two further same-code Flamingo attempts that failed before final labels, and two 83,732-face chair attempts that failed differently before labels. The successful Flamingo outputs agree on 98.1052% of final face groupings after region matching (875/46,180 assignments differ; 10 regions each). One chair attempt had proposals only in its starting view; the other found proposals in six views but left every face unassigned and failed completion. Neither provides face labels for a repeatability comparison. Full diagnostic details and sample limits: `api/runtime/adapters/parts/evidence/TICKET04_FLAMINGO_REPEATABILITY_PANEL_2026-09-27.md`. This is additional descriptive evidence, not a gate change or a quality score.

Separate geometry generations may have different face counts and are treated as separate assets, each with mappings bound to its own topology revision. No face-index comparisons are made across the Flamingo and chair assets. The frozen exact-label repeatability check concerns rerunning segmentation with identical mesh bytes, adapter revision, weights, parameters, and seed; it does not require distinct generated meshes to share topology.

A 104,454-face teapot run produced a valid four-region complete partition on its own topology. All 48 registered artifact digests validated; peak allocated/reserved VRAM was 13.60/15.55 GiB and confidence remained `unknown`. No truth was accessed and no quality score was produced. A same-mesh repeat with the same code/settings/seed failed before labels (`GEOSAM2_NO_FACE_LABELS`). Evidence: `api/runtime/adapters/parts/evidence/TICKET04_FLAMINGO_REPEATABILITY_PANEL_2026-09-27.md` and the `teapot-run-*.json` attempt records.

The exploratory Flamingo run `8bb46be6-1d5a-46e0-a023-7ddb8449257a` finished on the RX 7900 GRE with a complete, disjoint 16-region partition of its own 46,180-face topology; all 48 registered stage-artifact digests validate. It reported 3,642,435.9 ms load/inference latency, 12,869,627,904 peak allocated bytes, and 14,824,767,488 peak reserved bytes on the RX 7900 GRE. However, the run used the earlier unbounded diagnostics, and its Structured Asset sidecar grew to 545,638,320 bytes because the 13.4 MB proposal audit was copied into every part assertion's provenance. The workflow emitted `done` but the container exited 1; the valid sidecar therefore does not qualify the process or Ticket 04. The adapter now records the full audit once as a registered stage artifact and stores only its digest/size in each assertion's provenance. A focused regression covers the compact audit pointer. The changed path still requires a target rerun and exit-code verification; no truth or quality score was used for this exploratory asset.

The repeat pair used byte-identical mesh, sidecar, correspondence, topology, and render-manifest inputs. Independent artifact validation found all registered output artifacts present and digest-matching, and both mappings cover the same 1,536 faces exactly once. A synthetic completion-artifact test now exercises the actual AST- and hash-pinned upstream `complete_labels()` implementation with `-1` and `999` sentinels, checking raw-label preservation, derived labels, fill-mask provenance, confidence state, and manifest hashes. It passes in the project Ticket 04 container; this does not substitute for exercising the branch on RX 7900 GRE. Opt-in per-view digest/count diagnostics are implemented under `MODLY_GEOSAM2_DIAGNOSTICS=1` to locate repeat divergence without storing masks, images, embeddings, or coordinate payloads. See `api/tests/test_ticket04_geosam2_completion_artifacts.py` and `api/tests/test_ticket04_geosam2_diagnostic_telemetry.py`; default behavior remains unchanged. Current diagnostics do not yet capture NMS score/box/index digests or exact IoU/stability subfilter counts.

The local project image identity was read-only verified after the project runroot recovery; image ID/digest and command are in `api/runtime/adapters/parts/evidence/ticket04-geosam2-image-identity-2026-09-25.md`.

- [x] Before final comparison, segmentation correctness/quality thresholds and the selection rubric were declared; P3-SAM was probed on the target for native dependencies, AMD execution, peak VRAM, latency, mapping quality, and required low-memory measures, and GeoSAM2 was selected after P3-SAM failed without changing the capability contract.
- [ ] The chosen adapter accepts canonical imported/generated triangle meshes and emits part-region IDs plus explicit face correspondence tied to the current topology revision, with a disjoint complete-partition overlap policy. Target execution and multi-primitive/duplicate-face correspondence tests pass. **Revalidation pending:** current registered run left face 58 unassigned.
- [ ] The known-truth fixture is scored with face-level macro-IoU after current-code output arrays are frozen: require the unchanged 0.90 gate, 1.0 coverage, and zero overlaps. The prior result was 1.0 macro-IoU, 1.0 coverage, zero overlaps, with per-part IoU `[1.0, 1.0]`, but it predates the current adapter behavior and cannot close this check.
- [x] Unsupported geometry fails with stage-specific actionable diagnostics; correspondence and invalid-topology paths are covered by focused tests.
- [x] Segment output contains evidence/provenance and explicit confidence state; absent trustworthy confidence remains `unknown`.
- [ ] Repeated current-code runs preserve IDs only when exact topology revision and face membership establish the same region, or when a proven correspondence across changed topology establishes that same region; otherwise mappings/IDs are invalidated and dependent assertions require review or recomputation. Revalidation pending after the failed coverage run.
- [x] X-Part is not required by `Segment Parts` and is not silently substituted for semantic labeling.
- [ ] The current selected path fits 16 GB: prior runs peaked at 14,417,920,000 reserved bytes (13.425 GiB) on the RX 7900 GRE; repeat current-code measurement and supported allocator recovery evidence are pending.
- [x] GeoSAM2's CUDA-origin connected-components operator is replaced by the tested CPU OpenCV 8-connected provider. No shared specialized GPU operation justified HIP extraction; the candidate uses explicit PyTorch ROCm because the measured image-encoder MIGraphX path failed the declared tensor-tolerance gate.

## Current implementation evidence

### Modly production workflow follow-up (2026-09-25)

The pinned GeoSAM2 adapter is integrated into the Modly process extension and is the declared default; P3-SAM remains selectable for compatibility. Fresh run `cccccccc-cccc-4ccc-8ccc-cccccccccccc` on the RX 7900 GRE produced two topology-bound regions, separate canonical and raw upstream label arrays, unknown confidence, provenance, a Structured Asset sidecar, and intermediate artifacts. Post-run validation confirmed exact source/truth mesh correspondence, 1.0 macro-IoU, 1.0 coverage, zero overlaps, all 45 stage artifact references, and all 38 render bundle digests. Its repeat run produced bitwise-identical canonical and raw labels. The workflow uses explicit PyTorch ROCm because the narrowly compiled MIGraphX image encoder exceeded the frozen `1e-4` parity tolerance on 2/7 outputs. Full measurements and hashes are in `api/runtime/adapters/parts/evidence/ticket04-geosam2-production-workflow-amd-result-2026-09-25.md` and `.json`.

The prior completed workflow run was followed by the fresh current-code run above, which exercises separate canonical compact labels and untouched upstream labels end to end. Both maps match the exact persisted face regions. The target process and artifact audit passed; no criteria were waived.

The adapter-local Sonata compatibility subset now implements the selected `SubMConv3d`, `SparseConvTensor.replace_feature`, `torch_scatter.segment_csr`, module-predicate, and scoped `addict.Dict`/`timm.layers.DropPath` call paths with standard PyTorch/Python operations. A hash-pinned config overlay disables Sonata flash attention and selects its pure-PyTorch branch. A source-hash-locked overlay defers only the unused SciPy import required by `ElasticDistortion`; the transform’s numeric operations remain unchanged and explicit use fails clearly when SciPy is unavailable. Focused CPU tests pass 18/18 in the pinned ROCm 7.14 / PyTorch 2.11 image with no network or accelerator devices. Evidence, source/overlay hashes, exact unsupported subset, and the latest selected-path import outcome are recorded in `api/runtime/adapters/parts/evidence/ticket04-sonata-ops-cpu-tests.md`.

The pinned P3-SAM auto-mask overlay now routes its FPS and face-adjacency calls to adapter-local NumPy implementations. The FPS implementation is source-pinned to `fpsample==1.0.2`, Git commit `4124a21dc664c3ee745e3083833d310da814453b`, including wrapper point conversion, global NumPy start-index draw, float32 coordinate-ordered distance arithmetic, strict min-distance update, and highest-index tie choice. It was compared with the compiled official extension on tie-heavy, seeded-random, and translated known-truth fixture inputs at 100,000 points/400 prompts: outputs and post-call RNG draws matched for three repetitions of each fixture. The adapter calls upstream `set_seed(seed)` before each inference pass, which seeds the same NumPy global RNG. The NumPy-only focused tests pass 5/5 for FPS and 3/3 for face adjacency. On the exact translated known-truth fixture, FPS adapter median is 62.645 ms versus 156.432 ms for the compiled native extension. Overlay and adapter code hashes and detailed commands are in `api/runtime/adapters/parts/evidence/ticket04-fpsample-numba-cpu-screen.md`. The pinned P3-SAM and Sonata checkpoints have now been loaded for bounded RX 7900 GRE runs; their measurements and limitations are recorded below. The broader host test venv lacks PyTorch, so it is not the model runtime. The target probe proves inference on the frozen fixture in low-memory mode, but does not establish the user-facing workflow integration or candidate acceptance.

## Cached alternate screen

An independent, read-only CPU source screen examined the staged X-Part subtree at the Hunyuan3D-Part checkout. X-Part is not ready for operation-level parity or quality: its own README still lists pre-trained models as unreleased; no X-Part checkpoint exists in the staged/cache trees; and the demo's `from_pretrained` path downloads a missing snapshot. Its pipeline generates new per-part meshes from bbox/point prompts instead of assigning one deterministic, topology-bound label to every original input face. Its requirements/source also contain `spconv-cu124` and CUDA-specific torch-scatter/torch-cluster references. The adapter-local FPS/face-adjacency paths now have source-parity tests, and P3-SAM has bounded target runs recorded above; these do not qualify X-Part. CUDA-named autocast/event APIs remain unqualified porting points. Exact X-Part source hashes and interface findings are in `api/runtime/adapters/parts/evidence/ticket04-xpart-alternate-screen.md`. No X-Part weights were fetched or loaded.


## RX 7900 GRE target-run results

A bounded explicit PyTorch ROCm low-memory run on the frozen 1,536-face fixture completed with full face coverage, zero overlaps, and identical warm masks, but measured face-level macro IoU `0.7412493068757922`, below the frozen `0.90` minimum. Exact timing, VRAM, per-class IoU, backend, and output hashes are recorded in `api/runtime/adapters/parts/evidence/ticket04-p3sam-amd-probe.json` and the runtime result bundle. This is a quality failure; the low-memory model output is not selected.

The one additional official upstream-default comparison used the same source, checkpoint hashes, fixture, seed, and threshold with 100,000 points / 400 prompts / batch 32. It failed at `p3sam.seg_mlp_1` with a PyTorch ROCm OOM while requesting a 6.10 GiB allocation with 1.90 GiB free; no segmentation output or quality score was produced. The attempted external one-second monitor missed the active interval, so board-VRAM peak is unavailable and the target memory criterion is not certified for this run. It was not repeated. Full inputs, diagnostic, memory caveat, and raw report digest are in `api/runtime/adapters/parts/evidence/ticket04-p3sam-upstream-default-probe.md` and `.json`.

These results do not satisfy Ticket04: low-memory quality misses its frozen threshold, upstream-default cannot complete, and no alternate has passed the equivalent topology-bound quality/runtime gates.

### COPS primary-source screen (2026-09-28)

COPS is a distinct mesh-to-part research lead, but it does not clear pre-acquisition gates. Its official benchmark is point-cloud/category driven and does not document a complete topology-revision-bound mapping for every face of an arbitrary Modly mesh. Reviewed sources do not establish clear code/weight terms or immutable COPS/model identities. The published environment uses CUDA-specific PyTorch/PyTorch3D dependencies and benchmark instructions explicitly require CUDA; no COPS ROCm/MIGraphX/HIP or RX 7900 GRE evidence is published. No code/weights were acquired or run. Full findings and primary-source references: `api/runtime/adapters/parts/evidence/TICKET04_COPS_PRIMARY_SOURCE_SCREEN_2026-09-28.md`. Existing acceptance criteria remain unchanged.


## Frozen upstream-density chunked comparison (2026-09-25)

One follow-up target comparison is predeclared as probe mode `upstream_chunked`:
100,000 points, 400 prompts, prompt batch size 4, seed 42, and threshold 0.95.
It keeps the upstream point/prompt density and changes only prompt execution
chunking to reduce activation memory. Model, weights, config, fixture, quality
threshold, and acceptance gates are unchanged. The exact preset was added to
`api/runtime/adapters/parts/probe.py`, locked in the probe runbook, and covered
by the focused preset test before target execution. It will run once; no
parameter tuning against fixture truth is authorized by this screen.

The data-producing target run completed through explicit PyTorch ROCm after
MIGraphX aborted on the recorded zero-sized dynamic tensor lowering. It
produced complete, deterministic, non-overlapping face masks and stayed under
16 GiB (7.95 GiB peak reserved), but macro IoU was 0.7182146991, below 0.90.
The preset is rejected; no quality gate changed. Exact device/runtime,
latencies, per-module qualification, raw report digest, and the compiler
failure are in `api/runtime/adapters/parts/evidence/ticket04-p3sam-upstream-chunked-probe.md` and its JSON companion.

The focused Ticket 04 API/runtime suite passes 41/41 in 1.58 seconds in the
project-owned pinned container with networking disabled and repository,
dependency, and model-source mounts read-only. Two stale overlay test
expectations were corrected to assert the current PCA metadata key and the
adapter-local FPS/face-adjacency routing. This does not alter model quality or
acceptance status.

## Independent alternate license and source screen (2026-09-25)

No download, package install, container, or model execution occurred. PartField remains the closest direct mesh/face-clustering alternate; its source is pinned to `373025dbd283bb44cc4a6dc78c99994dbc91de32`, and its Objaverse checkpoint to Hugging Face revision `f8cda8fd7dcef0596654015a482cc89407977a29`, with official SHA-256 `463efc8a3afd3913142aa025e0125c00f16ef452b8de6a132ebe32bbe7877ee4`. The official source LICENSE and checkpoint repository LICENSE both apply NVIDIA License §3.3 limiting use to non-commercial research and education. Because this project's intended use scope is not established here, do not load or probe this model until the owner confirms the permitted scope or obtains separate authorization. The published runtime also requires a CUDA 12.4 / PyTorch 2.4 path and a CUDA-specific torch-scatter wheel; AMD operation parity remains unqualified.

GeoSAM2 is a separate Apache-2.0 lead with a pinned HF checkpoint revision `ba92f5f50418f2fe9af1078448b63176df13b1ee` and fp32 checkpoint SHA-256 `2e391c0d9455fe92b69d61cdc94754d9b8081b9b541d09e1b6a3a55ebf6c6de0`. Its output is per-face labels, but it requires an interactive 2D prompt propagated over 12 rendered views; that does not yet match the frozen automatic native-mesh candidate path. The official source commit SHA was not retrievable in this offline screen, so it is not immutable-source-ready. The candidate requires an interface/prompt-policy fit review and source pin before any target run. SAMPart3D and Find3D publish MIT identifiers but specify CUDA-only custom PointOps/FlashAttention and other NVIDIA requirements; SAMPart3D additionally trains per mesh from a backbone initialization, and Find3D outputs point-cloud labels. Neither is ready for an AMD face-label target probe. PartSAM's official repository still lists inference code and pretrained model release as TODOs.

Full source, checkpoint, licensing, dependency and output-contract findings, with official primary-source links, are in `api/runtime/adapters/parts/evidence/ticket04-alternate-license-and-source-screen.md`. No alternate is currently license-clear, fully source/checkpoint-pinned, AMD-preflighted, and interface-matched for target probing; Ticket04 remains unaccepted and all frozen gates remain unchanged. PartField specifically requires project-use clarification before any weights are loaded.


### Point-SAM primary-source screen (2026-09-25)

Point-SAM is the closest newly found direct mesh-propagation lead, but it is
not ready for a target probe. Its official checkpoint is pinned and Apache-2.0
(1.244 GB, SHA-256 `bfe0aa4fee2d3c08251271e597954e6a2d26209d003c724d0ff049f578410ab2`),
and the code license is MIT. However, the exact Git source revision could not
be established; setup requires `FORCE_CUDA=1` torkit3d and NVIDIA Apex CUDA
extensions; and no automatic part-proposal contract or RX 7900 GRE route is
documented. No source or checkpoint was fetched or executed. Evidence:
`api/runtime/adapters/parts/evidence/ticket04-point-sam-screen-2026-09-25.md`.

### OneFormer3D source screen (2026-09-25)

OneFormer3D is a no-go for target probing. Its official presets target ScanNet,
ScanNet200, and S3DIS rather than arbitrary input meshes with stable
source-face mapping; code/checkpoints are not immutably pinned, weights have
no published checksum, and the repository declares CC BY-NC 4.0. The documented
SpConv/MinkowskiEngine sparse backbones have no qualified AMD operation route
or RX 7900 GRE memory evidence; published training requirements cite 24–32 GB
GPUs. No code or weights were downloaded. Evidence: `api/runtime/adapters/parts/evidence/ticket04-oneformer3d-screen-2026-09-25.md`.

## Additional disjoint native-part candidate screen

A separate read-only screen reviewed CoSMo3D and MeshCNN without fetching source/checkpoints or running models. Neither advances to target probing: CoSMo3D's documented point/face labels depend on category-specific preprocessed point-cloud inputs and a CUDA-oriented stack, without a clearly pinned checkpoint/license and general arbitrary-mesh topology mapping; MeshCNN's pretrained routes are dataset/task-specific, with no immutable checkpoint/data license or preserved original-face correspondence evidence and no AMD target qualification. PartField was not reviewed or changed by this additional screen; its non-commercial research/education license and pending project-use clarification remain as recorded in the preceding section. Exact findings are in `api/runtime/adapters/parts/evidence/ticket04-additional-native-part-model-screen-2026-09-25.md`. No gate changed and Ticket04 acceptance remains failed/open.

An additional source-only screen reviewed SimpleGeoZe/GeoZe and PointCLIP V2. SimpleGeoZe/GeoZe is category-limited and documents a CUDA 11.8/PyTorch 2.1.2 point-cloud route; the inspected sources do not establish a root code license, checkpoint license/digest, RX 7900 GRE path, or arbitrary-mesh face mapping. PointCLIP V2 has MIT code but no qualified weight identity/license, AMD route, face mapping for arbitrary meshes, or target quality evidence. Neither advances to staging or target probing. Evidence with official primary-source links: `api/runtime/adapters/parts/evidence/ticket04-read-only-open-mesh-segmentation-search-2026-09-25.md`. No acceptance criterion changed; PartField remains untouched pending project-use scope clarification.

### Final public candidate frontier screen (2026-09-25)

A further source-only screen reviewed Find3D, PartSTAD, PartSAM's current release state, and PartDistill. None qualifies for staging or target probing: their documented interfaces require interactive text/category/query masks or preprocessed point clouds rather than complete arbitrary-mesh face partitions; checkpoint identity/terms remain incomplete for the relevant releases; and the documented stacks contain CUDA-specific operators without a qualified AMD path. No code, weights, or datasets were fetched or executed. Exact official-source findings and reconsideration conditions are recorded in `api/runtime/adapters/parts/evidence/ticket04-final-native-candidate-frontier-2026-09-25.md`. This does not prove that no suitable model exists, select an alternate, or change a gate. Ticket 04 remains open after the measured P3-SAM quality failure; PartField remains held pending project-use scope clarification.

### SAMesh alternate screen (2026-09-25)

SAMesh is a relevant multi-view mesh-segmentation research lead: its method renders surface-normal and shape-diameter views, segments them with SAM2, uses rendered face IDs to connect regions across views, and lifts labels back to mesh faces. It does not qualify for staging or target probing. The inspected official README documents a CUDA 11.8-tested environment and an unpinned submodule install, but does not establish an immutable source revision, a root-project license, exact model asset identities, AMD parity, or the frozen Modly fixture quality. SAM2's optional CUDA extension can be disabled, but the official install guide says this skips its mask postprocessing; that is not ROCm or quality evidence. No code or weights were downloaded. Details and official primary-source links are in `api/runtime/adapters/parts/evidence/ticket04-samesh-alternate-screen-2026-09-25.md`. No gate changed; Ticket 04 remains open.

### Late-2026 candidate frontier screen (2026-09-25)

A fresh primary-source screen reviewed MeshSegmenter, Roblox CubePart, and SAM3D-Part. None is ready for staging or RX 7900 GRE probing. MeshSegmenter remains an empty code-release README with no source, license, or pinned artifacts. CubePart generates a new mesh per user-supplied semantic part and does not return a complete topology-bound original-face partition; its parent artifacts use a research-only RAIL-MS license, and the README does not pin checkpoint revisions/digests. SAM3D-Part requires an interactive click and generates a new posed mesh; its reference stack is CUDA 12.4 with an 80 GB H100/A100 and multiple CUDA extensions, and component licenses include Meta's SAM License. No candidate has frozen-fixture quality or AMD 16 GB evidence. No assets were downloaded or run, and no acceptance criterion changed. Detailed primary-source links and exact blockers: `api/runtime/adapters/parts/evidence/ticket04-late-2026-part-model-frontier-screen-2026-09-25.md`.

### GeoSAM2 automatic-path clarification (2026-09-25)

The upstream `inference.py` has an automatic mask-generator route when no `--mask-path` is supplied: its default run seeds view 0 and the opposite view with generated masks, propagates them, and lifts results to per-face labels. The published demo also remains interactive; default auto generation does not establish complete disjoint partition quality, deterministic part identity, source-face/topology preservation, or RX 7900 GRE compatibility. A read-only official Git remote lookup resolved code `main` to full commit `b5de23c60ab487d407b623d394a1614f9714761c`; its complete 238-file source tree and dependencies are now locked and staged in project-owned ignored caches. The source import and ROCm device API preflight pass on the RX 7900 GRE, but no checkpoint was loaded. A separate review documented a canonical inference-GLB plus fail-closed face-map design; this implementation and its multi-primitive/duplicate-face loader tests remain outstanding. Evidence: `api/runtime/adapters/parts/evidence/ticket04-geosam2-source-screen-2026-09-25.md`, `ticket04-geosam2-pinned-source-qualification-2026-09-25.md`, `ticket04-geosam2-amd-import-preflight-2026-09-25.md`, and `ticket04-geosam2-face-correspondence-strategy-2026-09-25.md`; direct primary source: `https://github.com/VAST-AI-Research/GeoSAM2/blob/b5de23c60ab487d407b623d394a1614f9714761c/inference.py`.

### GeoSAM2 canonical topology and RX 7900 GRE candidate result (2026-09-25)

The canonical inference GLB and explicit topology-revision-bound face map are implemented in `api/runtime/adapters/parts/geosam2_correspondence.py`. Four focused tests pass, including a two-primitive GLB through Modly `_input_mesh()` and duplicate-row checks against the exact pinned GeoSAM2 loader. The original fixed `opposite_views` policy was deterministic but left 768 faces at upstream sentinel `999`; it fails complete coverage and was not truth-scored. A separately frozen all-rendered-view candidate uses each of the 12 views as an automatic proposal seed with all other parameters unchanged. Two isolated RX 7900 GRE processes returned identical 1,536-face arrays; the unchanged scorer measured macro-IoU 1.0, coverage 1.0, and zero overlap with 14.875 GiB peak reserved. This clears the candidate's fixture quality/resource screen, not Ticket04 acceptance. Exact source/model/input/output digests, timings, metrics, the initial in-process OOM, and the CPU connected-components fallback are recorded in `api/runtime/adapters/parts/evidence/ticket04-geosam2-allviews-amd-result-2026-09-25.md` and `ticket04-geosam2-allviews-quality-score-2026-09-25.json`. Correspondence checks remain in `ticket04-geosam2-correspondence-implementation-2026-09-25.md`.


### Acceptance status reconciliation (2026-09-25)

This issue's earlier candidate-frontier notes at lines 105, 111 and 115 describe the state before GeoSAM2 completed the selected all-view policy and fresh Modly process-extension proof. They are historical, not the current status. The acceptance checklist at the top is supported by `ticket04-geosam2-production-workflow-amd-result-2026-09-25.md` and `.json`; the runner script pins `localhost/modly-amd-geosam2:ticket04`, whose local image ID and manifest are recorded in `ticket04-geosam2-image-identity-2026-09-25.md`. The production report explicitly says it did not capture the child image digest during that exact run; the separate image inspection verifies the referenced local tag identity after runroot recovery, not an in-run image attestation. Preserve that evidence limitation. The successful selected-path run, label correspondence and scorer are the acceptance basis; the obsolete open/failed phrasing in the earlier screen notes does not override the current checklist/status.

### Empty automatic seed-view handling (2026-09-26)

A later Ticket 05 development batch exposed an upstream edge case: `show_anns([])` returns `[]`, while the pinned propagation guard checks only `sorted_anns is not None`, so a view with no proposals attempts propagation without any predictor objects. Modly now applies the AST-checked, hash-pinned `skip-empty-auto-proposal-seed-v1` policy at the adapter boundary, records generated/accepted proposal counts per seed view in a digest-bound audit artifact, and skips propagation only for empty automatic-proposal views. Source-authored target masks are explicitly rejected at GeoSAM2's prompt-input boundary and remain downstream semantic-target inputs. All-empty runs, missing labels, unassigned faces and incomplete partitions continue to fail; no acceptance threshold changed. The original synthetic policy tests and evidence are in `api/runtime/adapters/parts/evidence/TICKET04_GEOSAM2_EMPTY_PROPOSAL_POLICY_2026-09-26.md`. At the time this paragraph was written, the corrected path had not been rerun on AMD. A later run did exercise it and failed on sentinel coverage, prompting the separate completion policy and the 2026-09-26 repeatability failure recorded at the top of this issue; current acceptance remains open.

### Larger-mesh repeatability diagnostic (2026-09-27)

Four current-code RX 7900 GRE attempts used the same 46,180-face Flamingo mesh, face map, render manifest, model/weight/runtime locks, settings, and seed 42. Two completed outputs (`b6f57f83-6dd1-4d6d-bfb2-14e0d8033cad`, `5aa83b9a-f86a-437e-8a11-344e9f8f4b22`) each produced a valid 10-region partition with complete coverage and all 48 stage-artifact digests verified. Their raw upstream groupings match on 28,628 of the 28,631 faces assigned in both runs (99.9895%, ARI 0.999875); 17,549 upstream sentinel faces were excluded from that raw comparison. After unassigned completion, permutation-invariant matching gives 98.1052% agreement (45,305/46,180; ARI 0.937262), so 875 final face assignments differ. Fill masks differ at 10 faces. All 12 accelerated-mask and sampled-coordinate digests match; accepted proposal counts differ by one in views 0 and 6.

The two further same-code attempts (`fa73c52c-eceb-4411-a289-4b1a87088da5`, `c1a37e09-84bc-46ee-9c4d-f0ddae5ba303`) returned `GEOSAM2_NO_FACE_LABELS`: each had proposals only in view 0 (32 and 35 accepted, respectively) and none in views 1–11. Neither produced final labels or a manifest. For the first, accelerated-mask digests match successful run `b6f57f83` only in views 0–2 and sampled-coordinate digests only in views 0–1. For the second, mask digests match in views 0–2 and coordinate signatures in views 0–1; its view-0 mask and coordinates match the successful baseline although the generator returned 37 masks. Treat both as failed repeat outcomes; do not include them in face-agreement or quality scoring. An earlier completed Flamingo run (`f8c6a77e-7090-4713-bb7d-3ddcedadeb1b`) uses a different adapter-code digest, so its 86.9251% matched agreement with `b6f57f83` is exploratory, not part of the same-code pair.

The two successful runs took about 10.4 and 11.1 minutes and used 14,790,649,344 bytes peak allocated and 16,418,603,008 / 16,651,386,880 bytes peak reserved on the 16 GiB card. They each emitted a normal `done` result and persisted valid registered artifacts, but the wrapper returned status 1; the cause is not present in the run directories. No truth was accessed. Two completed outputs and two same-code failures on one mesh do not estimate typical variation across assets. The frozen exact-label gate remains unchanged and fails; no quality score is implied. Full inputs, hashes, metrics, and limits are in `api/runtime/adapters/parts/evidence/TICKET04_FLAMINGO_REPEATABILITY_PANEL_2026-09-27.md` and the ignored `.modly-amd-runtime/results/ticket04-variation-panel/` artifacts.

The completion failure report now includes the raw count of `-1`/`999` faces and the chained exception type/message capped at 512 characters. Its focused regression test passed:

`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=api .modly-amd-runtime/api-test-venv/bin/python -m unittest api.tests.test_ticket04_geosam2_adapter.GeoSAM2AdapterLabelTests.test_completion_failure_reports_raw_sentinels_and_chained_cause -v`

This diagnostic change does not alter the pinned completion routine, label behavior, failure code, or acceptance gate.

### Raw face-label count diagnostics (2026-09-27)

When `MODLY_GEOSAM2_DIAGNOSTICS=1`, the proposal audit now records total, assigned, distinct, and `-1`/`999` sentinel face-label counts if the pinned inference returns labels. If inference returns no face labels, the audit instead records `face_labels_unavailable` / `not_returned` and lists accepted non-prompt seed views. The audit is persisted before completion, so a completion failure retains any raw label summary; no face IDs or label arrays are stored. Focused diagnostic telemetry/audit tests pass 6/6 with:

`TMPDIR="$PWD/.modly-amd-runtime/tmp" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=api .modly-amd-runtime/api-test-venv/bin/python -m unittest api.tests.test_ticket04_geosam2_diagnostic_telemetry api.tests.test_ticket04_geosam2_diagnostic_audit_return -v` (5 tests, pass).

This adds only payload-free output-boundary diagnostics. The pinned upstream source is available in the project-owned runtime source cache and is read-only; its control flow shows why the teapot attempt returned no labels, but the run does not retain per-object post-filter survivors or projected point-to-face samples. The audit now distinguishes the no-label outcome and reports non-prompt seed views that had accepted proposals, without claiming mask-to-face coverage. No segmentation behavior or acceptance criterion changed.

The next same-mesh teapot attempt (`0b670b96-9416-4fcb-979b-5b299f2ec448`) exercised the digest/count diagnostics on the RX 7900 GRE. Modly's proposal audit recorded 54 accepted proposals in view 0 (55 returned by the wrapped generator), none in views 1–11; the upstream path returned no face-label field, so no completion/final-label comparison exists. In the pinned source, automatic proposals on starting view 0 seed propagation but that prompt-seed view is skipped by `save_this_result`; non-prompt views 1–11 were empty and the frozen empty-seed policy skipped them. This explains why no seed reached lifting for this attempt. Its saved audit predates the explicit `face_labels_unavailable` and non-prompt-view fields added immediately afterward. The attempt's files remain under `.modly-amd-runtime/results/ticket04-variation-panel/teapot-repeat-03/`.

### Automatic prompt-seed lift policy candidate (2026-09-27)

The pinned source intentionally skips final face lifting on the automatic prompt-seed view. A separate, unselected candidate policy now AST-matches that exact guard and allows the automatic seed through the first lift only when `opposite_auto_segmentation=True`, `is_prompt_seed=True`, and no source-authored prompt masks are present. The existing production behavior remains the default; `MODLY_GEOSAM2_PROMPT_SEED_LIFT=1` explicitly enables this candidate through the project Modly workflow runner. It leaves upstream files and prompt-mask behavior unchanged, verifies its own versioned lock before inference, records policy identity in provenance, and fails closed on source drift. Its versioned identity is in `api/runtime/adapters/parts/GEOSAM2_PROMPT_SEED_LIFT_POLICY_LOCK.v1.json`; implementation is `api/runtime/adapters/parts/geosam2_prompt_seed_lift_policy.py`.

The candidate's synthetic and adapter-lock tests pass 6/6 and verify the composed empty-seed and prompt-seed guards, prompt-mask exclusion, unchanged existing branches, AST drift rejection, and module/lock digests. The composed transform also matches and compiles the actual pinned runtime inference function. One opt-in target workflow (`0340842b-df11-4f94-b43b-9a85902fa92c`) ran on the RX 7900 GRE with candidate module digest `aeae679f2e15857e6e64b1cbfc6e0f0e05cbfc9e4308181857068e6a2e31275b` and lock digest `c60bb1b1c358205f0c31ba90ccb67c0b9cf00bc0561168e6bac87cda5e957df9`. It registered 11 automatic proposals from view 0; views 1–11 had none. The upstream result contained 104,454 labels, all sentinel `999` (zero assigned faces), and Modly failed closed with `GEOSAM2_UNASSIGNED_COMPLETION_FAILED` because there was no region label to extend (`ValueError: max() iterable argument is empty`). No final manifest or Structured Asset was produced; no face-label quality, same-input repeatability, or VRAM peak gate passed. The audit is retained at `.modly-amd-runtime/results/ticket04-variation-panel/teapot-prompt-lift-candidate-01/StructuredAssets/runs/0340842b-df11-4f94-b43b-9a85902fa92c/geosam2-inference/proposal-audit.json`. The candidate remains unselected; all acceptance thresholds remain unchanged.

### Mask-stage failure localization and second teapot run (2026-09-27)

The second opt-in prompt-seed-lift workflow (`a4921308-3de2-4337-861b-0a8948a64c73`) used the same 104,454-face teapot and frozen candidate settings on the RX 7900 GRE. It again failed with `GEOSAM2_UNASSIGNED_COMPLETION_FAILED`, produced zero assigned labels (all 104,454 were sentinel `999`), and created no final manifest or Structured Asset. This run's 12-view mask-stage audit contains 108 propagated mask/frame records: every record had zero positive foreground pixels before and after propagation, shrink, stability, area, and boolean filtering. The float-logit summaries were nonzero-valued but had no positive foreground; the boolean masks passed to lifting were all false. The audit identifies no filtering stage as the cause; the failure is upstream of 3D lifting. Accepted proposal counts changed from 11 in the earlier run to 9 here, so these are not interchangeable repeat outcomes. Neither run produced a quality score or peak-memory measurement.

The opt-in diagnostics store only stage counts and digests, not masks, images, logits, or point payloads. Their collector/lock implementation and focused prompt-policy suite pass 21/21 in the project Python 3.12 test venv. The complete completion-artifact module passes 2/2 in the network-disabled project AMD image (CPU-only invocation), including the actual hash-pinned upstream completion path. The all-unassigned completion guard now rejects this state before invoking GeoSAM2's expensive pinned completion routine; its regression test passes. That guard only avoids wasted completion work and does not repair segmentation. The run audit is at `.modly-amd-runtime/results/ticket04-variation-panel/teapot-prompt-lift-diagnostic-01/StructuredAssets/runs/a4921308-3de2-4337-861b-0a8948a64c73/geosam2-inference/proposal-audit.json`; code and lock identities are recorded in `docs/orchestration/progress-state.md`.

### Prompt registration and propagated-logit audit (2026-09-27)

A third opt-in run (`9f76dc6b-0645-4d9c-9a88-92bd854e1173`) used the same 104,454-face teapot, seed, model and candidate settings on the RX 7900 GRE with prompt-registration diagnostics enabled. All 12 automatically proposed prompts were registered on view 0 (one positive point for each distinct object ID); views 1–11 had no proposals. The audit records 156 propagated-logit summaries across the 12 frames and registered objects. Every summary was finite with minimum and maximum `-1024.0` and zero positive pixels. The output retained 104,454 raw labels, all sentinel `999`; Modly failed closed with `GEOSAM2_UNASSIGNED_COMPLETION_FAILED` and no final manifest or Structured Asset. The summaries locate absent foreground at the propagated tracker output, before the 3D lift; they do not establish why the tracker returned empty logits or whether the proposed point locations were semantically correct. No face-quality score or usable peak-memory acceptance measurement was produced. Audit: `.modly-amd-runtime/results/ticket04-variation-panel/teapot-prompt-lift-diagnostic-02/StructuredAssets/runs/9f76dc6b-0645-4d9c-9a88-92bd854e1173/geosam2-inference/proposal-audit.json`.

While enabling this audit, the completion policy was advanced to immutable lock v2 because the all-unassigned fail-fast behavior changed its module digest. A schema check still expected v1 and stopped one attempted rerun before model load; the expected schema is corrected, v1 is retained, and v2 plus the diagnostics lock are included in package data. A regression test now invokes the adapter's real v2 lock verifier. The project GeoSAM2 image passes 17/17 completion, prompt-policy and diagnostic-audit tests; the focused diagnostic/seed-lift suite separately passes 31/31 in the project Python 3.12 environment. These checks validate support code, not segmentation acceptance.

Different generated meshes, including meshes with different face counts, remain independent assets with topology-revision-bound mappings. Ticket 04's repeat check applies only to reruns with byte-identical mesh and inference inputs; this diagnostic does not compare face indices across separately generated assets.

### Ticket 04 follow-up run and video-index regressions (2026-09-27)

The opt-in diagnostic workflow `f072ca48-09d9-4b7b-91c0-9f5979ae8b2a` completed on the RX 7900 GRE using the 104,454-face teapot asset and the prompt-seed-lift diagnostic path. It emitted a valid Structured Asset and one topology-bound region; all 48 registered stage-artifact references resolve and match their SHA-256 digests. The raw GeoSAM2 output assigned 8,081 faces and left 96,373 as sentinel `999`; Modly's separate completion step filled those faces, so this is a complete artifact but not evidence of semantic segmentation quality. Confidence remains `unknown`; no known-truth score was produced. Peak allocated/reserved memory was 13,958,460,928 / 16,271,802,368 bytes on the reported 17,163,091,968-byte RX 7900 GRE, and model-load plus inference time was 727,954.1 ms. This is one run, not a run-to-run estimate or acceptance result. Exact artifacts are under `.modly-amd-runtime/results/ticket04-variation-panel/teapot-prompt-lift-diagnostic-02/StructuredAssets/runs/f072ca48-09d9-4b7b-91c0-9f5979ae8b2a/`.

The video-index correction now derives an omitted start frame from both temporary and already-committed conditioned prompts, preserves the pinned upstream `torch.inference_mode()` decorator when compiling its source-locked method, preflights before moving the seed to its temporal slot, and fails closed when no seed exists. The combined video-index, diagnostic-audit, and prompt-registration regression suite passes 28/28 both in the project Python 3.12 environment and inside the project AMD image. Tests exercise point/mask prompt state, earliest conditioned seed selection, preflight-before-remap, forward/reverse slot behavior, and no-seed failure. Same-mesh reruns remain the basis for repeat checks. Separately generated models may have different face counts; each asset is evaluated on its own topology revision, and no cross-asset face-index comparison is made. Ticket 04 remains open because its unchanged truth-quality, same-input repeat/identity, strict raw/derived coverage policy, and supported memory recovery criteria have not all passed.

An additional diagnostic workflow (`6d4a24ab-e1ab-47fa-9301-89f6c493b6c8`) completed on the same 104,454-face teapot and emitted a valid Structured Asset. Its 48 artifact references all resolve and digest-match. An independent audit verified byte-identical geometry, face map, render manifest, all 39 rendered files, model/dependency/source locks, weights, seed, and listed inference settings against `f072...`. The raw GeoSAM2 output assigned 9,892 faces and left 94,562 sentinels, compared with 8,081 and 96,373 in `f072...`; the raw assigned-face masks overlap on only 348 faces (union 17,625; IoU 0.0197447). Each completed result nevertheless collapses to one region covering all faces, so their canonical arrays are byte-identical all-zero labels. This does not show the upstream segmentation was repeatable. It is not a controlled repeat pair because the recorded adapter-code digests differ (`8ba80e…` versus `ffc5b3…`); no random-variation conclusion can be attributed to this pair. No truth score was produced; confidence is `unknown`. The second run peaked at 14,624,537,088 allocated and 16,670,261,248 reserved bytes against the reported 17,163,091,968-byte total, and load plus inference took 609,426 ms. The workflow emitted a `done` result but the launcher exited 1; the artifact audit is valid, but the launcher discrepancy remains unresolved. This diagnostic is not acceptance evidence for quality or same-code repeatability.

A new repeat (`b190eb45-f871-4b4a-93bf-6d49841edc19`) used the same teapot mesh, topology, 39 render files, model, seed, thresholds, source/video-index policy, and current on-disk adapter code digest `sha256:ffc5b3988d32ecf2227f85e1e08a910a918774656ada812e5273cbe126f257e2` recorded by `6d4a...`; its audit independently matches the upstream source and all three policy module/lock identities. It registered prompts on all 12 views, but no final face survived the upstream stability filter: the raw output was all sentinel `999` (104,454/104,454 faces unassigned) and strict completion failed with `GEOSAM2_UNASSIGNED_COMPLETION_FAILED`. No final sidecar or labels were emitted. The successful `6d4a...` run assigned 9,892 faces and completed one region. This is a same-input, same-code success-versus-failure repeat outcome. The failed run did not persist a final manifest containing its full adapter digest or numeric parameter record, so its run command and matching source/policy audit fields are the available identity evidence. This remains one paired observation, not a broad variation estimate or known-truth quality result. No truth was read. Audit: `.modly-amd-runtime/results/ticket04-variation-panel/teapot-prompt-lift-diagnostic-02/StructuredAssets/runs/b190eb45-f871-4b4a-93bf-6d49841edc19/geosam2-inference/proposal-audit.json`.

The next same-code repeat (`439c5d8f-c871-48b5-8a11-9e27f2cdf0a3`) also failed with all 104,454 raw labels at sentinel `999` and `GEOSAM2_UNASSIGNED_COMPLETION_FAILED`; only its starting view had accepted proposals. Its two input files and all 39 rendered files are byte-identical to the `6d4a...` and `b190...` runs, and its audit matches the same pinned upstream and policy identities. Within these three current-code attempts on one exact mesh, one produced a low-information one-region completion and two failed before final labels, with distinct proposal patterns. This is useful empirical evidence of unreliability on the teapot, not a quality score, multi-shape estimate, or change to the fixed acceptance gates. No truth was read. Audit: `.modly-amd-runtime/results/ticket04-variation-panel/teapot-prompt-lift-diagnostic-02/StructuredAssets/runs/439c5d8f-c871-48b5-8a11-9e27f2cdf0a3/geosam2-inference/proposal-audit.json`.

A fourth attempt (`a7248a91-2363-442a-bde2-ae03ea546e08`) on the same current code and fixed teapot inputs also failed with 104,454 sentinel labels and `GEOSAM2_UNASSIGNED_COMPLETION_FAILED`. Its two input and 39 render-file hashes match the successful/failed runs above; only the starting view registered proposals. This brings the current-code single-mesh panel to four attempts: one completed one-region result and three no-label failures. This is descriptive evidence about this mesh and opt-in prompt-seed path only. No truth was read. Audit: `.modly-amd-runtime/results/ticket04-variation-panel/teapot-prompt-lift-diagnostic-02/StructuredAssets/runs/a7248a91-2363-442a-bde2-ae03ea546e08/geosam2-inference/proposal-audit.json`.

The fifth attempt (`c0489b49-1a35-4bf0-9bc8-ded606fd6b69`) again failed with all 104,454 raw labels at `999` and `GEOSAM2_UNASSIGNED_COMPLETION_FAILED`; only the starting view had accepted proposals. Independent hashing confirmed that all five runs share identical input and render bundles, and the current `geosam2.py` digest is `ffc5b3988d32ecf2227f85e1e08a910a918774656ada812e5273cbe126f257e2`, matching the successful `6d4a...` run's persisted identity. In this five-attempt, one-mesh exploratory panel, one run completed a one-region asset by filling 94,562 faces and four failed before a final partition. Proposal patterns varied between runs. This is a small empirical failure-rate sample for one opt-in candidate path, not a quality score, general expected rate, acceptance pass, or comparison across differently generated meshes. No truth was read. Audit: `.modly-amd-runtime/results/ticket04-variation-panel/teapot-prompt-lift-diagnostic-02/StructuredAssets/runs/c0489b49-1a35-4bf0-9bc8-ded606fd6b69/geosam2-inference/proposal-audit.json`.

### Diagnostic telemetry bounds (2026-09-27)

A read-only review found that the earlier opt-in telemetry synchronously copied whole registration-mask batches from GPU to CPU, retained uncapped records, and rewrote the growing audit repeatedly. That instrumentation version is not suitable for large diagnostic runs. The collectors and persistence path have now been bounded: prompt and propagation record caps include omitted counts; propagated tensor summaries perform device-side scalar reductions; preflight is limited by object, frame, and total-output counts; mask-stage captures have bounded frame/mask totals; generator telemetry is limited to 12 views and 64 candidate/batch records per view; audit writes are batched; wrappers forward upstream generator close. Telemetry remains opt-in and inference outputs, thresholds, labels, and completion behavior are unchanged.

The project Python focused suite passed 36 tests with 1 skipped because PyTorch is absent from that CPU test environment. It verified the three diagnostic locks and CPU/stand-in paths; no GPU execution tested the torch-side reductions. Exact command:

```sh
TMPDIR="$PWD/.modly-amd-runtime/tmp" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=api \
  .modly-amd-runtime/api-test-venv/bin/python -m unittest \
  api.tests.test_ticket04_geosam2_prompt_registration_diagnostics \
  api.tests.test_ticket04_geosam2_mask_stage_diagnostics \
  api.tests.test_ticket04_geosam2_diagnostic_telemetry \
  api.tests.test_ticket04_geosam2_diagnostic_audit_return -v
```

Ticket 04 acceptance remains open. The prior repeat on the target that is still running started before this bounded version was written; its emitted diagnostics and timing are exploratory. Do not use that run to qualify the revised telemetry or claim acceptance.

### Compact-audit prompt-seed diagnostic result (2026-09-28)

The fresh RX 7900 GRE Flamingo run `f2c25070-3ee3-42a5-9bf8-c98d72fd9401` used the opt-in, unselected prompt-seed-lift path with diagnostics. It terminally failed closed with `GEOSAM2_UNASSIGNED_COMPLETION_FAILED` (`raw_unassigned_faces=46180`) and command exit 1. The audit records 34 accepted seed-view proposals, but all 34 registered predictor outputs returned mask logits entirely at `-1024`; across 192 propagated-logit summaries, all 427,819,008 values were `-1024`, with zero positive pixels. All 46,180 canonical face labels therefore remained sentinel `999`. Views 1–11 had no registered proposals. This is evidence against the opt-in prompt-seed candidate, not a failure attribution to the selected default route. No final Structured Asset or sidecar was written, so compact assertion-provenance serialization and successful launcher exit remain unverified. The bounded diagnostic audit is 2,190,083 bytes; full evidence: `.modly-amd-runtime/results/ticket04-variation-panel/flamingo/StructuredAssets/runs/f2c25070-3ee3-42a5-9bf8-c98d72fd9401/geosam2-inference/proposal-audit.json`. No truth was accessed or quality scored. Ticket 04 remains acceptance-open; gates unchanged.

The current-code default-path Flamingo diagnostic `c3d25070-3ee3-42a5-9bf8-c98d72fd9401` also terminally failed with exit 1: `GEOSAM2_NO_FACE_LABELS`. It used the same 46,180-face input mesh SHA-256 `e099909f…33aa99` and exact render-manifest digest `b2845065…1e7ab` as the prior completed Flamingo runs, seed 42, and no prompt-seed-lift option. The audit reports 28 accepted starting-view proposals and none in views 1–11; 12 per-view diagnostic summaries were recorded, but no upstream label file, final Structured Asset, or sidecar was produced. This code revision includes the bounded diagnostic/provenance changes, so it is not a same-code repeat against prior runs. It does show that this selected default path can still fail on the unchanged input. No truth was accessed. This is a failure diagnostic, not a quality score; all Ticket 04 acceptance gates remain open.

### Prompted-frame empty-mask root-cause audit (2026-09-28)

A read-only audit of both `f2c25070…` (opt-in prompt-seed lift) and `c3d25070…` (default path) locates the earliest confirmed failure at the first prompted SAM2 frame, not in face lifting: for each registered proposal, temporary `object_score_logits` are non-finite (`torch.bfloat16`) and every `pred_masks` value is `-1024` (`NO_OBJ_SCORE`). All propagated frames consequently have zero positive pixels, leaving no face labels. Both paths show the same failure, so prompt-seed lift is not the cause. The pinned upstream `inference.py` enables BF16 autocast under the CUDA PyTorch API on ROCm; this is a diagnostic lead only, not a proven cause. Do not disable autocast or change precision without a pinned target comparison and resource measurement. The next step is bounded per-layer finite/range telemetry from image features through mask decoding and object-score gating. No thresholds, masks, or truth were changed/accessed.

### Bounded numeric trace of prompted inference (2026-09-28)

The prompt-registration diagnostic is now v4 and records payload-free tensor summaries at the first image-feature output, raw mask-decoder output, and post-object-score-gating output. Each trace is capped, reports shape/dtype/finite counts/ranges/sentinel counts and first nonfinite stage, stores no tensor/image/prompt payload, restores wrapped methods after tracing, and leaves inference behavior unchanged. The exact source/lock identity is verified fail-closed. Focused project CPU tests pass 26 tests (one optional skip), including tamper rejection, numeric-only records, bounds, and instrumentation restoration. No GPU or truth was used for this code change. Run `d4e25070-3ee3-42a5-9bf8-c98d72fd9401` is now using the instrumented default path on the same Flamingo mesh to find the first nonfinite layer; no result is yet available.

Initial target trace from `d4e25070-3ee3-42a5-9bf8-c98d72fd9401` confirms `image_features` is already non-finite before mask decoding: across the first 16 traces, `output[1].backbone_fpn[1]` is partially non-finite (655,872 finite of 1,048,576 bfloat16 values) and `output[1].backbone_fpn[2]` is wholly non-finite (0/1,048,576 float32 values). Corresponding flattened feature tensors are likewise non-finite. This places the first observed fault inside or before image-feature construction and ahead of mask decoding/object-score gating. The trace caps at 16 of 64 prompt calls; zero prompt or image payload is recorded. It does not identify the internal encoder operation or prove BF16/ROCm causality. Run remains active while collecting terminal output.

The `d4e...` run subsequently completed its diagnostic execution and produced a manifest, but it does not qualify the expanded per-branch trace: the process imported the earlier coarse v4 instrumentation before that patch was written. It assigned 44,661/46,180 raw faces and retained sentinel `999` for 1,519; completion produced 15 labels, but this Flamingo has no truth labels and no quality claim is available. All five segmentation artifact references resolve and match their recorded sizes and SHA-256 digests. The proposal audit is 11,649,051 bytes and records no tensor/image/prompt payload. RX 7900 GRE recorded 13,141,639,680 bytes peak allocated and 16,145,973,248 bytes peak reserved out of 17,163,091,968 total; load plus inference was 787,313.21 ms. This does not pass the supported-memory/recovery gate. The workflow exited 1 and emitted no final sidecar, so compact assertion-provenance serialization and successful workflow completion remain unverified. A run using the expanded trace has been launched as `e5e25070-3ee3-42a5-9bf8-c98d72fd9401`; its result is pending. Acceptance criteria are unchanged.

### Frame-0 prewarm trace correction (2026-09-28)

Review of the pinned `sam2_video_predictor_geosam2.py` showed that `init_state()` calls `_get_image_feature(..., frame_idx=0, ...)` to warm and cache frame-0 features before the first prompt. The earlier expanded diagnostics only activated a numeric trace inside `_run_single_frame_inference`; therefore a later prompt read the cache and skipped `forward_image`, `image_encoder`, `pos_map_encoder`, and `feature_fusion`. The active `e5e25070-3ee3-42a5-9bf8-c98d72fd9401` trace with only `image_features`, decoder, and gating stages cannot attribute which encoder/fusion branch first became non-finite.

The v4 collector now starts a bounded trace around the existing `_get_image_feature` cache-miss call, including the actual frame-0 prewarm. It observes the forward already required by initialization and does not recompute or replace the cached feature. The module and lock hashes were updated and remain fail-closed; the schema shape did not change. A CPU stand-in test models the prewarm miss followed by the later cache hit and verifies branch stage order and no duplicate forward. Focused Ticket04 diagnostic and audit-return tests pass 29 tests with one PyTorch-only skip; Python compilation and the lock verifier pass. No GPU run has used this correction yet. Any branch-level cause remains unproven until a fresh target run emits and validates these stages; no inference precision, acceptance gate, truth, or output was changed.

### Exact-input recovery search (2026-09-28)

A bounded read-only search initially omitted
`.modly-amd-runtime/results/ticket04-variation-panel`, so its negative result
was incomplete. The exact Flamingo bundle is now recovered and identity-verified
under
`.modly-amd-runtime/results/ticket04-variation-panel/flamingo/StructuredAssets/runs/f8c6a77e-7090-4713-bb7d-3ddcedadeb1b/`.
Its render-manifest SHA-256 is
`b28450652e331bfa23bcde92479c1942a83755e58ed745faf1b091165cb1e7ab` and
face-map SHA-256 is
`4f48bf28fc677b62326fcd79e9c7a35943c5511edeafd7231c7f01b6c95d3f44`, matching
the requested identity. The rendered-mesh SHA-256 is
`e099909ff028dfeffa77a842635c06da66de85b82f92741fe2ce5ca9f533aa99`. The
schema-v3 scalar report still states `payloads_persisted:false`; that flag
describes its own report, while the bundle is retained at the path above.
Recovery and the earlier search-scope correction are recorded in
`api/runtime/adapters/parts/evidence/TICKET04_EXACT_INPUT_RECOVERY_SEARCH_2026-09-28.md`.
Ticket 04 acceptance criteria remain unchanged.

### Position-map encoder child-output trace (2026-09-28)

The latest RX 7900 GRE frame-0 trace confirms finite RGB, position-map, and normal-map inputs and a finite whole image-encoder result. The first bad module boundary is `predictor.pos_map_encoder_output`: `vision_features` and all three FPN outputs are non-finite, while its positional encoding is partly finite. The loaded image and position-map encoders each have 402 finite parameter/buffer tensors; feature fusion has 6 finite tensors. This points to the position-map encoder as the first observed failing module, but does not identify its internal layer or operation.

The opt-in v4 collector now temporarily wraps named child modules under `pos_map_encoder` during the actual frame-0 prewarm. It records only deterministic module paths and bounded numeric summaries, capped at 512 hooked children, 512 output records per run, and 8 tensor summaries per child output; truncated hook inventory and omitted outputs are explicit. Inspection failures are marked `failed` with type-only errors and are not treated as finite. Existing whole-module summaries remain. Wrappers are restored on cleanup and inference precision/model/output behavior is unchanged. CPU stand-ins verify child output order, first non-finite child localization, record caps, failed-closed summaries, and exact forward restoration. Focused tests pass 31 with one PyTorch-only skip; Python compilation and the lock verifier pass. No GPU run has loaded this child-output patch, and the active `f6` process cannot report it. No child-layer or precision cause is established pending a fresh RX 7900 GRE run after `f6` is terminal; acceptance criteria remain unchanged.

Read-only review of the expanded trace during `e5e...` found an instrumentation-timing gap: the pinned predictor precomputes/caches the first-frame feature bundle during inference-state creation, before `_run_with_empty_proposal_policy` installs these diagnostic hooks. The current prompt-time `image_features` record therefore reads a cached bundle; the image encoder, position-map encoder, and fusion hooks do not execute while a trace is active. The target audit confirms only the three coarse stages appear, although the loaded parameter/buffer scan completes and finds finite state tensors. This run cannot locate activation NaNs within those modules. The next diagnostic must observe the original prewarm forward, without altering inference outputs; no layer or precision cause is established.

Target run `e5e25070-3ee3-42a5-9bf8-c98d72fd9401` is terminal with exit 1 and `GEOSAM2_NO_FACE_LABELS`. It recorded 34 registrations and 192 propagated-logit summaries; all 427,819,008 propagated values were finite `-1024` with zero positive pixels. The temporary object-score logits were non-finite BF16 and registered masks were entirely `-1024`. The completed loaded-state scan covered 810 state tensors across the image encoder, position-map encoder, and feature-fusion modules; all values were finite. The actual prompt trace contained only the cached feature bundle, decoder, and object-score gate, so this run cannot locate the NaN layer. Its 2,193,264-byte payload-free audit is the only inference artifact; it emitted no face labels, segmentation manifest, or sidecar. No truth or quality scoring was involved. The corrected prewarm trace passed 29 focused tests (one Torch-only skip), compilation, and lock verification. A new RX 7900 GRE run using it is active as `f6e25070-3ee3-42a5-9bf8-c98d72fd9401`. Acceptance remains open and unchanged.

The in-progress `f6e...` target audit now includes the corrected frame-0 prewarm trace. Its numeric summaries show finite RGB, position-map, and normal-map inputs; the image encoder output is finite; the earliest non-finite module-boundary output is `predictor.pos_map_encoder_output`. In that output, the 1,048,576 `vision_features` values and all FPN levels are non-finite; one positional-encoding level is partially non-finite. The model-state scan is complete and all 810 parameters/buffers are finite. Fusion, decoder, and object gating receive invalid position-map features downstream, so they are not yet implicated as the first source. This localizes the first observed bad activation to the position-map encoder as a whole, not to a specific layer or precision mode. No change to inference behavior or quality claims follows. A bounded child-module trace inside this encoder is the next diagnostic, and `f6e...` must still be polled to terminal status before another GPU run.

Run `f6e25070-3ee3-42a5-9bf8-c98d72fd9401` is terminal. It emitted a `done` event and a 870,373-byte Structured Asset sidecar with 14 assertions; all 14 assertions reference the single 11,662,546-byte audit by digest and size rather than embedding it. All five segmentation-manifest artifact byte counts and hashes validate. Raw label telemetry records 44,662/46,180 assigned faces and 1,518 sentinel `999`; the completion policy fills those faces in the derived output, but confidence stays unknown and no truth is available. The command exited 1 because the wrapper treated known pinned upstream stdout progress lines as errors despite the final `done` event. Source inspection confirmed those exact lines are emitted by pinned `utils/inference_utils.py`. The wrapper now accepts only those anchored progress forms and still fails on unknown text, explicit error events, or missing `done`; eight regression tests pass and `bash -n` passes. The run took 795,084.78 ms; RX 7900 GRE peak allocated was 13,140,590,592 bytes and peak reserved was 15,118,368,768 of 17,163,091,968 bytes. This is about 14.08 GiB reserved, below Ticket 04's frozen 16 GiB peak-memory ceiling in `api/runtime/adapters/parts/SELECTION_GATES.md`; the earlier “above 14 GiB acceptance limit” wording was incorrect and is superseded here. This peak does not prove the separate memory-recovery gate. The audit localizes the first non-finite branch to the whole position-map encoder as described above, but this process predates submodule-level tracing. New bounded child-module instrumentation is ready; its focused suite passes 33 tests (one Torch-only skip) and a fresh target run is needed. Ticket 04 acceptance remains open; no quality gates were waived.

### Same-predictor A-B-A proposal trace (2026-09-28)

A payload-free A-B-A diagnostic completed on the fixed 46,180-face Flamingo input. All three calls processed 4,096 prompt points in 32 batches and had the same prompt digest. Reusing generator/predictor A returned zero proposals on both A calls; a fresh generator/predictor B sharing the same loaded model returned 103. B's sampled image-feature tensors were finite (8/8 sampled values in each tensor), while A's first and final sampled features were non-finite (0/8 in most samples). The first predicted-IoU filter event was also stable across A's two calls (384 candidates, zero finite, zero retained) while B had 384 finite candidates and retained 87. This one instrumented run suggested a reused-versus-fresh predictor difference, but a follow-up varied substantially, so this pattern is not established as repeatable. No score, segmentation artifact, or quality claim follows. The trace's peak reserved memory was 13,157,531,648 bytes, below the 16 GiB device ceiling for this diagnostic only. It is not whole-workflow or recovery-gate evidence. The instrumentation can affect execution and records no payloads. Exact identities, failed integration attempts, interpretation limits, and test commands are documented in `api/runtime/adapters/parts/evidence/TICKET04_PROPOSAL_ONLY_ABA_2026-09-28.md`; the target summary SHA-256 is `8eca9b9e8540ac83e1a034d4ee987cc9a1ec48d8d58ed1ed259503842b3850fe`. Ticket 04 remains acceptance-open with all gates unchanged.

The next A-B-A run added bounded samples from selected encoder/fusion/decoder module outputs on the same mesh and prompt digest. It returned 28, 65, and 100 proposals for A, B, and A. The eight sampled values at the position-map trunk, neck, and full position-map encoder matched across these three calls. The first call's sampled image-encoder and feature-fusion outputs differed from the next two, whose samples matched; the proposal-filter digest for final A matched B. These are small samples only: they do not prove full-tensor equality, absence of other non-finite values, or a single cause. This follow-up varies from the prior A=0, B=103, A=0 trace and shows that the proposal-only result itself varies across runs. Peak reserved memory was 16,659,775,488 bytes (about 15.51 GiB), but four of 57 synchronized free-memory snapshots reported zero available bytes during the first A call. The reason global free memory reached zero while allocator-reserved bytes remained below the device total is unknown; this is a memory-pressure signal and does not prove full-workflow recovery. Summary SHA-256 `7cb231b0131ee0007c8704fd38771b08662bfea3df4d865e7bfd0edc260d16fe`; full report and test evidence remain in the linked A-B-A evidence document. No truth was used and no acceptance gate changed.

The child-trace target `f7e25070-3ee3-42a5-9bf8-c98d72fd9401` is terminal with explicit `GEOSAM2_NO_FACE_LABELS` and exit 1. It recorded 32 registrations, 192 propagated-logit summaries, and no face-label manifest or sidecar; audit size is 1,624,423 bytes. Coarse stage summaries now show non-finite values in the image encoder output (partial `vision_pos_enc[0]`/FPN[0]), then partial non-finites in the position-map encoder outputs. This differs from f6, whose image encoder outputs were finite and position-map encoder outputs were fully non-finite. The 810 loaded state tensors remained finite. The position-map child trace captured 429 child modules with no inventory truncation, but reported impossible counts on some BF16 outputs (negative or larger than tensor element counts); its `linear_a_q` first-child designation is therefore not accepted as root-cause evidence. The output counter needs a fail-closed numeric fix, and child tracing must cover both encoders because the earliest non-finite top-level branch varied between the two same-input runs. No precision cause is established; repeat variation remains descriptive only.

### Validated encoder child summaries (2026-09-28)

The f7 child outputs are unqualified for localization because Torch HIP integer reductions returned counts outside `[0, tensor.numel()]`; the claimed `linear_a_q` culprit is not accepted. Per the later whole-module target observation, image encoder and position-map encoder each had 402 loaded state tensors and feature fusion had 6, all finite. Frame inputs and the image encoder were finite in that observation, with non-finite values first at the whole position-map encoder. f7 differs: its coarse image encoder outputs were already partly non-finite. This same-input run-to-run difference means the next child trace must cover both encoder branches in actual forward call order.

The diagnostic schema is now v5. Torch count reductions use float64 accumulators and validate finite, positive, and sentinel counts as exact integers in `[0, numel]`; impossible values become type-only `diagnostic_error` summaries and never set a first-nonfinite marker. Named child tracing now covers both encoders in runtime execution order with per-encoder hook inventory, 1,024-child caps, a shared 2,048-output cap, eight tensor summaries per output, explicit omission/partial-coverage reporting, deterministic paths, and restoration. CPU tests cover counter bounds and summary-error handling, cross-encoder order, caps, and restoration. The focused suite passes 33 tests with one PyTorch-only skip; modified files compile and the v5 lock verifier passes. The active f7 process loaded earlier code; no GPU run used v5. The first bad internal layer remains unknown. No precision/inference/output behavior or acceptance criteria changed.

### Schema-v5 target trace (2026-09-28)

Fresh same-input truth-free RX 7900 GRE run `f8e25071-3ee3-42a5-9bf8-c98d72fd9401` is terminal with exit 1 and `GEOSAM2_NO_FACE_LABELS`; it wrote no face labels, segmentation manifest, or final sidecar. The schema-v5 trace covers 429 child modules under each encoder, with 866 outputs and zero omitted outputs. All 810 loaded model-state tensors and every recorded whole-module boundary (both encoders, fusion, cached features, and inputs) are finite. One position-map child output at `blocks.7.attn.qkv.linear_b_v` reported 2,126,638/2,195,200 finite values, while the preceding `linear_b_q` summary failed closed with a type-only diagnostic error. The later parent `attn.qkv` output reports all 6,585,600 values finite, as do the enclosing encoder outputs; pinned source adds the V branch into that parent tensor. The child and parent summaries therefore conflict, so the child count is unqualified as an actual non-finite activation and does not establish why the final workflow produced no labels. The complete propagated-output summary shows only the starting view retained proposals (32), while views 1–11 had none, and no faces were returned. No truth was inspected and no quality score was produced. At start and after exit, sampled VRAM use was 783,028,224 and 772,972,544 bytes; peak allocated/reserved and latency were not captured. Full record: `api/runtime/adapters/parts/evidence/TICKET04_GEOSAM2_V5_DIAGNOSTIC_F8E25071_2026-09-28.md`. Ticket 04 remains unaccepted pending the frozen quality/coverage/repeatability/topology/provenance/resource gates.

The v5 trace metadata now accurately names its float64 count accumulation; the source lock and adapter pin were updated, and 33 focused tests pass with one Torch-only skip. The exact first non-finite child remains unknown; do not change precision or acceptance thresholds without a controlled follow-up.

### Block-7 LoRA Q/V frame-zero cross-check (2026-09-28)

The first reviewed diagnostic invocation stopped before container creation because the wrapper expected Podman's `.Id` to include `sha256:`. Podman reports a bare 64-hex image ID; the runner was corrected to normalize it for report identity while passing the bare immutable ID to Podman. A subsequent one-frame frame-0 cache-prewarm probe completed on the RX 7900 GRE using the exact f8 Flamingo render bundle and topology map. Independent CPU summaries of temporary copied tensors confirmed non-finite values in `linear_b_q` (1,291,520/2,195,200 finite), `linear_b_v` (681,856/2,195,200 finite), and returned parent Q/V slices (360,192 and 194,880 finite, respectively). This resolves the prior contradictory child Q/V diagnostic as a bad earlier child count; it does not prove a first bad operation or precision cause because block input, base QKV, and both LoRA A outputs were not observed. Exact command, report/image identity, hashes, validation, memory sample limits, and interpretation: `api/runtime/adapters/parts/evidence/TICKET04_GEOSAM2_BLOCK7_QV_TARGET_CROSSCHECK_2026-09-28.md`. No face labels, segmentation, truth, or quality score were produced. Ticket 04 remains unaccepted.

#### Expanded QKV input/base/A/B boundary trace (2026-09-28)

The eight-event v2 follow-up observed non-finite values already at the outer block-7 QKV input (1,013,376/2,195,200 FP32 finite). Base QKV output had 3,040,128/6,585,600 finite; both LoRA A outputs had 9,048/19,600 finite; both B outputs and final parent Q/V slices each had 1,013,376/2,195,200 finite. Thus B is not the first bad observed boundary in this call; the non-finites reach QKV before its base and LoRA branches. The probe completed with only scalar counts, verified eight-event call order/shapes, and no face labels or quality result. It does not trace the earlier position-map encoder layers that produced the input. Full command, identities, hashes, validation, and limits are in `api/runtime/adapters/parts/evidence/TICKET04_GEOSAM2_BLOCK7_QV_BOUNDARY_CROSSCHECK_2026-09-28.md`. Ticket 04 remains open with all acceptance criteria unchanged.

#### Schema-v3 Hiera prefix and runtime-state cross-check (2026-09-28)

The independently reviewed schema-v3 trace added patch embedding, positional embedding, their block-0 sum, and read-only runtime precision/determinism flags to the existing Hiera boundaries. One fresh RX 7900 GRE frame-zero prewarm on the same locked Flamingo input found all 13 events fully finite. Runtime state at patch embedding was BF16 autocast enabled, deterministic algorithms disabled, both TF32 flags enabled, and evaluation mode. No setting was changed, and these observations do not establish the cause of the earlier v2 non-finite report; the discrepancy remains unexplained. The scalar report passed host identity/shape validation and no segmentation, labels, truth, or quality scoring occurred. Full identity, hashes, and sample limitations: `api/runtime/adapters/parts/evidence/TICKET04_GEOSAM2_HIERA_PREFIX_RUNTIME_STATE_CROSSCHECK_2026-09-28.md`. Ticket 04 remains unaccepted; all criteria are unchanged.

#### Bounded proposal-filter trace implementation (2026-09-28)

Added a separate opt-in diagnostic that verifies the pinned automatic-mask-generator source digest before temporarily wrapping predicted-IoU, stability, crop-edge, per-crop NMS, and cross-crop NMS filtering. It records scalar score/margin samples and digests/counts for retained indices, with caps of 12 views, 256 events per view, and 256 candidates per event. It saves no mask, image, coordinate, box, embedding, or prediction-tensor payload and restores hooks on normal return and exceptions. The actual pinned generator source SHA-256 matches the source lock (`cefa1934b0410119950514310728b811192e9f02584a479a7388bfe5a2e2bdac`). Focused CPU tests passed 5/5 and both modified Python files compiled. Independent review and target use remain pending; its temporary module-level hooks require an exclusive diagnostic process. The module is not wired into normal Modly inference and has not been run against the model/GPU, so it does not locate the observed repeatability difference or satisfy any segmentation acceptance gate. Full design and hashes: `api/runtime/adapters/parts/evidence/TICKET04_PROPOSAL_FILTER_TRACE_DESIGN_2026-09-28.md`.

#### Padded Hiera boundary cross-check (2026-09-28)

The v3 Hiera trace was held after independent review found that its first report contract incorrectly equated block-7 norm output and post-window QKV input shapes. The corrected schema-v2 runtime and host checks model the pinned 14x14 window partition, including padding of a 64x64 feature map; the realistic CPU fixture and malformed-shape guards passed 18/18 focused tests and independent review cleared the diagnostic for one target run. On the same f8 Flamingo frame-0 cache prewarm, the RX 7900 GRE run completed in 9,025.245 ms with all eleven captured Hiera boundaries fully finite; QKV input was 2,195,200/2,195,200 FP32 finite, shape `[25,14,14,448]`. The earlier v2 report on this exact locked input had only 1,013,376/2,195,200 finite at that boundary. This run-to-run difference confirms variability before or at the observed QKV boundary but neither locates a bad boundary in this run nor establishes a cause. Scalar JSON is `.modly-amd-runtime/results/ticket04-qv-crosscheck-20260928/hiera-block0-7-v3.json`, SHA-256 `be52091d6370d38f8df6298858ce2844c7c1150d9c830ec2cc9adae0c6207f88`; full source/device/lock identity and limits are in `api/runtime/adapters/parts/evidence/TICKET04_GEOSAM2_HIERA_BOUNDARIES_TARGET_CROSSCHECK_2026-09-28.md`. No segmentation, truth, quality scoring, or peak-memory trace occurred. Ticket 04 acceptance criteria remain unchanged.

#### Fifteen fresh same-input Hiera prewarms (2026-09-28)

To check whether the v2 non-finite result recurred with the corrected full-boundary collector, fifteen fresh schema-v2 frame-0 cache prewarms were run sequentially, each in a fresh project container using the same f8 Flamingo render bundle, face map, model/source/image identities, and seed. All 15 reports passed host identity validation. Every report had zero non-finite values across all eleven block boundaries; the QKV input was `2,195,200/2,195,200` finite and block7 output was `1,835,008/1,835,008` finite in every run. Per-run build/prewarm time was 8.90–11.54 seconds (mean 9.33 seconds). This screen did not reproduce the earlier v2 partial QKV input, which used an earlier/different diagnostic module; it does not estimate a variation rate for that earlier collector or identify the cause. All report digests and fixed identities are in `api/runtime/adapters/parts/evidence/TICKET04_GEOSAM2_HIERA_15_RUN_PREFLIGHT_VARIATION_SCREEN_2026-09-28.md`. No segmentation, face labels, quality scoring, or truth access occurred. After the batch there were no KFD PIDs; sampled VRAM increased by about 110 MiB from the earlier pre-run point sample, which is neither a peak measurement nor allocator-recovery proof. Ticket 04 acceptance and thresholds remain unchanged.

#### Exact-input proposal-filter target trace (2026-09-28)

The opt-in proposal-filter trace is now wired into the exclusive probe and has been independently reviewed. After correcting a BF16-to-NumPy capture error, the focused CPU suite passed 12/12. Separate fresh-process runs used the exact same f8 Flamingo render bundle, face map, checkpoint, source/dependency/model locks, image ID, seed, and parameters. Run 6 completed 12/12 generator calls and captured 780 events with no omissions; its label output still left 17,549/46,180 faces at sentinel `999`, so coverage failed. Run 7 captured two of twelve views and failed with no input points/masks. Their first view-0 predicted-IoU event differed: run 6 had 59/384 non-finite scores and 65 survivors; run 7 had 0/384 non-finite and 77 survivors. Run 7's view 1 produced 12,288 non-finite scores and no survivors. This establishes an exact-input proposal-stage difference and an empty-proposal failure, not the responsible operation or quality. A 20-call same-process proposal-only batch returned zero proposals nine times, 34 proposals ten times, and one proposal once, with a sharp VRAM-use regime change after repeat 10. Since the model and generator were reused, this does not compare separately generated assets or establish a cross-asset variation rate. Investigate sampled prompt points, image features, raw model scores, and synchronized memory before changing inference policy or thresholds. Full hashes and limits: `api/runtime/adapters/parts/evidence/TICKET04_PROPOSAL_FILTER_TRACE_F8_SAME_INPUT_2026-09-28.md` and `api/runtime/adapters/parts/evidence/TICKET04_PROPOSAL_ONLY_REPEAT_F8_2026-09-28.md`. Ticket 04 remains open; no frozen criterion or threshold changed.

### Known-truth score eligibility audit (2026-09-28)

Two current-code requalification runs on the exact frozen 1,536-face mesh have
matching source/model/settings, geometry, correspondence, render manifest,
seed, and topology identities, and both contain complete sentinel-free label
arrays. Their canonical arrays differ: run
`73b2d9ce-59ee-4d4f-9cc3-61a67b5d36a1` has 3 regions with counts 768/722/46;
run `e1a3e501-9e2b-4580-86d2-a6ec59536a24` has 2 regions with counts 768/768.
The frozen repeatability prerequisite therefore fails. The scorer was not
called and known-truth labels were not opened. Next investigate the divergence
without truth, then obtain a matching current-code output pair before the
frozen one-shot score. Full identities, array/artifact hashes, commands, and
decision: `api/runtime/adapters/parts/evidence/TICKET04_KNOWN_TRUTH_SCORING_ELIGIBILITY_2026-09-28.md`
(SHA-256 `4c97c6231a46f3a64b9f86cb75d0fec60f3be7d356731c306c97b94ffcd09014`).

### Repeatability and trace-counter source audits (2026-09-28)

Two independent read-only audits narrowed the next useful diagnostic. Prompt
coordinate RNG is already disfavored by matching coordinate digests despite
same-input score divergence; predictor reuse by itself also does not explain
the selected-feature observations. A controlled single-frame comparison of a
fresh predictor, a reset reused predictor, and a new predictor sharing the same
model could separate model-state changes, cached encoder features, and later
prediction-stage divergence while retaining exact source/settings/RNG-state
identities. Separately, previous Hiera reports have conflicting finiteness
counts across instrumentation versions, and one v5 child count conflicts with
its finite parent output. Before treating internal activations as causes, add
an exact-tensor dual count (device reduction and transient CPU count) at a
small set of fixed boundaries, with tensor shape/layout/alias metadata and
strict payload-free caps. Neither audit changed inference behavior or gates;
no truth was accessed. Reports:
`api/runtime/adapters/parts/evidence/TICKET04_GEOSAM2_RNG_LIFECYCLE_SOURCE_AUDIT_2026-09-28.md`
(SHA-256 `de65312e89b859ab3ccdb869ee283aad3d0a9187eb5bdca491ce33f3bed26888`)
and `api/runtime/adapters/parts/evidence/TICKET04_GEOSAM2_DIAGNOSTIC_COUNTER_AUDIT_2026-09-28.md`
(SHA-256 `2f05ae4478acb73985fa2b278bcbe868b68988b7e965d3e3f0e6a3fb7e2635c1`).

### Dual-counter RX 7900 GRE cross-check (2026-09-28)

The corrected one-frame schema-v4 diagnostic ran twice on the same recovered
Flamingo mesh/view, pinned source, model, image, and runtime settings. At the
post-window QKV input, separate LoRA Q/V outputs, and returned parent QKV output,
ROCm-device float64 counts agreed exactly with independent CPU/NumPy counts;
all captured values were finite. The thirteen original Hiera boundary summaries
were also all finite on both calls. This validates agreement for these tensors
in these two frame prewarms only. It does not explain full-segmentation
variation/failure, reproduce the older partial counts, establish full-workflow
repeatability or quality, measure peak VRAM, or pass Ticket 04. No segmentation,
truth access, threshold change, or inference-setting change occurred. Full
identities, commands, outputs, caveats, and counts:
`api/runtime/adapters/parts/evidence/TICKET04_GEOSAM2_DUAL_COUNTER_TARGET_CROSSCHECK_2026-09-28.md`.

### Ticket 04 predictor lifecycle target result (2026-09-28)

The opt-in fixed-view diagnostic completed in three fresh RX 7900 GRE runtime
processes. On each process's first encoder pass, captured image features and
prediction tensors were non-finite; resetting the predictor and repeating the
same image with restored random states produced fully finite outputs. A new
predictor sharing the model produced identical digests to the repeated call,
and recovered digests matched across all three processes. Model parameter,
buffer, and captured random-state digests remained unchanged. This isolates a
reproducible first-use failure for the one Flamingo view tested, but does not
establish its cause or a general treatment. A repeated pass is only a
candidate to verify through the actual registered segmentation workflow; no
production behavior changed. No truth, quality scorer, or other generated
topology was used. Ticket 04 remains open. Full run identities, hashes, and
limits: `api/runtime/adapters/parts/evidence/TICKET04_GEOSAM2_LIFECYCLE_TARGET_RESULTS_2026-09-28.md`.

### Finite-feature retry candidate and car development fixture (2026-09-28)

A one-time replay candidate is implemented behind
`MODLY_GEOSAM2_FINITE_RETRY_CANDIDATE=1` and remains disabled by default. It
checks cached image features after `set_image`, resets and repeats identical
inputs once only when features are empty/non-finite, and fails closed if the
retry is still invalid. The proposal audit records the helper digest and
bounded invocation/retry/failure counts. The retry and empty-proposal focused
CPU suites pass 17/17, and an inspected host-built wheel includes the helper.
The first registered RX 7900 GRE workflow run had zero invalid first passes
and zero retries, and failed raw coverage/repeatability. A second workflow
attempt was interrupted during connected-component CPU processing and
produced no new artifacts. Neither run tested a successful retry, so the
candidate is not selected and does not pass any Ticket 04 gate. Current
adapter SHA-256 is
`6d9a245728d88d355c2cc2da4d35c87e1b8dbb7075ecd992cac29562cd965621`; helper
SHA-256 is
`16785f905e0e3f24074c6e451311d3f38d25b8a5e032ca110278e30e18541d1e`. Evidence:
`api/runtime/adapters/parts/evidence/TICKET04_FINITE_FEATURE_RETRY_CANDIDATE_2026-09-28.md`.

Historical note: the procedural 656-face car described in the original run record below was a poor vehicle example. Its fixture, labels, generated model, and saved preview were deleted on 2026-10-01 at the user's direction. It is not part of the current workspace and must not be recreated or used. The user-provided realistic SUV is the intended car sample.

The first registered target run with the opt-in retry candidate completed its
segmentation stage on the RX 7900 GRE and wrote 48 digest-valid artifacts, but
the wrapper returned exit 1 after the `done` event; the exact wrapper failure
cause was not captured, so the process is exploratory. The audit confirms 12
image calls and zero non-finite first passes or retries. Raw labels covered
only 768/1,536 faces (768 sentinel 999); completion filled the rest as derived
labels. Its one-region output also differs from earlier two- and three-region
same-input outputs. No truth was accessed or scored. The retry was not
exercised, and Ticket 04 remains open. Full measurements/hashes:
`api/runtime/adapters/parts/evidence/TICKET04_FINITE_RETRY_REGISTERED_WORKFLOW_2026-09-28.md`.

A second registered workflow attempt used the same frozen fixture and opt-in
candidate with pipeline logging enabled. It was stopped while GeoSAM2 was
processing connected components. Saved pipeline statuses were
`input_json=0`, `container=1`, `stdout_capture=130`, and `event_filter=130`;
the container stack ended with `KeyboardInterrupt` during the mask-to-CPU
conversion in `connected_components_8_cpu`. The output directory contains no
new run artifacts; its two files predate this attempt. No retry count or
completed inference result can be attributed to this interrupted run. A
process check afterward found no project worker/container process. This is
interruption evidence only and does not satisfy or fail a segmentation quality
criterion. The retry candidate remains unaccepted and Ticket 04 remains open.
Logs and exact status file are under
`.modly-amd-runtime/results/workflow-geosam2/` for run
`a1bbfcd0-b853-4d3f-b607-e287270a6c5f`.

### Isolated finite-feature retry target probe (2026-09-28)

A separate locked one-view diagnostic used the fixed f8 Flamingo view 0,
production finite-retry helper, and RX 7900 GRE. The first image-feature pass
was non-finite; exactly one identical replay produced finite cached features,
and the fixed positive-point prediction outputs were finite. This is one
successful recovery in one diagnostic process, not a success-rate estimate or
segmentation-quality result. The probe did not run mesh lifting, post-process
the segmentation, access truth, or use the car fixture. The optional upstream
`_C` extension was unavailable and the predictor skipped its optional
post-processing path; no mask-quality conclusion can be drawn. Ticket 04 and
the candidate remain unaccepted. Full hashes, runtime, resource limits,
commands, packaging, and attempt history:
`api/runtime/adapters/parts/evidence/TICKET04_FINITE_RETRY_TARGET_PROBE_2026-09-28.md`.

A subsequent opt-in registered run used the same locked f8 Flamingo mesh and
12-view render bundle. It registered 14 proposals on the starting view and
none on the next view. On the third image-predictor call, features were
non-finite and remained non-finite after the one identical replay, so the
workflow emitted `GEOSAM2_INFERENCE_FAILED` and produced no face labels,
manifest, or Structured Asset. Pipeline statuses were input 0, container 0,
stdout capture 0, and event filter 1; the container completed normally after
reporting the inference error. This directly shows the successful isolated
retry does not generalize to every view. No truth or quality score was used;
Ticket 04 remains open and criteria are unchanged. Full report and hashes:
`api/runtime/adapters/parts/evidence/TICKET04_FINITE_RETRY_MULTIVIEW_WORKFLOW_2026-09-28.md`.


A follow-up audit found the earlier observer used retry invocation count as its view index. The corrected probe now tracks outer seed-view setups independently, with a regression for a successful replay. A fresh corrected RX 7900 GRE run again reached view 1 after finite view 0 features and 24/25 accepted/generated proposals, then failed after one non-finite replay; it emitted no final labels. This confirms the earlier view-1 failure for this sample but does not qualify the retry. Evidence and hashes are in `api/runtime/adapters/parts/evidence/TICKET04_FINITE_RETRY_VIEW2_SEQUENCE_DIAGNOSTIC_2026-09-28.md`. All acceptance criteria remain open.

### Image-path boundary diagnostic (2026-09-28)

A locked RX 7900 GRE probe captured image encoder, position-map encoder,
feature-fusion, decoder high-resolution features, and `forward_image` outputs
for the exact Flamingo view 0 lifecycle sequence. In the cold first setup,
`image_encoder` output `backbone_fpn[0]` was already partly non-finite
(2,232,064/16,777,216 finite values); the position-map output was finite, and
invalid values propagated to `conv_s0` (0/2,097,152 finite outputs). The same
input was fully finite after predictor reset and with a new predictor. This is
a cold-first-use lead from one run, not a causal finding or a segmentation
result. The diagnostic did not produce labels or access truth. Evidence and
limits: `api/runtime/adapters/parts/evidence/TICKET04_IMAGE_PATH_BOUNDARIES_F8_2026-09-28.md`.
Ticket 04 remains unaccepted; gates are unchanged.

### Proposal-to-lift trace integration (2026-09-28)

An opt-in, locked and bounded trace now connects accepted proposals,
predictor registration, and face-lift inputs through the GeoSAM2 adapter.
It treats propagated object IDs as global across rendered views. The trace
retains counts and keyed aliases rather than masks, coordinates, or face IDs.
Focused validation passed 19/19 tests, Python compilation, a synthetic
cross-frame join, and helper-lock verification. No GPU inference or frozen
truth was used, and the frozen macro-IoU remains 0.50 versus 0.90 required.
Evidence: [`TICKET04_PROPOSAL_LIFT_TRACE_INTEGRATION_2026-09-28.md`](../../../api/runtime/adapters/parts/evidence/TICKET04_PROPOSAL_LIFT_TRACE_INTEGRATION_2026-09-28.md).

### Five corrected-lock image-path repeats (2026-09-28)

Five fresh RX 7900 GRE project-container processes ran the current locked
six-boundary probe on the same view 0 input and seed. All 15 lifecycle calls
were complete and finite, including the cold first image-encoder outputs. The
first image-encoder digest varied in all five fresh runs, while the reset and
new-predictor role digests were stable across the five runs. Thus the earlier
single cold non-finite event did not recur, but this run set confirms digest-
level output variation. Because only hashes and finite counts were captured,
the magnitude and quality impact of those differences are unknown. The probe
has not run full segmentation and no quality gate passed. Full artifacts,
hashes, and limits:
`api/runtime/adapters/parts/evidence/TICKET04_IMAGE_PATH_BOUNDARIES_FIVE_RUNS_2026-09-28.md`.
Ticket 04 remains unaccepted.

### First image-encoder numeric summaries (2026-09-28)

The current locked diagnostic now records aggregate min/max/mean/spread for
the first image-encoder FPN tensor without retaining its values. Three fresh
RX 7900 GRE runs used identical locked view-0 inputs; all captured values were
finite. The cold first-pass ranges and spread varied across runs and differed
from the identical reset/new-predictor summary observed in all three. This
shows a repeatable distribution-level difference in this probe, not its cause
or any impact on segmentation labels. It did not run full segmentation or use
truth. Summary table, artifacts, hashes, and limits:
`api/runtime/adapters/parts/evidence/TICKET04_IMAGE_PATH_NUMERIC_SUMMARIES_2026-09-28.md`.
Ticket 04 remains open and no acceptance gate changed.

### Registered Flamingo retry failure rechecked (2026-09-28)

A saved registered RX 7900 GRE workflow attempt (`e7a3d64e-a3c1-4f08-96bc-c02f2a451fe0`)
used the opt-in finite-feature retry. It accepted 14 proposals on view 0, none
on view 1, then failed on the next image setup after one non-finite replay.
The workflow emitted `GEOSAM2_INFERENCE_FAILED`; no face labels or completed
asset were written. It therefore cannot be scored and is not the frozen
1,536-face known-truth fixture run. Evidence and exact statuses:
`api/runtime/adapters/parts/evidence/TICKET04_REGISTERED_RETRY_FAILURE_E7A3_2026-09-28.md`.
No gate changed.

### Frozen fixture registered run with current adapter (2026-09-28)

A fresh RX 7900 GRE workflow used the same frozen 1,536-face GLB and sidecar
bytes as the two earlier current-code runs. The container emitted a complete
`done` event and artifacts, but the host returned 1 because the event filter
rejected ROCm HIP attention descriptor debug lines. The exact captured stdout
was replayed through the narrowly updated filter and passed; no GPU rerun was
made after that host-only fix. This run's adapter hash differs from the prior
two, so its face labels are not a valid repeat comparison. Raw labels were
768 faces of label 29 and 768 sentinel 999; the completion routine filled all
faces to label 29, producing one canonical region. Truth stayed sealed and no
quality score was made. Exact hashes, resources, and status:
`api/runtime/adapters/parts/evidence/TICKET04_FROZEN_FIXTURE_RUN3_2026-09-28.md`.
No gate changed.
### Historical compacted proposal trace on the retired synthetic car (2026-09-28)

Historical run record only: the compacted trace helper was exercised on a second run of the same 656-face procedural car and topology revision on the RX 7900 GRE. That vehicle fixture and its saved model/preview were deleted on 2026-10-01. The
workflow completed in 507,446 ms using PyTorch ROCm; reported peak allocated /
reserved VRAM was 13,628,366,336 / 13,864,271,872 bytes. The 36,959-byte report
fits below the 49,152-byte cap and verifies the current helper lock, but is
still correctly marked partial: per-lift caps omit 370 input leaves and 480
join aliases. It records no raw IDs or label payloads and does not establish a
per-object-to-output-face-label mapping.

The raw model assigned 28/656 faces (27 with raw ID 12, one with ID 36); 628
were unassigned. Completion filled the partition, leaving 655 faces in one
canonical region and one in another. No fixture truth was read and no score
was run. Per-view accepted proposal counts were `[8,16,11,18,9,1,0,0,0,0,0,0]`,
different from the prior sample. This supports run-to-run proposal variation,
not a variation estimate or quality claim. Full bundle and hashes are recorded
in `api/runtime/adapters/parts/evidence/TICKET04_CAR_DEVELOPMENT_TRACE_2026-09-28.md`.
Ticket 04 remains unaccepted and every acceptance criterion is unchanged.

### First-use image-encoder stage diagnostic (2026-09-28)

A locked RX 7900 GRE lifecycle diagnostic captured the image encoder's patch
embedding, positional embedding, post-add input, and all 24 Hiera block outputs
for fresh, reset/reused, and new/shared-model predictor roles. The first call's
patch-embedding output digest differed from both warm roles; the positional
embedding matched across all three. The two warm roles matched at every
captured stage, while the first role differed from the patch embedding onward.
All captures were finite. Input, model-state, and recorded RNG-state digests
matched across roles, and model state did not change. This localizes the first
observed difference but does not explain its cause or measure segmentation
quality. No face labels or truth were accessed. Full report and bounded JSON:
[`TICKET04_IMAGE_ENCODER_INTERNAL_STAGES_2026-09-28.md`](../../../api/runtime/adapters/parts/evidence/TICKET04_IMAGE_ENCODER_INTERNAL_STAGES_2026-09-28.md),
[`digest-only capture`](../../../api/runtime/adapters/parts/evidence/TICKET04_IMAGE_ENCODER_INTERNAL_STAGES_2026-09-28.json).
Ticket 04 remains unaccepted; gates are unchanged.


### SUV box-reduction reserve-stop and process cleanup (2026-10-01)

The opt-in 32-prompt-batch bounded box-reduction path was attempted twice on
the 75,000-face SUV through Modly. Both runs were terminated by the board-wide
4 GiB reserve monitor before final labels, after VRAM rose close to the reserve.
Each produced five partial per-view label arrays only; neither was scored. The
first shell-group stop and a second Podman stopped-state report both left the
container Python task alive, which was then force-stopped by its exact observed
PID. The launcher now saves a unique container ID and the reusable monitor
kills and verifies the exact container cgroup before reporting stopped. A
CPU-only Podman container test passed, but no monitored inference stop has yet
validated the new cgroup cleanup path. Evidence and partial-run limits are in
`api/runtime/adapters/parts/evidence/TICKET04_SUV_VRAM_RESERVE_STOP_AND_CONTAINER_CLEANUP_2026-10-01.md`.
The GPU candidate remains unqualified; no acceptance gate changed.


### Rejected procedural car fixture retired (2026-10-01)

The 656-face procedural car fixture and its saved preview/evidence bundle were
deleted at the user's direction because the model was not a realistic vehicle.
Do not use or recreate it as a car example. Historical measurements above are
retained only as an account of earlier exploratory runs, not as current assets
or acceptance evidence. The user-provided realistic SUV remains the intended
vehicle sample; its segmentation quality gate is still open.
