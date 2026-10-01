# Modly AMD Semantic 3D implementation state

Updated: 2026-10-01

## Latest continuation (2026-10-01)

Corrected Ticket 04 same-code repeat `47eb0d54-bfbf-47f3-8913-0c9b92460bee`
completed all 12 views with exit 0 and a valid Structured Asset sidecar. The
host/device finite counts again match for the exact position-map
`vision_features` output (1,048,576/1,048,576) and the image convolution
child/parent (16,777,216/16,777,216 each). It used the same frozen geometry
and input sidecar as run `fbff9d6f-9cb6-426b-95d3-93986d7d072a`, but its
upstream, canonical, and completed face-label arrays are not byte-identical.
All arrays agree elementwise on 768/1,536 faces. The first run's canonical
output has two regions of 768 faces; the repeat's canonical output has one
region of 1,536 faces, even though its raw upstream labels contain two labels
of 768 faces. Therefore the required exact-output repeatability precondition
for the one-time frozen-truth scorer is not met. No truth was opened and no
score was made. Keep Ticket 04 open and investigate why the completion step
collapses the repeat's two raw labels. The 5-second supervisor recorded a
10,765,660,160-byte peak board use and at least 6,397,431,808 bytes free; it
never crossed the 4 GiB stop line and the workflow exited 0. Run evidence is
under `.modly-amd-runtime/ticket04-v7-repeat-20261001-run2/` and is excluded
from publication.

Same-input Ticket 04 follow-up run `b76b3cf1-1266-42ae-a9c8-f3f35e0c39b1`
completed all 12 rendered views and exited 0. Geometry and input-sidecar
digests match the frozen car input. On this run, both historical image
convolution child/parent outputs were fully finite, and device counts matched
independent CPU-copy counts at 16,777,216/16,777,216 for each. Nine captured
position-map `vision_features` summaries were fully finite at
1,048,576/1,048,576, but v6 accidentally host-checked the neighboring
`backbone_fpn[0]` output instead; it does not answer whether the historical
position-map discrepancy was a measurement error. Raw labels again cover
768/1,536 faces; the locked completion policy fills the other 768, resulting
in one valid final region. No truth was opened and no quality score was made.
The 5-second supervisor recorded a 10,710,204,416-byte peak board use with
6,452,887,552 bytes free; it never crossed the 4 GiB stop line, and use
returned to 3,737,739,264 bytes after exit. Version 7 now host-checks the exact
`vision_features` tensor and has 37/37 focused diagnostic/lock tests passing.

Corrected Ticket 04 same-input run `fbff9d6f-9cb6-426b-95d3-93986d7d072a`
completed all 12 views with exit 0 and a valid sidecar. It independently
checked the exact position-map `vision_features` output: host and device
counts both report 1,048,576/1,048,576 finite values. The image convolution
child and parent likewise match at 16,777,216/16,777,216 each. The earlier
non-finite count remains unexplained because it has not recurred in these
same-input runs. This v7 run's raw labels form two 768-face groups with no
unassigned faces, unlike the prior one-group, half-covered outputs. No fixture
truth was opened and no score was made: this exact v7 implementation still
needs a repeat run before the frozen one-time scorer can be considered. Its
5-second supervisor recorded a 10,996,363,264-byte peak board use and
6,166,728,704 bytes of minimum free VRAM; it stayed above the 4 GiB stop line
and returned to 3,643,510,784 bytes used after exit.

The Ticket 04 opt-in prompt/layer trace now has a versioned v6 diagnostic that
independently copies only three capped outputs to a temporary CPU snapshot:
the anomalous image-convolution child, its parent, and the position-map
encoder's first feature tensor. It records host and device finite counts plus
shape/stride/dtype metadata, then discards the tensor copy. The diagnostic
byte cap is 64 MiB per captured output, and each target is sampled once per
run. The lock verification and diagnostic suites pass **36/36**; Python
compilation passes. Its target run verified only the convolution child and
parent counts; the position-map host check was attached to adjacent
`backbone_fpn[0]`, so that result did not resolve the earlier position-map
anomaly. Version 7 corrects this target. A separate CPU-only venv run of the
current pending Ticket 05/08/11/12 changes passes **28/28** focused tests. A
single finite v7 run does not explain the earlier discrepancy or pass
segmentation quality/repeatability. The test venv is outside the repository.

Ticket 05's workflow-binding assembler now has a pure join helper for its
seven required inputs, with regression coverage for the real registered
segment-stage artifact shape (topology map, render manifest, camera metadata,
12 color views, and sidecar-derived evidence/mapping). The focused CPU suite
passes **5/5**. No Ticket 05 fixture images or truth were opened. This checks
workflow binding only; semantic quality, frozen development score, and Ticket
04 acceptance remain outstanding.

Ticket 08's project-owned process node now exposes the v2 region-inverse PBR
estimator and a bounded albedo-region-prior parameter (0–10, default 0). Its
stage provenance and inputs include the estimator/config identities and full
parameter set; unsupported normal and bump channels remain explicitly unknown.
The focused process suite passes **4/4**. The lambda=1 sweep remains
development-only; frozen fixture, held-out, rights, and RX 7900 GRE gates are
still open.

Ticket 09 implementation now includes explicit `ambiguous` fused claims and a
confidence-state summary without comparing model scores across adapters. The
project-owned material-identity node routes reruns through capability-scoped
evidence replacement, preserving other capabilities' assertions and active
user corrections. The focused fusion suite passes **14/14**, the node process
suite passes **12/12**, and TypeScript plus scoped lint pass. These are
implementation checks only; Ticket 09 acceptance remains gated on Tickets
05–08. See the implementation and test files for exact commands and evidence.

Ticket 10 now has an opt-in GeoSAM2 `segment-parts` stage-cache slice in the
mesh workflow. Its focused suite passes **21/21**, covering restart reuse,
targeted invalidation, failure non-caching, exact disjoint face coverage,
assertion-to-region binding, runtime identity, symlinks, integrity, partials,
and concurrent access; Python syntax compilation passes. Production identity
binds the locked ROCm/Torch build, GPU and runtime library, and device
selection; identity changes during cache lookup or validation fail closed.
The test environment itself cannot satisfy that production pin (it contains
no Torch package and host ROCm is 7.2.4 versus the pinned 7.14.0), so no live
cache reuse is claimed. This bounded implementation is not Ticket 10
acceptance.

Real-SUV GPU requalification run `808d2a8d-19d6-4109-8298-30fc601e3185`
completed successfully on the RX 7900 GRE with PyTorch ROCm. It rendered and
processed all 12 views; the opt-in exact-output bounded box reducer recorded
1,536 calls without failure. The validated Structured Asset contains 25
topology-bound regions covering all 75,000 faces exactly once, zero unassigned
faces, and 25 membership assertions. Sidecar `validation_state=valid`; the
output labels match its region face memberships. The model's confidence state
is explicitly `unknown`. Raw model masks covered 61,966/75,000 faces (82.6%);
the completion policy filled the remaining 13,034, so complete final coverage
must not be reported as complete raw prediction coverage. The largest region
contains 45,739 faces (61.0%), the second 6,705, and the smallest 7. A
qualitative render is saved only in the isolated runtime run folder. The
visible grouping is coarse and no ground-truth quality gate or user correction
has been completed; do not claim Ticket 04 acceptance from this run.

The worker ran about 19 minutes and exited with status 0. The locked
pre-inference budget allowed a 4 GiB display reserve and capped the PyTorch
allocator at 7.64 GiB; sampled board readings peaked around 10.7 GiB used of
16 GiB, leaving over 4 GiB free. No continuous peak sampler was active, so
these readings are diagnostic evidence, not a complete VRAM trace. The process
exited and sampled VRAM use returned to about 3.5 GiB.

The Electron production build passed using the bundled Node runtime and
`build-builtins.mjs` followed by `electron-vite build`; this confirms the
current source compiles but does not establish packaged launch or interactive
viewer acceptance. The implementation snapshot is published on GitHub `main`;
the bootstrap history and audit log remain present. Runtime outputs, caches,
build products, and development-generated correspondence datasets were
excluded from publication.

### Ticket 07 rotation-invariant texture classifier screen (2026-10-01)

Added one fixed 10-bin rotation-invariant local binary pattern descriptor to
the existing 30 topology-masked appearance cues and preregistered one RBF
mean-view candidate before scoring. The object-disjoint development run
retained the frozen gates, evaluated only 140 development views, and did not
open held-out truth. It failed four gates: macro-F1 0.8015 (>=0.85), minimum
class recall 0.40 for glass (>=0.80), supported coverage 0.84 (>=0.87), and
unknown abstention 0.40 (>=0.90); ambiguous abstention was 1.00 (>=0.90).
There were zero feasible threshold pairs. The candidate performed worse than
the existing mean-view RBF candidate and was not promoted. The focused
classifier suite passes 11/11, including rotation invariance, finite scoring,
model integrity, and artifact round-trip; process-node integration passes
12/12. The project test venv lacks Pillow, so the development-only scoring
used installed system Python 3.14 without GPU. Exact source, candidate,
unchanged thresholds, and report/OOF hashes are in
`api/runtime/adapters/material-identity/evidence/PROJECT_OWNED_RI_LBP_DEV_PREREGISTRATION_2026-10-01.md`.

Ticket 04's locked, opt-in prompt-seed-lift candidate was tried on the frozen
1,536-face car input in isolated run `8c745798-3ac2-4d68-b25b-d3d5fd37c6d9`.
It accepted 43 automatic proposals across the 12 views, including three at
the prompt seed, but its raw face-label array is byte-identical to the prior
run-4 output: one label covers 768 faces and 768 remain unassigned. The locked
completion policy then produces one 1,536-face region. The output sidecar
validates and the workflow exits 0, but this candidate did not improve raw
segmentation, so it was not repeated or scored; the frozen truth was not
opened. Ticket 04 remains open. Full run identities and hashes are in
`api/runtime/adapters/parts/evidence/TICKET04_PROMPT_SEED_LIFT_AND_LAYER_TRACE_2026-10-01.md`.

A separate selected-default-path trace on that same fixture,
`a5673975-b5bb-4a1f-bdd1-d60c415235a8`, exited 0 and produced a valid
Structured Asset with one region and the same 768/1,536 raw coverage. The
v5 diagnostic captured all 429 named modules in both encoders. Its stage
summary reports finite image-encoder output but a fully non-finite
position-map `vision_features` output; preflight then reports non-finite
object scores and only the `-1024` no-object mask sentinel. However, the
child-module summaries contradict their parent outputs twice: one image
convolution and one position-map MLP layer are reported partly non-finite,
while the immediately enclosing/next module output of the same shape and
dtype is reported fully finite. A separate synthetic BF16 count check on the
RX 7900 GRE matched the known count on both GPU and CPU, but it cannot settle
these live activation discrepancies. Treat the trace as a diagnostic
consistency failure, not a confirmed root cause; precision or inference
behavior has not been changed.

During the selected-default diagnostic, sampled board-wide VRAM peaked at
10.34 GiB used of 16 GiB (5.64 GiB free); the workflow's PyTorch peak was
4.81 GiB allocated and 6.96 GiB reserved. Latency was about 11.74 minutes.
This is one safe resource sample, not repeatability or quality acceptance.
The real SUV prompt-seed-lift development output had 28 regions versus 25 in
its earlier default run, but raw coverage fell from 82.6% to 68.9%; it has no
ground truth and does not establish an improvement. Keep the candidate
unselected and resolve the trace-count inconsistency before drawing a
precision conclusion.

## Implementation continuation (2026-10-01)

Latest continuation evidence: `apply_gpu_budget` now normalizes GeoSAM2's
unindexed `cuda` device before calling the allocator API; the focused budget
and first-convolution suite passes **14/14**. The fixed full-grid memory
chunking candidate changed automatic-mask `points_per_batch` from 128 to 32,
while preserving the full 4096-point prompt ceiling, all 12 views, M2M, and
quality thresholds. Its CPU factory test passes **1/1**. A guarded real-SUV
run confirmed the original 128 batch could not fit the current shared-GPU
budget: it OOMed in M2M refinement on the first view before proposals. The
32-point path has not yet been GPU-qualified, and the Ticket 05 development
policy locks still pin the prior GeoSAM2 adapter digest; keep those historical
locks intact pending an explicit versioned requalification plan.

The real SUV retry on the 32-point path (`da36e85a-0287-44da-8612-1025117a7935`)
completed rendering/input preparation but GeoSAM2 again failed in M2M before
producing proposals. PyTorch reported 2.15 GiB allocated, a 7.42 GiB process
ceiling, 13.21 GiB device-wide free, and an attempted 5.62 GiB allocation.
Only render, input, and failure-audit artifacts exist; no segmentation sidecar
was emitted. This is an unresolved peak-allocation path, not proof that the
smaller prompt chunks improve completion. Further GPU attempts are paused while
the pinned upstream allocation path is reviewed. No quality gate was changed.

The failed run's audit localizes the allocation request to pinned upstream
`batched_mask_to_box`, but it did not record the actual mask shape or confirm
the live generator batch setting. The new versioned v1 requalification path is
now wired behind `MODLY_GEOSAM2_BOUNDED_BOX_REDUCTION=1`; it remains off by
default. It verifies a new lock, requires the actual 32-point batch setting,
uses the exact-output reducer in chunks of four, records shape/dtype/counts
without mask values, and restores the upstream function on both success and
failure. The policy retains all prompts, views, M2M, thresholds, weights, and
the 4 GiB VRAM reserve; no historical policy lock was edited. Its seven-module
focused CPU suite passes **76/76**, and Python/launcher syntax checks pass.
No GPU requalification or VRAM-improvement evidence exists yet. Details:
`api/runtime/adapters/parts/evidence/geosam2-box-reduction-runtime-requalification-2026-10-01.md`.

Ticket 10 now also has a narrow Electron process-extension cache prototype in
`electron/main/structured-stage-cache.ts`. Its identity binds validated input
sidecar/geometry digests, full extension dependency tree, built-in capability
identity, authorized parameters, and host/Node runtime. Reuse is deliberately
limited to the built-in JavaScript, no-weight Structured Asset validator;
Python, model, third-party, ambiguous, unsupported-output, or invalid assets
bypass caching. Assets pass Modly's live validator, paths are checked for
symlink escape, writes publish atomically, and hits preserve supported result
fields. Six cache tests plus four workflow-store tests pass **10/10** with the
platform-bundled Node 24.21 runtime; focused ESLint is clean. This does not
satisfy Ticket 10: it does not cache expensive semantic stages or prove full
workflow restart/resume, targeted descendant reruns, or topology invalidation.
Its declared ticket dependencies (03, 04, 08, 09) are not all accepted.

Ticket 07's best project-owned development classifier now uses four-view
region pooling and an RBF head. Macro-F1 is **0.865**, minimum class recall
**0.60** (glass), supported coverage **0.92**, unknown abstention **0.40**,
and ambiguous abstention **1.00**. No threshold pair satisfies all gates.
Calibrating a separate learned unknown score at scales 0–4 did not improve
unknown abstention beyond 0.40; all candidates remain rejected and no held-out
truth was accessed. A preregistered 560-row procedural augmentation run
improved unknown abstention to **0.60** and ambiguous abstention to **1.00**,
but macro-F1 fell to **0.7643**, minimum class recall to **0.40**, and coverage
to **0.76**; no threshold pair met all gates. This candidate remains rejected.
Evidence and the unchanged frozen gate digest are recorded in
`api/runtime/adapters/material-identity/evidence/PROJECT_OWNED_PROCEDURAL_RENDER_AUGMENTATION_DEV_2026-10-01.md`;
the expanded focused suite passes **14/14**.

The existing project-owned material-identity process node now accepts a
digest-pinned uncalibrated candidate through the real Ticket06 view and
topology-mapping path. It emits only explicit `unknown` assertions with
`uncalibrated`, `unqualified`, and `not-accepted` provenance; no supported
material predictions or production model artifact are produced. Its CPU
Structured Asset/process suite passes **12/12**. Ticket 07's quality gates
remain failed and unchanged.

Ticket 08's preregistered region-prior sweep found a promising development
candidate at lambda 1.0: base-color MAE **0.02868**, SSIM **0.86246**, roughness
MAE **0.02343**, metallic MAE **0.00871**, and novel-light MAE **0.00333**, over
2,408/2,408 visible dev texels. All four frozen candidate outputs and hashes
are recorded in
`api/runtime/adapters/pbr/evidence/ticket08-region-inverse-v2-albedo-prior-sweep-results-2026-10-01.md`;
focused estimator/scorer tests pass **11/11**. This is development evidence
only; held-out acceptance, target hardware/VRAM, and other Ticket 08 gates
remain unverified.

The structured review panel now has a user action to reopen a validated GLB
and its matching structured sidecar in the existing Three.js viewer. The
frontend built with the installed bundled Node 24.21 executable; ESLint passed
and the Node test suites passed **40/40** and **114/114**. The focused Python
selection for GPU budgeting, Ticket 04 lock, Tickets 09–11, and Structured
Asset routes/schema passed **84/84**. The normal `/usr/bin/node` remains
unusable because its linked `libsimdjson.so.33` is absent; validation used the
separate platform-bundled executable and did not alter project requirements.

The seam-correction panel now checks viewer/sidecar face-count agreement and
current-topology face mappings, then highlights the exact face memberships
that will transfer before confirmation. Its synthetic car door/fender suite
passes **3/3**; TypeScript compilation and focused ESLint pass. This corrects
surface partition membership through the existing correction route; it does
not displace mesh vertices or prove the interactive viewer on the realistic
SUV. The complete Ticket 13 acceptance remains pending.

Ticket 11's populated CPU export/reopen fixture now also exercises the
canonical export handler/service, independent validator and Trimesh geometry
load, versioned sidecars, correction and topology mappings, provenance and
stage artifacts, explicit unknown PBR channels, normal/bump distinction,
GLTF material factors/slots, coordinate basis, deterministic output, and
negative validation cases. The focused suite passes **11/11**. Compatibility
telemetry for real pipeline GPU stages and viewer-rendered color-space behavior
remain unverified, so Ticket 11 is not accepted.

The Python launcher now discovers the flat `api/tests` directory with the
correct import root. The full Python suite (`npm run test:py`) used an isolated Python 3.14
environment installed from `api/requirements.txt`, with GPU visibility disabled
and the uninstalled P3-SAM source explicitly marked absent. It reached **813 tests in
178.7 seconds**: **1 failure, 4 errors, and 5 skips**. The failure is the
preregistered Ticket 07 gate-file SHA mismatch; the baseline bytes are not
available and the frozen pin remains unchanged. The errors are stale Ticket
05 source identities plus a Ticket 07 batch run blocked by the same immutable
gate pin. The canonical image integration is skipped under the GPU hold, not
counted as a pass. Python tests are still not all passing.

After the Ticket 13 75k-face drag optimization, current frontend checks pass:
`npm run test:node` **114/114**, `npm run lint`, and `npm run build`. Build
warnings remain for the optional `sharp` install script and stale Browserslist
data. These checks do not qualify manual interaction or GPU inference.

Ticket 13 now indexes face centers once into bounded spatial bins, examines
only the latest drag segment (up to 128 samples), caps stored pointer history
at 256 points, and limits preview publication to 100 ms. Selection/removal
validation uses sets. Two CPU helper tests and the focused build/lint passed.
A very long pointer jump can miss faces between capped samples; normal pointer
events are processed as successive segments. The edit is an on-surface
face-transfer correction, not geometric vertex/seam displacement; no manual
SUV interaction was run.

Ticket 07's additional dev-only ridge comparison did not pass its unchanged
gates: ridge 0.5 reached macro-F1 0.7049, minimum recall 0.35, coverage 0.75,
unknown abstention 0.35, ambiguous abstention 0.60; ridge 20 reached 0.6797,
0.35, 0.70, 0.40, and 0.45 respectively. No candidate had a feasible gate
pair. The report is `.modly-amd-runtime/ticket07-project-owned-dev-20260930-ridge/report.json`
(SHA-256 `31a5a8fe1abc8350464a978b3aede59f8ed77bdc9a407f228a54a6fae3853d61`).
Heldout truth and GPU were not used. The dev evaluator now persists the
truth-free feature artifact before joining development labels.
The later supervised-abstention candidate improves macro-F1 to 0.7138 and
ambiguous abstention to 0.90, but minimum recall is 0.50, coverage 0.78, and
unknown abstention 0.30; it still has no feasible threshold pair.

Ticket 08 now has a truth-free provisional pipeline: RGB appearance clusters
from allowlisted training observations pass through the Modly Ticket 06
processor, map to validated unchanged topology, and feed the project-owned
inverse renderer. Raw maps were durably saved before evaluation. The audited
fixture has no separate development target split or one-way development
scorer, so those maps remain unscored; held-out scoring was not run.

Ticket 04 now has one completed RX 7900 GRE diagnostic using the locked f8
single-view input. The first-convolution hooks captured the patch embedding
input, Conv2d input/output, weights, bias, and runtime flags. Within this run,
the three lifecycle roles had identical transformed-input and first-Conv2d
output digests (`175f9325…fe24` and `f68204e6…2b52`). This shows no first-convolution
variation across those roles in this one run; it does not establish run-to-run
reproducibility or explain the earlier downstream segmentation divergence.
Elapsed time was 66.3 s. The intermediate capture remains diagnostic-only and
does not count as Ticket 04 acceptance.

That run exposed an unsafe telemetry mismatch: PyTorch reported almost all
VRAM free while ROCm SMI showed about 3.6 GiB already occupied by system and
desktop use. Its recorded allocator allowance therefore cannot be treated as
the shared-GPU cap. The policy now cross-checks PyTorch free memory with the
system-wide DRM VRAM usage counter, fails closed if the active device cannot be
matched uniquely, and records both values. The updated policy, integration,
and lock checks pass **17 focused CPU tests** plus Python and shell syntax
checks. The initial diagnostic output remains preserved as evidence of the
mismatch; it was not run under the corrected cross-process budget.

The corrected device-wide counter path was verified in both the read-only
container check and a no-model ROCm budget smoke check on the RX 7900 GRE. The
smoke check saw 4.12 GB system-wide VRAM use and applied an 8.14 GiB PyTorch
allocator ceiling after reserving 4 GiB. This is an allocator cap, not a
hardware partition, and does not prevent GPU compute contention.

A guarded end-to-end SUV attempt then stopped before GeoSAM2 model load. The
worker calls `inference.init_env()`, which returns the unindexed device
`torch.device("cuda")`; PyTorch rejects that device when
`set_per_process_memory_fraction` is applied (it requires `cuda:0`). This was
reproduced in a model-free, guarded container check and is now the concrete
Ticket 04 runtime blocker. `gpu_budget.py` now resolves an unindexed CUDA
device to the current indexed device before reading memory and setting the
allocator limit. Its focused CPU suite passes **11/11**. A fresh model-free
container check with the GeoSAM2 initializer reproduced the original exception
before the fix; the device normalization is covered by stand-in tests, but the
production worker has not yet been rerun with the new implementation.

After that fix landed, a fresh real-SUV workflow run cleared the memory guard,
then stopped before its first proposal: upstream `refine_with_m2m` exhausted
the worker's **7.58 GiB** allocator ceiling while requesting another 512 MiB.
The system-wide device reading remained within the 4 GiB desktop reserve; the
failed run's proposal audit shows zero proposals/registrations and preserved
the exception stack. This identifies the next repair point as memory staging
inside full-coverage automatic mask generation. All 12 rendered views and
quality thresholds must remain unchanged; reducing point batch size is being
assessed as chunking, with quality verified against the frozen gate after the
worker can complete. No segmentation sidecar was emitted.

The API suite's failures and errors remain open; the canonical blocked
workflow is not counted as a success. No RX 7900 GRE end-to-end vertical
slice, manual correction session, or target VRAM telemetry has been completed.
No ticket acceptance gate was changed.

SUV importer verification (2026-10-01): the user-provided realistic SUV GLB at
`.modly-amd-runtime/suv-structured-import-run11/source.glb` was imported through
`POST /workflow-runs/from-mesh` (run `972a7cf6-2303-4735-92ef-ef12c8b6e92d`).
The saved sidecar was then passed through the real `POST /structured-assets/validate`
handler with its workspace correctly pointed at this run; it returned
`validation_state=valid`, topology revision
`sha256:f65ea5a3000e4b366a8628a7360c7164f8b6fa4524b4186d5ba1ce1e0ae0a057`,
and source geometry digest
`sha256:39eae91adc8dc42f5ab896579d666aa00c806b80ce1d9a4f0fdadcaea5dbed68`.
This verifies only CPU import and sidecar validation. The sidecar has no segmentation;
SUV semantic/material processing, manual correction, export/reopen, GPU metrics, and
the complete vertical slice remain outstanding. The former GPU pause marker was
renamed after the user's explicit resume; further target runs are paused during
source-level memory investigation.

Additional 2026-10-01 implementation notes:

- Ticket 09 now groups active corrections by target and property. Conflicting
  correction values remain unresolved with their IDs and model evidence
  preserved; agreeing corrections resolve deterministically and retain every
  contributing correction ID. Its focused fusion suite passes **12/12**.
  Ticket acceptance remains open.
- Ticket 07's separate supervised-abstention ridge candidate uses only
  object-disjoint training folds. Supported labels use one-hot targets,
  unknown uses all-zero supported scores, and ambiguous uses uniform scores.
  Its one bounded dev run scored macro-F1 0.7138, minimum recall 0.50,
  supported coverage 0.78, unknown abstention 0.30, and ambiguous abstention
  0.90; no feasible threshold pair passed. Evidence:
  `api/runtime/adapters/material-identity/evidence/PROJECT_OWNED_SUPERVISED_ABSTENTION_DEV_2026-10-01.md`.
  Heldout truth and GPU were not used; gates remain unchanged. Its direct
  focused classifier/evaluator suite also passed 10/10 under system Python
  3.14 with Pillow; the project API venv cannot import the evaluator because
  that venv lacks Pillow.
- Ticket 08 now has a CPU-only training correspondence builder using explicit
  mesh/UV/view/visibility inputs. Its focused correspondence and projection
  checks pass 10/10. A truth-free provisional path has also projected three
  appearance-cluster regions through Ticket 06 and durably saved raw PBR
  output before scoring at
  `.modly-amd-runtime/ticket08-provisional-training-only-run10/ticket08-raw-provisional-output.npz`
  (SHA-256 `fe8f129e744afbffb6262506bafc12bc1f8a453775df0ed0e8f8b9220d244358`).
  Its candidate manifest SHA-256 is `c18da2c26a29ff6fd320c166b89a3e90f00b3094a195c98bf66b2bc16f48613f`.
  An initial candidate measurement used the v1 dev fixture; its maps had MAE
  0.1288/0.2763/0.2914 (base/roughness/metallic) and novel-light MAE 0.0034.
  The fixture repeated identical training frames, so v2 now provides varied views.
  During the attempted preparation, a metadata JSON in the nominal train-only directory
  was opened and found to contain heldout/scoring metadata; that read is
  excluded from candidate inputs, no values were used, and the agent stopped
  using that file. A deterministic RGB clustering producer now creates
  unqualified provisional masks, but it does not pass Ticket 06 acceptance
  and is not a substitute for a separate dev target split/scorer. The separate
  preregistered training scene input has been
  live-verified at SHA-256 `a460ee742fefb0cd1f29c55bf8f4290de0bbe08556fca61daabb200c28e52dc2`
  with only four camera matrices and a three-light training rig; candidate
  preparation is restricted to that source and the five named training-only
  NPZ members in its existing allowlist.
- Ticket 08 now also has a separate seeded development fixture and one-way
  scorer at `api/runtime/adapters/pbr/development_fixture_v1.py` and
  `score_development_v1.py`. Candidate inputs and dev targets are separate
  hash-bound NPZ files; the scorer accepts only output maps and visibility and
  runs no estimator. Its v1 isolation/scorer suite passes 4/4, including
  unknown texels. The v2 view-diverse fixture/scorer suite passes 5/5. A
  candidate was measured on v2 as recorded below; neither result is acceptance
  or generalization evidence.
- Ticket 07 tested one affine view-plane plus photometric training augmentation
  on object-disjoint folds. It did not improve enough to pass any frozen gate:
  macro-F1 0.6276, minimum recall 0.30, coverage 0.69, unknown abstention
  0.40, ambiguous abstention 1.00 (all 20 ambiguous examples were instead
  called unknown). The unaugmented comparison was macro-F1 0.6261, minimum
  recall 0.40, coverage 0.69, unknown abstention 0.45, ambiguous abstention
  0.45. No heldout data, GPU, weights, or gate edits. Evidence:
  `.modly-amd-runtime/ticket07-project-owned-view-affine-20261001-v1/report.json`.
- Ticket 04 now has a CPU-authored, source-bound first-convolution capture
  helper that records digests for the actual patch input, Conv2d input/output,
  per-call weights/bias, and read-only precision flags. Its focused CPU tests
  pass 2/2; Python compile, shell syntax, and source-lock checks pass. The
  host runner exited 78 before container setup because the pause marker was
  active at that earlier point. The marker was later renamed at the user's
  explicit resume; the subsequent SUV attempt and unresolved memory cause are
  recorded at the top of this file.
- Ticket 07's latest affine-view/framing and photometric augmentation run
  failed unchanged gates (macro-F1 0.6276, minimum recall 0.30, coverage 0.69,
  unknown abstention 0.40; all ambiguous examples were classified unknown).
  Exact ambiguous status recall was 0.00. Focused tests passed 8/8 under system
  Python 3.14 because the project API test venv lacks Pillow. Evidence:
  `api/runtime/adapters/material-identity/evidence/PROJECT_OWNED_AFFINE_AUGMENTATION_DEV_2026-10-01.md`.
- Ticket 08 v1's initial estimator measurement exposed identical training
  frames. A new v2 scene now uses three view-diverse camera poses, independent
  hash-bound target/input files, and strict input allowlisting. The current
  deterministic three-start bounded GGX candidate fit at 32x32 and was
  expanded using the existing nearest-map rule to score 2,408/2,408 visible
  texels. Development errors: base color MAE 0.0333, SSIM 0.5821; roughness
  MAE 0.0259; metallic MAE 0.0327; novel-light MAE 0.0032. SSIM remains below
  the unchanged 0.85 threshold; no frozen gate was applied. Candidate output,
  score, and provenance are preserved under
  `api/runtime/adapters/pbr/evidence/ticket08-development-fixture-v2/`.
  The complete Ticket 08 test selection passes **61/61** in the dependency-
  complete Python 3.14 environment.
- Ticket 10 adds API process-run telemetry with host executor identity,
  monotonic duration, timestamps/status, and explicit unknown backend/device/
  VRAM when processors do not report them; records are atomically persisted.
  The Electron Python-runner success path contributes host timing, and its
  failure/cancellation paths now persist explicit failed/cancelled telemetry.
  Workflow cancellation terminates the active extension process; JS worker
  failures also report host telemetry. CPU checks passed: API process runs
  13/13, Electron runner/workflow-store focused checks 14/14, and Python
  compilation. Backend/device/VRAM stay unknown unless a processor reports
  them; no hardware telemetry is implied.
- GPU-hold-aware unit corrections were made after the 751-test run:
  quarantine metadata now reports before the model-load guard, collection
  sanitizer tests isolate their fake registry, and the canonical generation
  integration check skips explicitly when the real pause marker prevents
  inference. Their focused suite passes **52 tests with 1 explicit skip**.
  The latest dependency-complete 813-test run has 1 failure, 4 errors, and 5 skips;
  frozen source/gate and optional-runtime issues remain visible and unchanged.

## Implementation and validation refresh (2026-09-30)

The project-owned material-identity and inverse-rendering candidates are now
registered Modly workflow process nodes. Ticket 07's current development-only
candidate still fails unchanged gates (macro-F1 0.6913, minimum class recall
0.35, supported coverage 0.70, unknown abstention 0.45, ambiguous abstention
0.55); held-out truth was not read and no GPU was used. Ticket 08's process
node preserves caller topology, checks topology-bound view correspondence and
material-region inputs, records outputs/provenance, and leaves unsupported
channels unknown. Its CPU process tests pass, but its candidate has not passed
the frozen map or novel-light quality gates and has no RX 7900 GRE evidence.

Ticket 13's review panel now previews which configured workflow stages a saved
correction would rerun, which outputs would be replaced or preserved, and
which capabilities/blockers are missing. Running the listed stages requires
an explicit user action. Seam edits, seam undo/redo, and property corrections
all request a preview. `npm run build` and `npm run lint` pass using temporary
Node 26.10.0; the build reports the existing optional `sharp` install-script
warning and stale Browserslist data. The web TypeScript check still reports
pre-existing Asset Library test/source errors; it reports no current errors
in the Structured Asset review panel or workflow store.

GPU work remains paused. A shared GPU execution guard now checks the project
and workspace pause markers at the model registry, generation, and workflow
boundaries; its focused guard tests pass. The guard does not resolve the
shared-desktop impact or authorize a new inference run.

Validation limits: the final focused Structured Asset, correction, cache,
export, Ticket 08, and pause-guard run passed 78/78 CPU-only tests. The Ticket
07 classifier and its complete Ticket 06-to-07 process-node tests passed
16/16 under system Python 3.14 with Pillow available. The Python test launcher
now accepts an explicit interpreter through `MODLY_API_TEST_PYTHON` and
preloads installed `typing_extensions` before exposing the API's compatibility
module. The first full API run used system Python 3.14 and failed from
environment/import-path issues. The project-environment full discovery then
ran 530 tests and reported 45 errors, one failure, and 7 skips. Its
remaining failures include Ticket 07's preregistered gate-file digest mismatch
(current `SELECTION_AND_GATES.md` digest `bbdb9e...`; pinned digest
`6998d8...`) and optional/incompatible runtime imports. Do not repin or
reconstruct the gate file: the baseline bytes are not recoverable in this
checkout. Node tests pass 103/103. Overall acceptance remains open, including
Tickets 04, 05, 07, 08, 09–13 and the full RX 7900 GRE headless vertical slice.

## First-use image-encoder diagnostic (2026-09-28)

A new opt-in, source- and lock-pinned diagnostic captured the image encoder's
patch embedding, position embedding, position-added input, and outputs from all
24 Hiera blocks for three predictor lifecycle roles on the RX 7900 GRE. It
completed in 47.6 seconds. The first setup's patch-embedding output digest
differed from both warm roles; the two warm roles matched at all 27 captured
stages. The position-embedding digest, input, recorded random state, and model
parameter/buffer digests matched; all captured values were finite. This
narrows the first measured difference to the patch-embedding operation, but
does not identify its cause or establish segmentation quality. No face labels
or truth were read. The focused diagnostic tests pass 2/2; lock verification,
shell syntax, and Python compilation pass. Evidence:
`api/runtime/adapters/parts/evidence/TICKET04_IMAGE_ENCODER_INTERNAL_STAGES_2026-09-28.md`
and its digest-only JSON. Next: reproduce the patch-embedding difference and
inspect the ROCm operation without changing candidate settings or acceptance
gates.

## Compacted trace live check and classifier source recheck (2026-09-28)

The compacted Ticket 04 proposal trace helper was exercised on a second run of
the same procedural car on the RX 7900 GRE. The workflow completed using
PyTorch ROCm in 507,446 ms. Raw output assigned 28/656 faces and left 628
unassigned; completion filled those faces but did not improve raw model
coverage. The 36,959-byte trace fit under its 49,152-byte cap and verified the
current helper lock, but stayed partial because per-lift caps omitted 370
input leaves and 480 join aliases. Proposal counts varied from the first
sample, confirming variability without estimating its rate or proving quality.
Truth was not read and no score was run. Full bundle and limits are documented
in `api/runtime/adapters/parts/evidence/TICKET04_CAR_DEVELOPMENT_TRACE_2026-09-28.md`.

A fresh Ticket 07 primary-source recheck found no eligible material classifier:
FMMC still lacks a released pinned checkpoint/terms and the required selective
taxonomy/topology/runtime evidence; ObjectFolder provides object-level labels
without Modly's selective output contract. No model or data was acquired or
run. Ticket 07 remains open with unchanged gates. Ticket 08's MatMart screen
and Ticket 04's native-segmenter screen also remain negative. Tickets 04, 07,
and 08 remain open; dependent tickets remain blocked by the audited graph.

New source screens added no ready models. DirectPart has car-part benchmarks,
but its official path outputs labels for a separate point cloud rather than
Modly's caller-mesh faces and documents CUDA rather than the target AMD device;
its report is `api/runtime/adapters/parts/evidence/TICKET04_DIRECTPART_OFFICIAL_SOURCE_SCREEN_2026-09-28.md`.
The Ticket 08 MaterialMVP recheck confirmed its default remesh/UV-wrap/downsample
path does not preserve caller topology and its CUDA stack is unqualified; this
is a recheck of an existing lead, not a new candidate. These screens did not
download or execute assets. The existing same-process lifecycle probe already
compares fresh/reused/new predictors with full feature/output digests and model
and RNG-state hashes; saved reports show the first image setup differs while
the two warm roles agree. Next diagnostic work should isolate the first
diverging internal image-encoder boundary, without changing model settings.

## Ticket 04 synthetic-car AMD run and opt-in trace (2026-09-28)

An RX 7900 GRE GeoSAM2 development run completed on the procedural 656-face
car fixture in 525,497 ms (reported peak allocated/reserved VRAM:
13,628,366,336 / 13,822,328,832 bytes). Raw model output assigned 28/656
faces (27 with ID 10, one with ID 34); 628 remained unassigned. Completion
filled the partition but yielded 655 faces under one canonical region and one
under another. The truth-label file was not read and no score was run. The
opt-in proposal-registration-lift trace verified its lock and ran, but its
report is partial: 39 rows and 176 lift aliases were omitted at the 48 KiB
limit. A compacted helper subsequently passed its 19-test suite and a 12-view
volume check under the same cap, but was not rerun on the GPU. The audit shows
accepted proposals on the first five views only (6, 16, 11, 18, 9); views 5–11
had none. This is one stochastic sample and does not establish a variation
rate or pass Ticket 04. Full evidence and bundle:
`api/runtime/adapters/parts/evidence/TICKET04_CAR_DEVELOPMENT_TRACE_2026-09-28.md`.

## Ticket 04 fresh native part-segmenter screen (2026-09-28)

SegviGen's documented path remeshes generated TRELLIS.2 geometry and calls for
an NVIDIA GPU with at least 24 GB. S²AM3D segments point clouds but has no
published complete face mapping for Modly's unchanged input mesh, and its
documented stack requires CUDA. Neither is ready for acquisition or target
probing. No models, fixtures, truth, or GPU were accessed. Ticket 04 remains
open with gates unchanged. Evidence:
`api/runtime/adapters/parts/evidence/TICKET04_FRESH_NATIVE_PART_SEGMENTER_SOURCE_SCREEN_2026-09-28.md`.

## Ticket 08 MatMart source screen (2026-09-28)

MatMart's CVPR 2026 paper is relevant to image-plus-geometry PBR recovery and
describes albedo, roughness, metallic, and unseen-UV completion. The reviewed
official publication trail does not establish an implementation, deployable
inference package, pinned weights/terms, caller-topology/material-region
preservation, AMD/RX 7900 GRE 16 GB support, or frozen channel/novel-light
quality. It was rejected before acquisition; no data, code, fixtures, or truth
were accessed and no acceptance gates changed. Ticket 08 remains dependency-
ready and open. Evidence:
`api/runtime/adapters/pbr/evidence/TICKET08_MATMART_OFFICIAL_SOURCE_SCREEN_2026-09-28.md`.

## Ticket 08 LightSwitch source screen (2026-09-28)

LightSwitch's official interface produces relit views and a Gaussian-splat
appearance model, not separate PBR maps bound to Modly's caller topology and
material regions. Its MIT code license does not establish complete weight
terms/identities. The documented A100 setup provides no AMD/RX 7900 GRE or
<=14 GiB evidence, and no frozen map or novel-light quality evidence exists.
Rejected before acquisition or evaluation; no files, fixtures, or truth were
accessed and no gates changed. Evidence:
`api/runtime/adapters/pbr/evidence/ticket08-lightswitch-official-source-screen-2026-09-28.md`.

## Ticket 07 fresh primary-source screen (2026-09-28)

FMMC (CVPR 2026) is a relevant masked-region material classifier, but its
official paper says code and data will be released and the inspected sources
provide no immutable inference-code/checkpoint identities or checkpoint
terms. The described inference path uses GPT-4V without a pinned hosted model
revision/offline route. Frozen Modly taxonomy, unknown/ambiguous and
metal-subtype abstention, topology-bound Modly output, AMD <=14 GiB evidence,
and frozen fixture quality remain unproven. Rejected before acquisition or
evaluation; no models, weights, data, fixtures, or truth accessed. Ticket 07
gates remain unchanged. Evidence:
`api/runtime/adapters/material-identity/evidence/TICKET07_FMMC_CVPR2026_PRIMARY_SOURCE_SCREEN_2026-09-28.md`.

MatPredict is retained as an unresolved synthetic-data lead only. Its current
README/dataset card describes per-pixel material labels, while its linked
paper describes released regressors for base color and roughness and says
each mesh has one uniform material. No trained segmentation checkpoint,
frozen-label abstention behavior, Modly topology binding, or RX 7900 GRE
evidence is established; its published dataset licensing and upstream asset
provenance also need reconciliation. No files, model, data, fixture, or truth
were acquired. Ticket 07 gates remain unchanged. Evidence:
`api/runtime/adapters/material-identity/evidence/TICKET07_MATPREDICT_OFFICIAL_SOURCE_SCREEN_2026-09-28.md`.

## Pattern and phases

- Pattern: full implementation in an existing application, governed by the audited ticket graph.
- Phase 0: recover the actual Modly host source while preserving the handoff. Complete: upstream source retrieved at `1476fd0b1c19c9ab177c1ca3ee4d1842119e9f65` and copied into this workspace without replacing the handoff.
- Phase 1: Tickets 01, 02, 03, and 06 have passed acceptance. Ticket 01 was reopened for FINAL_AUDITED_SPEC §15 camera/capture metadata revalidation and has now passed with digest-bound optional metadata plus a real headless worker no-op round-trip. Ticket 04 remains acceptance-open: same-code RX 7900 GRE runs 3 and 4 produced bitwise-identical canonical labels, enabling the one-time frozen score; macro-IoU was 0.50 (<0.90), raw model coverage was 0.50, completed coverage 1.0, and overlap zero. Its quality gate fails. The prior macro-IoU 1.0 and bitwise-repeatable runs are historical evidence only. Ticket 05 has its frozen ontology contract, verified 280-object fixture/evaluator, reviewed pinned Florence files, and synthetic AMD inference evidence. Florence's prompt/abstention adapter tests pass 10/10; its Modly integration still lacks packaged/registered `modly.semantic_adapters` discovery and installed-package asset-path validation. No fixture development predictions have been committed or scored; semantic quality and unknown-versus-ambiguous policy remain unaccepted. Tickets 07 and 08 remain open; their latest source screens found no eligible new candidate, while existing candidates remain stopped by source/rights or quality gates. Ticket 05 remains blocked by Ticket 04 and its remaining integration/quality gates; Tickets 09–13 remain blocked by unmet gates and transitive dependencies. Ticket 13 now records the user's car seam-edit requirements and uses a synthetic car development fixture, but remains blocked by Ticket 12.

- Ticket 05 frontier update: the frozen 8-role contract, object-disjoint cohorts, verified rendered fixture, and truth-boundary evaluator are implemented. Florence-2's exact nine-file snapshot is locally digest-pinned and reviewed; the corrected production `predict_jsonl` entrypoint completed network-disabled synthetic inference on RX 7900 GRE using the project's pinned Transformers 4.51.3 overlay. It returned a versioned header and preserved an unnormalized full-prompt category hallucination with a near-full-frame box on a uniform synthetic image. This is runtime/protocol evidence and a serious negative semantic signal, not a quality score. The 4.57.1 route failed Florence generation/cache compatibility and is excluded. The Modly renderer pins `FORCE_ROTATION=0` and binds camera transforms from `meta.json` into the render manifest. The automatic node-to-producer integration and public source-provenance binding pass together: 38 tests, 12 subtests, two non-failing synthetic NetworkX discovery warnings, in the project AMD container. A deterministic four-view exact-consensus policy and development-only runner are now frozen; the runner's synthetic boundary tests pass 6/6 and commit raw predictions before truth scoring. No fixture development predictions have been committed or scored. A read-only deployment audit found the Florence module is not yet packaged/registered as an installed `modly.semantic_adapters` entry point and its source-checkout-relative asset root will not work from site-packages; this must be resolved before claiming actual Modly node execution. Next are that deployment seam, then a truth-isolated development-only semantic evaluation; stop if any frozen quality/abstention threshold fails. Synthetic output detail: `api/runtime/adapters/parts/evidence/ticket05-florence-adapter-open-vocabulary-synthetic-2026-09-25.md`; model/runtime evidence: `api/runtime/adapters/parts/evidence/ticket05-florence2-amd-synthetic-preflight-2026-09-25.md` and the issue.
- Contract integrity: the v1 semantic evaluation contract's `status`, model/weight access flags, and initial limitation list remain immutable freeze-time declarations. An independent source audit confirmed its full-file digest is bound into the fixture manifest and tests, so changing only those metadata fields would invalidate the existing fixture verification. Current execution state is recorded in this orchestration ledger and the Ticket 05 issue; the contract JSON is unchanged.
- Acceleration: after the fixed Ticket 04 low-memory quality failure and upstream-default OOM, one upstream-density chunked preset was frozen at 100,000 points / 400 prompts / batch 4 / seed 42 / threshold 0.95 and run on the RX 7900 GRE. It used PyTorch ROCm after MIGraphX failed on a zero-sized dynamic tensor lowering. Coverage/determinism/non-overlap and 16 GiB VRAM gates passed (7.95 GiB reserved), but macro IoU was 0.7182 (<0.90), so the candidate remains rejected. Exact evidence: `ticket04-p3sam-upstream-chunked-probe.md` plus JSON. The complete focused Ticket 04 suite passes 41/41 in the project container. Additional source screens found no eligible candidates: Point-SAM/OneFormer3D (Ticket 04), FMMC/MatSim/MaRI (Ticket 07), Material Palette/SfD/MatSpray (Ticket 08). SuperMat has source-level per-view correspondence feasibility, but upstream model terms/identity, color-space semantics, runtime, and quality remain unresolved. Evidence is recorded in the ticket issues and dedicated evidence files.

- Ticket 04 source screen (2026-09-28): COPS (WACV 2025) was checked from official repository, benchmark, and dependency sources. It does not clear pre-acquisition gates: required code/weight terms and immutable identities are unresolved, its documented point-cloud/category workflow lacks arbitrary-mesh complete topology-bound face mapping, and its setup/benchmark documents CUDA-specific dependencies and CUDA requirement without a qualified COPS AMD route. No code or weights were acquired or run. Full evidence: `api/runtime/adapters/parts/evidence/TICKET04_COPS_PRIMARY_SOURCE_SCREEN_2026-09-28.md`; Ticket 04 remains open with all gates unchanged.
- Later phases: follow `MODLY_AMD_SEMANTIC_3D_HANDOFF/tickets/TICKET_INDEX.md` strictly; do not advance dependent work before blockers pass.

The GitHub repository matching this workspace, [`Rolaand-Jayz/modly-seg-tool-osp`](https://github.com/Rolaand-Jayz/modly-seg-tool-osp), was checked through the connected GitHub account on 2026-09-24 and is empty. The local Modly source and audited handoff are therefore the implementation source until the remote receives project contents. The local `.git` directory remains an injected, read-only empty mount, so local commit/status evidence is unavailable.

## Workspace evidence and constraints

### Dependency-ready ticket refresh and Ticket 04 A-B-A trace (2026-09-28)

Tickets 07 and 08 were refreshed in parallel because their declared dependency set (01, 02, 06) is accepted. Ticket 07's distinct MINC GoogLeNet Caffe route lacks an immutable checkpoint/weight terms, the required label/abstention behavior, and AMD evidence; it was rejected before acquisition or fixture evaluation. A separate Material Magic Wand screen found a 3D material-aware part-grouping task whose group IDs and part indices do not satisfy the required fixed material labels, abstention, or topology-bound Modly evidence. Ticket 08's distinct MIRReS route documents albedo/roughness/metallic outputs but forces CUDA extensions and does not guarantee fixed caller topology/region IDs; it lacks AMD, pinned model/weight terms, and full topology identity evidence. Reports are linked from their ticket issues; none changed an acceptance gate.

Ticket 04's opt-in A-B-A diagnostic is now integrated in the project-only proposal trace command, bound by a versioned lock to the diagnostic module and pinned GeoSAM2 generator/predictor source, and covered by focused CPU tests. One fixed Flamingo run completed three calls with an identical prompt digest: same generator/predictor A returned 0 proposals on both calls; freshly constructed B, sharing the loaded model, returned 103. A follow-up with selected module-output samples used the same input and prompt digest but returned 28, 65, and 100 proposals; its first A image-encoder sample differed from B and final A, while the selected position-map samples matched. Four of 57 synchronized free-memory snapshots in that second trace reported 0 available bytes during the first A call; the discrepancy from allocator-reserved readings is unexplained and records actual GPU memory pressure. These results show proposal and sampled feature outputs vary across runs; they do not identify cause or prove quality. The first JSON is under `.modly-amd-runtime/results/ticket04-variation-panel/flamingo-aba-20260928-cap128/`, SHA-256 `8eca9b9e8540ac83e1a034d4ee987cc9a1ec48d8d58ed1ed259503842b3850fe`; the follow-up is under `flamingo-aba-20260928-module-stages/`, SHA-256 `7cb231b0131ee0007c8704fd38771b08662bfea3df4d865e7bfd0edc260d16fe`. Full limits, identities, resource counters, failed setup attempts, and commands are in `api/runtime/adapters/parts/evidence/TICKET04_PROPOSAL_ONLY_ABA_2026-09-28.md`. The focused A-B-A/filter/repeat suite passes 16/16; shell syntax, compilation, and the lock/upstream hash verifier pass. Ticket 04 remains unaccepted; no truth was accessed and no gate changed.

- Initial project checkout contained the audited handoff and an empty `.git` directory, but no Modly application source. The visible `.agents/` and `.codex/` paths are injected read-only session mounts, not project files.
- Public Modly source was retrieved from `https://github.com/lightningpixel/modly` at upstream commit `1476fd0b1c19c9ab177c1ca3ee4d1842119e9f65` and copied into the writable workspace. The user explicitly directed that their custom install not be used; it was not inspected, invoked, or changed. API acceptance runs use a project-owned Python 3.12 venv inside `.modly-amd-runtime/`; AMD inference uses the same project-local Podman store.
- Docker's daemon is unavailable. The project-owned Podman runroot had a stale boot-ID record. Before resetting it, I copied it to `.modly-amd-runtime/run.stale-boot-20260925` and `/tmp/modly-podman-runroot-preserved-20260925`, saved a file manifest at `/tmp/modly-podman-runroot-manifest-20260925.sha256`, and created `/tmp/modly-podman-runroot-backup-20260925.tgz` (SHA-256 `cfe38646dd16aae8dc2c1220de37c73f13bbc5bf626335b89b0af9856ffaf774`). The active project runroot was recreated at `.modly-amd-runtime/run`. The runner now forces XDG state to that project-owned path and inspects the expected AMD images under sandbox escalation. Ordinary sandbox execution still cannot grant capabilities to `/usr/bin/newuidmap`; target containers therefore require the same elevated project-local invocation. No global/custom ROCm or Torch install was changed. The AMD base image `rocm/pytorch:rocm7.14_ubuntu24.04_py3.12_pytorch_release_2.11.0` resolves to manifest digest `sha256:a223aee17aef5d21c3b9f63436dd19d27d1c665ec8b2f40011c9546cabae2a80`. It reports PyTorch `2.11.0+rocm7.14.0`, HIP `7.14.60850`, and exposes the RX 7900 GRE (`gfx1100`). The shared Modly runtime has completed both eager ROCm and accepted Torch-MIGraphX inference on the target. Ticket 02 runtime acceptance passed; the full semantic 3D pipeline remains in progress through Tickets 03–13.
- `scripts/modly-amd-runtime.sh` now forces XDG runtime state to the project-owned runroot and enforces mode 0700. `bash -n scripts/modly-amd-runtime.sh` passes, and `scripts/modly-amd-runtime.sh status` lists the five expected AMD images under sandbox escalation even when an unrelated `XDG_RUNTIME_DIR` is inherited. The sandbox still denies `newuidmap` capability setup for an ordinary invocation; use the managed elevated context for target containers. No system binaries or custom install were changed.
- The project-owned AMD runtime image is `localhost/modly-amd-migraphx:ticket02`, image ID `sha256:c96b56796c753d749cb02b31b88da7567c3712861d214bfe26452d21bd7e6b8d`, manifest digest `sha256:64fb89f2167f56a7381529e8e4fb69b322c3685f9805ea301263477c25aa0554`. It exposes the RX 7900 GRE (`gfx1100`) with PyTorch `2.11.0+rocm7.14.0`, HIP/ROCm `7.14.60850`, MIGraphX `2.16.0.dev+20250912-17-575-g4bcfe75b2`, Torch-MIGraphX `1.2`, and Python `3.12.3`. The accepted FP16 dense MLP ran 31 timed repetitions after warm-up through `torch.compile(backend="migraphx")`: eager ROCm 3.7041 ms, MIGraphX 1.1811 ms, numerical tolerance `atol=rtol=0.005`, peak allocated VRAM 232,820,736 bytes. Three repeated runs selected MIGraphX at 1.1783–1.1944 ms against eager 3.6856–3.7220 ms. A separately compiled candidate was rejected and routed through PyTorch ROCm; its eager output identities matched. Two sequential FP16 stages completed and returned allocator usage to the same 33,554,432-byte allocated/reserved baseline after each release. Python and OS package inventories, MIGraphX shared-library hashes, resolved `ldd` dependencies, and an explicit no-NVIDIA result are in `api/runtime/amd/evidence`. No custom/global ROCm or Torch installation was used. The runtime image and installed set are identified; the source build is not byte-for-byte reproducible because transitive AMD wheel/archive and OS repository content hashes are not fully pinned. That limit is disclosed in `api/runtime/amd/LOCK.md`.
- A project-owned Python 3.12 API test environment lives in `.modly-amd-runtime/api-test-venv`, populated from `api/requirements.txt`; no user's Python/ROCm installation was used. The complete API suite passes **148/148**, including the image-generation ASGI regression and the actual worker restart test. Node tests pass **20/20** (8 TypeScript, 12 JavaScript), the separate CLI suite passes **34/34**, and `npm run build` plus `npm run lint` pass. The build reports the optional `sharp@0.32.6` install script blocked, while the application build succeeds; Browserslist metadata is stale. The mesh picker has a separate Electron IPC/preload path; its focused suite passes 11/11 and esbuild checks pass for main IPC, preload, and workflow node. API OpenAPI generation includes `/process-runs` start/status/cancel and `/workflow-runs/from-mesh`. The built-in Structured Asset import source and production copy have matching tree digests: `sha256:f9974a500e80fbaad23a269b3ba2156a03340d9373232c87bdfac00f2119e680`.
- The `/workflow-runs/from-mesh` endpoint is async and delegates to an async subprocess supervisor using the same JSONL process-extension protocol. An actual ASGI HTTP request through `httpx.AsyncClient` executes the host-pinned importer and returns a valid sidecar response. The legacy image-to-mesh ASGI regression also passes in Python 3.12. The host Python 3.14 venv does not resume a successfully completed `run_in_executor` future for the legacy generation path; acceptance runs use the project-owned Python 3.12 environment instead of changing production semantics.
- Ticket 04 integration exposed that a qualified MIGraphX callable was retained privately but not available through a public inference-routing seam. `AMDInferenceRuntime.execution_callable()` now routes subsequent calls to the qualified MIGraphX module or the actual ROCm/authorized-CPU module, rejects stale or mismatched reports, and clears a previous compiled entry before a new qualification. Focused AMD runtime tests pass 15/15 in the project Python 3.12 environment.
- `npm ci --ignore-scripts --no-audit --no-fund` restored locked Node dependencies to ignored `node_modules`; package manifests and the lockfile were not changed. Electron UI/installer runtime exercise remains unverified.
- `npm run lint` initially tried to traverse protected files inside `.modly-amd-runtime/storage`, so the project-local generated runtime store is now included in `eslint.config.mjs` ignores. Lint then passed. `npm run build` also passed; the optional `sharp@0.32.6` install script remains blocked and Browserslist reports stale local metadata.
- Codacy MCP analysis tools are not available in this session. The repository instruction requiring Codacy analysis after edits cannot currently be satisfied; source-level checks will be recorded separately.

## Ticket status

| Ticket | State | Acceptance evidence |
| --- | --- | --- |
| 01 | Acceptance passed | All ten audited criteria are met by the Structured Asset schema/importer, topology-bound mappings, provenance and capability contracts, host-owned importer pin, pre-dispatch validation, and compatible image-generation route. The real canonical `/workflow-runs/from-image` ASGI request emitted a valid triangle GLB; run state and artifact bytes were verified. The full API suite passes 148/148 in the project Python 3.12 container; Node 20/20, CLI 34/34, lint, and production build also pass. |
| 02 | Acceptance passed | All ten criteria passed on RX 7900 GRE/gfx1100. Evidence includes three repeated FP16 MIGraphX wins, an explicit measured rejection/fallback, per-region version/native hashes and latency/VRAM telemetry, CPU policy tests, project-owned runtime and package inventories, sequential release plateau, bounded diagnostics, real worker reaping/restart, and no NVIDIA/CUDA dependency. See the audited issue and `api/runtime/amd/evidence`. The runtime is identified by immutable image digest with package and native inventories; transitive build/download sources are not yet byte-for-byte reproducible and remain disclosed in `LOCK.md`. |
| 03 | Acceptance passed; Hunyuan3D-2 Mini/Turbo selected, TripoSR rejected | Frozen rubric declared before comparison; Hunyuan3D and TripoSR were screened on RX 7900 GRE through Modly. Hunyuan's early failed chair attempts included two no-output runtime failures, one 16,126,386,176-byte VRAM stop above the 14 GiB ceiling, and one process timeout that cut generation off at 105.13 seconds before the 180-second gate. The bounded MIGraphX/ROCm comparison for the identical Hunyuan denoiser region measured MIGraphX 692.82 ms vs ROCm 418.04 ms; compilation was rejected as slower, so the same source/weights/device/module/profile/512px tensor shapes used explicit PyTorch ROCm for the final fixture series. The three successful source-pinned CPU-offload runs completed through the Modly process extension, with valid GLB + Structured Asset, per-image source/model provenance, and board/modeled memory under 14 GiB. Chair: model load 35,352.5 ms, setup 86,696.7 ms, generation 99,366.9 ms, adapter peak 2,609,977,344 B, sampled board peak 6,537,334,784 B, 428,137 vertices/856,288 faces, IoU 0.7786803059833489, GLB `sha256:412e3481a3d4137718f4aac5790d2fb0cd330b42877c0e6548805502925c6520`. Flamingo: model load 94,160.8 ms, setup 120,906.4 ms, generation 99,116.1 ms, adapter peak 2,609,977,344 B, sampled board peak 6,494,683,136 B, 268,890 vertices/537,860 faces, IoU 0.6190061584200467, GLB `sha256:b89cd8c9fd135d49f50d5f2c80a681ab94daea2838ddb01cad4095195b297b03`. Teapot: model load 33,057.2 ms, setup 58,134.3 ms, generation 98,638.9 ms, adapter peak 2,609,977,344 B, sampled board peak 6,502,207,488 B, 496,288 vertices/992,628 faces, IoU 0.9564278353668578, GLB `sha256:05f8180e9cd68c5557c870f3dafd933e4e0f99b7f24652aa9e5fcd5b535d9c44`. Hunyuan median IoU is 0.7786803059833489 and minimum 0.6190061584200467, meeting frozen thresholds (median >=0.50 and each >=0.35). TripoSR completed its screen but failed its median/per-image minima (0.33790/0.18581/0.59528; median 0.33790) and its MIGraphX parity gate, so it is ineligible. Focused Ticket03 tests pass 10/10, including pinned provenance, generic capability contract, offload hook routing, and failure cleanup preserving a pre-existing sidecar with no GLB/sidecar/staging residue. Full evidence per fixture is under `api/runtime/adapters/geometry/evidence/hunyuan-offload-{chair-run2,flamingo,teapot}/`; selection order makes Hunyuan the sole passing candidate. |
| 04 | Revalidation required; GeoSAM2 selected; explicit PyTorch ROCm route retained after MIGraphX parity failure | Prior frozen-fixture runs passed at macro-IoU 1.0, full coverage, zero overlap, and 14.41792 GB reserved; however, that sentinel failure led to a separately locked Modly-side completion policy. Two current-code runs on the 1,536-face fixture both completed, with 46/1,536 face memberships different after matching regions by their face sets (97.0% agreement; region IDs alone differed on 768 labels). On the larger Flamingo mesh, two matched same-code runs completed but differ on 875/46,180 final face assignments (98.1052% matched agreement); two further attempts on that mesh failed before labels. Two attempts on a distinct 83,732-face chair both failed before final labels, with different proposal outcomes. These are descriptive repeat and completion outcomes, not quality scores or a typical cross-shape estimate. Repeatability, quality, completion-branch target coverage, and current resource gates remain open; no truth was scored. Earlier runs remain historical evidence only. See the Ticket 04 issue and the 2026-09-27 repeatability panel.
| 06 | Acceptance passed | MaterialSeg3D was screened against its published CUDA/native requirements and is not AMD/resource-qualified; the CPU projection/back-projection adapter passed its 1.0 IoU/coverage/boundary fixture with independent material regions, full mask/projection provenance, unknown-face handling, and topology invalidation. An additional process fixture proves each of two parts contains both materials and both material regions span both parts. Focused Ticket06 tests pass 7/7. It consumes masks from a replaceable upstream segmenter; raw-image mask inference remains an explicit end-to-end dependency. |
| 07 | In progress, acceptance blocked | Ticket07 retains the frozen quality/coverage gates and independently routed DMS46/SigLIP2 candidates. Parent reported the corrected pinned-image suite at 10/10 in 18.687 seconds: actual SigLIP2 process-extension inference over only one generated 4×2 solid-color observation and two topology-masked crops, with `--network=none`, `--cpus=2`, no device mounts, read-only source/model/wheel-overlay mounts, and offline Hugging Face flags. The pinned processor returned `pixel_values [1,3,224,224]` and `input_ids [5,64]`; tests assert the five frozen primary class prompts plus two auxiliary metal subtype prompts, raw logits, ambiguous/uncalibrated output, topology-bound crop/prompt/projection/mask digests, CPU-only telemetry, and PBR preservation. `/preflight` must precede the API test venv in `PYTHONPATH` for hash-locked Transformers 4.50.0/huggingface-hub 0.28.1 to override the image's incompatible hub 2.0.0. Separately, project-owned NumPy/stdlib CPU renderer `fixtures/render_fixture.py` generated a 145-object, 580-region synthetic holdout bundle under ignored `.modly-amd-runtime/material-identity-fixture-v1/`: 440 heldout (80/class ×5, 20 unknown, 20 ambiguous) and 140 development (20/class ×5, 20 unknown, 20 ambiguous), with five-object-per-cohort development/heldout splits for unknown/ambiguous, 4 views/object, and no class/recipe/split leakage into `inputs.json`. All-region selective-coverage denominator is 440, including unknown/ambiguous; >=80% requires at least 352 accepted single-label outputs. Fixture checks pass 6/6 in 50.469 seconds, including two fully regenerated bundles with identical hashes, object-disjoint splits, pixel-exact mask/face-map agreement, calibration cohort counts, truth isolation, and file/digest/manifest checks. Parent independently verified all 1,887 indexed data-file sizes/digests and manifest sidecar. Manifest SHA-256: `c7c5d9c2ab31d3d956411c019e2ac19a72f991da5829ed18ceacf36e47094466`; inputs: `453ac070aadd84eaf45eb381fb2910e906026b47cc929a5e6fdc4d6dd5f29694`; truth: `8ca5699c1f3f7d96c2d28b024df6a67db0d9a6d725fbbf2e2f6f604bede6e6e8`. The new disjoint evaluator `api/runtime/adapters/material-identity/evaluator.py` loaded pinned SigLIP2 once in a networkless CPU container, verified all 1,887 fixture-indexed file hashes/sizes, reconstructed 580 topology-bound crops, saved truth-free raw logits before opening truth, calibrated on 140 development crops only, froze policy SHA-256 `1f98fb9d46a440b01ab3479ca3c4bda92827fc624858cf6059829ba247b0cd0c`, then scored all 440 heldout crops. Development searched 19,881 threshold pairs and found none meeting every frozen gate. Heldout macro-F1 is 0.224914 (>=0.85 required), minimum recall 0.0 (>=0.80), all-region coverage 314/440=0.713636 (>=352/440 required), unknown abstention 0.10 (>=0.90), ambiguous abstention 0.45 (>=0.90); the coverage/OOD conjunction fails. Clear plastic, metal, and rubber/latex have zero true positives. CPU telemetry: PyTorch `2.11.0+rocm7.14.0`, HIP `7.14.60850` visible in the build but CPU device selected, CUDA false, no device mounts, processor/model loaded once, inference 350,695 ms, peak host RSS 2,037,297,152 B. This is a synthetic CPU quality failure, not AMD acceptance or real-world quality. Raw predictions SHA-256 `54776b8efc0498f44c3e2145de6d120773a5248d796989286ef9b0007a1df85e`; heldout report SHA-256 `f86e20a6c6512e024d7ecefc80e200a9116576e05e7324866587fd7238744474`; files under `api/runtime/adapters/material-identity/evidence/siglip2-rendered-cpu-v1/`. Evaluator deterministic tests pass 5/5. A follow-up CPU-only development prompt study tested frozen v2 plus three exact close-up variants and mean/max aggregations over only the 140 development crops/35 objects. All six candidates searched 19,881 threshold pairs each and had zero feasible pairs; minimum class recall was 0 for all. Best macro-F1 was 0.3084 with 0.75 all-region coverage, 0.15 unknown abstention, and 0.45 ambiguous abstention; a high-coverage surface prompt had only 0.3020 macro-F1, 0.0 unknown abstention, and 0.05 ambiguous abstention. Per the frozen rule the candidate is rejected before heldout; no heldout logits were opened/scored or truth labels consulted for selection, and the earlier heldout report was not touched. A new FMMC CVPR 2026 lead was screened, but its public page metrics do not match frozen gates and its referenced checkpoint lacks an immutable identity/hash and clear terms; no weights were fetched or loaded. Study evidence including the exact prompt text/aggregation definitions is in `api/runtime/adapters/material-identity/evidence/siglip2-development-prompt-study-v1/`. Study tests plus evaluator tests pass 7/7. The frozen gates and all-region 440 denominator are documented in `SELECTION_AND_GATES.md`; no threshold or gate changed. Official Meta standard DINOv2 ViT-B/14 (Apache-2.0, pinned source/checkpoint) was evaluated CPU-only after project-context preflight: all 580 truth-free embeddings were durably saved and 140 development crops were scored with five-fold object-disjoint OOF prototypes. Development gates failed (macro-F1 0.6652343 <0.85, minimum recall 0.25 <0.80, coverage 0.8357143 passes, unknown abstention 0.0 <0.90, ambiguous abstention 0.5 <0.90; feasible pairs 0/19,881); evaluator stopped before opening truth, and no heldout results or policy were created. Embedding/OOF/report SHA-256: `97039cbb28301ffb3d5c9582d4fc380c98572be35e70c56d00732781c5f7a086` / `f89e268ab1ff71778dddfad54eff0a84d9d8d9de0199e274db695c78c38e9d1c` / `744fd6bb479d471011043deb18b900fb21ee19b99898b27deecf1e73c87bb3b0`; details and exact command are in `api/runtime/adapters/material-identity/evidence/DINOV2_LICENSE_PROVENANCE_REVIEW_2026-09-25.md`. No acceptance gate changed. One separately predeclared fixed DINOv2 dual-ridge head (`lambda=1.0`, object-mean train rows, five object-disjoint folds, no parameter sweep) was evaluated using only the durable embeddings and development recipe labels. Gates also failed: macro-F1 0.6668177, minimum recall 0.20, all-region coverage 0.7214286, unknown abstention 0.20, ambiguous abstention 0.80, feasible pairs 0/19,881. It stopped without opening heldout truth. Ridge OOF/report SHA-256: `afef82d434d39a1759aefbbdda14af08ae16b20ec194a615188056762a78ac57` / `290bbe720e947120e8926912a3a27ce3e393a161cbc25ce01f6658fda64baac5`; details in `api/runtime/adapters/material-identity/evidence/DINOV2_RIDGE_DEV_SCREEN_2026-09-25.md`. DMS46 remains unexecuted: its pinned license requires an ACKNOWLEDGEMENTS file that returns 404 at both the pinned and checked current Apple source paths; the staged checkpoint archive contains no notice files. Evidence: `api/runtime/adapters/material-identity/evidence/DMS46_LICENSE_REVIEW_2026-09-25.md`. No RX 7900 GRE MIGraphX/ROCm parity, latency, VRAM, or target-resource evidence exists. Details: fixture/evaluator evidence and Ticket07 issue. DINOv2 direct embeddings and the fixed ridge head both failed frozen development gates and stopped before heldout truth. Source-only MINC screening found no weight license or immutable archive identity; RMSNet screening found MIT code and CC BY 4.0 dataset terms but no weight terms or checkpoint hashes, plus a 20-class road-scene domain mismatch. Neither is cleared as a deployment candidate. Evidence: `api/runtime/adapters/material-identity/evidence/MINC_MODEL_SCREEN_2026-09-25.md` and `api/runtime/adapters/material-identity/evidence/RMSNET_MODEL_SCREEN_2026-09-25.md`. No acceptance gates changed. Separate frozen-encoder candidate: pinned SigLIP2 `get_image_features` emitted normalized 768-D vectors for all 580 truth-free fixture views, durably saved before dev label derivation. A single fixed dual-ridge head (`lambda=1.0`) was OOF-scored on 140 development crops with five object-disjoint folds; gates failed (macro-F1 0.6797408, min recall 0.45, coverage 0.7285714, unknown abstention 0.40, ambiguous abstention 0.40, 0/19,881 feasible pairs; 102/140 accepted). It stopped before opening heldout truth; no policy/results for heldout. Embeddings/OOF/report SHA-256: `c0e40370aeb0c2b57cb67c561a9fae8e83297597be0789a63551b3d5ce9861bf` / `0817c6b959858413a86dc62a9632ade557cfccebe904f2212d080f0b6bdbf39b` / `e58f652c3fee55408c04da8294ad3a9718cd75c76352f5097a36fe0a7acb7b2f`; CPU telemetry and exact Podman command: `api/runtime/adapters/material-identity/evidence/SIGLIP2_IMAGE_EMBEDDING_RIDGE_DEV_SCREEN_2026-09-25.md`. Focused regression tests pass 2/2. No AMD target acceptance. Source-only current open VLM screen: Qwen3-VL-2B-Instruct is a research lead, not cleared. HF marks Apache-2.0 and model.safetensors is 4.26 GB (SHA-256 `7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0`), but inspected Hub page exposes only abbreviated commit `78448d7` and no complete snapshot manifest. Model requires Transformers >=4.57; AMD documents Radeon Qwen3-VL-8B on a different Radeon platform, with no RX 7900 GRE proof for 2B. It is generative with no native calibrated class/OOD scores; a fixed grammar and token-score adapter would first require validation. No downloads, execution, fixture truth, or heldout access. ROCm Linux lists RX 7900 GRE support, establishing a generic backend route but not model-specific parity, memory, latency, or quality acceptance. Details: `api/runtime/adapters/material-identity/evidence/QWEN3_VL_SMALL_OPEN_MODEL_SCREEN_2026-09-25.md` and `TICKET07_IMMUTABLE_MODEL_AND_RX7900GRE_ROUTE_SCREEN_2026-09-25.md`. No candidate cleared; gates unchanged. A new fixed image-appearance descriptor plus dual-ridge candidate also failed dev gates (macro-F1 0.5897, min recall 0.30, all-region coverage 0.5143, unknown abstention 0.60, ambiguous abstention 0.90; 0/19,881 feasible pairs). All 580 features were durably saved before dev-label use; heldout truth stayed unopened. Its focused tests ran under system Python 3.14.7 because Pillow is absent from the project API test venv, so this remains candidate research evidence. Detailed evidence: `PHYSICS_APPEARANCE_RIDGE_DEV_SCREEN_2026-09-25.md`. Current project-owned Python 3.12 Ticket07 suite passes 37 tests with 6 skips (one model-assets skip plus five optional-Pillow research-candidate skips); no model acceptance is implied. A mask-aware v2 descriptor excluded neutral padding but still failed dev gates: macro-F1 0.6261, min recall 0.40, all-region coverage 0.6214, unknown and ambiguous abstention 0.45 each, 0/19,881 feasible pairs. All 580 features were durably stored before dev labels; heldout truth remained unopened. Two candidate tests passed in system Python; optional Pillow skips in the project venv. Evidence: `PHYSICS_REGION_RIDGE_DEV_SCREEN_2026-09-25.md`. |
| 08 | Implementation advanced; acceptance blocked | Project-owned fixed-geometry inverse rendering is registered with the Modly process contract. A preregistered CPU dev sweep's lambda=1 candidate measured base-color MAE 0.02868, SSIM 0.86246, roughness MAE 0.02343, metallic MAE 0.00871, and novel-light MAE 0.00333 over all 2,408 visible development texels. This does not establish held-out acceptance or RX 7900 GRE VRAM qualification; upstream material segmentation and other Ticket08 gates remain open. See `ticket08-region-inverse-v2-albedo-prior-sweep-results-2026-10-01.md`.
| 05 | In progress; Florence development candidate rejected; Decider synthetic adapter integration passed | Frozen role contract, candidate-independent semantic attachment, truth-separated scorer, and rendered fixture are verified. Florence failed the frozen development gates. The user-supplied Decider GGUF now has a pinned llama.cpp/HIP batch harness and a successful network-disabled synthetic call through its registered local adapter on RX 7900 GRE; ten raw choice scores and four evidence refs passed the existing semantic response parser, with temporary inputs removed. The 16 M-RoPE recurrent-position warnings remain unresolved, upstream PyTorch/GGUF parity is unproven, and no Ticket 05 fixture scoring or heldout access occurred. Semantic quality and full workflow acceptance remain open. Evidence: `api/runtime/adapters/parts/evidence/TICKET05_DECIDER_ADAPTER_LIVE_SYNTHETIC_2026-09-26.md`. |
| 09 | Implementation advanced; acceptance blocked | Fusion preserves stale topology claims separately, correction writes use digest compare-and-swap, conflicting active corrections produce explicit unresolved conflict, and agreeing corrections resolve deterministically with all IDs retained. Focused fusion suite passes 12/12 CPU tests. Final acceptance still depends on upstream capabilities and full workflow integration.
| 10 | Partial implementation; final acceptance blocked | API import-stage caching and geometry-bound targeted reruns are implemented. Electron caching now fails closed outside the built-in no-weight validator capability and includes path, validation, output-shape, and atomic-write checks. Six cache plus four workflow-store tests pass. Restart/reuse for every major stage, semantic-stage caching, and exact targeted-descendant invalidation remain open.
| 11 | Implementation present; final acceptance blocked | GLB/glTF export, structured sidecars, compatibility reporting, and independent validation are implemented. A populated CPU round-trip through the canonical handler preserves corrections, mappings, provenance, supported/unknown PBR, basis, and material slots; deterministic and negative checks pass. Ticket11 suite passes 11/11. Real-stage compatibility telemetry, viewer color-space behavior, target hardware, and accepted-upstream integration remain open.
| 12 | Acceptance blocked | RX 7900 GRE full headless POC gate remains pending. The latest SUV attempt stopped before proposals under the safe per-process VRAM ceiling; source-level memory review is in progress. |
| 13 | Interactive editor implemented; runtime acceptance open | Viewer supports selecting regions/faces, boundary-handle drag with swept-face preview, confirm/cancel, correction history, property edits, and targeted-rerun preview. Preview validates topology/revision and viewer face-count agreement, highlights exact moved faces, and rejects invalid membership. The 75k-face path uses spatial bins, bounded sampling/history, and throttled previews; focused helper tests pass 3/3. This edits face ownership along a surface rather than moving mesh vertices. SUV interaction, accessibility/usability review, and full vertical-slice verification remain open.

### 2026-09-26 continuation evidence

- Ticket 05's Decider integration now has a weight-free installed-entry-point regression test. The combined focused suites pass 10/10 using the project-owned Python 3.12 venv; test temporary files are routed to `.modly-amd-runtime/test-tmp`. The live synthetic adapter call and M-RoPE warning evidence remain runtime-only; no candidate quality score or ticket acceptance is claimed.
- Ticket 05's paired answer-slot gate was re-audited. The custom probe reads the intended final answer slot, but hidden-state correctness cannot be established from the persistent native smoke. The workdrive has the GGUF/projector pair but no upstream PyTorch checkpoint or matching processor/tokenizer for a paired synthetic comparison. No model or fixture input was acquired/accessed for that audit. Evidence: `api/runtime/adapters/parts/evidence/TICKET05_DECIDER_ANSWER_SLOT_PARITY_GATE_2026-09-26.md`; parity and semantic evaluation remain open.
- Ticket 07's source-only Decider/adjacent-route screen found no new eligible inference candidate. Decider is excluded from Ticket07 scoring due prior model use outside its ticket-specific preregistration, partial rights lineage, and unresolved answer-slot alignment; DMS46 remains unloaded pending its license/acknowledgement terms. No Ticket07 weights or fixtures were accessed and gates did not change. Evidence: `api/runtime/adapters/material-identity/evidence/TICKET07_DECIDER_AND_ADJACENT_ROUTE_SCREEN_2026-09-26.md`.
- Ticket 08's independent source screen found Marigold IID Appearance v1.1, which advertises the required albedo/roughness/metallicity channels, but it is not eligible for acquisition or evaluation because its model-use restrictions are placeholder text, its linked base-model license is inaccessible in reviewed sources, and the reviewed listing lacks a complete immutable revision/hash manifest. No model or fixture was accessed and thresholds were unchanged. Evidence: `api/runtime/adapters/pbr/evidence/marigold-iid-appearance-v1-1-source-gate-2026-09-26.md`.
- Ticket 07's new official-source screen evaluated MatSpectNet (WACV 2025), a relevant dense RGB material segmenter. It is ineligible before acquisition: author sources provide an unpinned Google Drive checkpoint without checkpoint terms or an immutable digest, no repository license surfaced, no frozen unknown/ambiguous or generic-metal abstention behavior is evidenced, and the documented Torch 1.12.1/faiss-gpu/eight-RTX-3090 setup provides no AMD path. No code, weights, dataset, fixture, or truth was acquired/accessed. Evidence: `api/runtime/adapters/material-identity/evidence/MATSPECTNET_OFFICIAL_SOURCE_SCREEN_2026-09-26.md` (SHA-256 `a22f36293cd34dac753965ef1b6a003fef33b52575897e2da7d6f030593b1fcb`). Ticket 07 remains blocked; gates unchanged.

## Ticket 03 rubric consistency and focused revalidation (2026-09-28)

- Reconciled the stale `api/runtime/adapters/geometry/SELECTION_RUBRIC.md` probe-status paragraph with the already recorded target evidence in the Ticket03 issue and status row. Frozen thresholds and candidate ordering were unchanged: Hunyuan passed with median IoU 0.778680 (minimum 0.619006), all three runs under 14 GiB board VRAM and 180 seconds generation; TripoSR failed the frozen image-quality and MIGraphX parity gates.
- Repeated the focused Ticket03 suite without installing packages, adding the existing project-local Pillow cache to `PYTHONPATH`: `TMPDIR="$PWD/.modly-amd-runtime/tmp" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/.modly-amd-runtime/package-cache/north-micro-vision-v1/site-packages:api" .modly-amd-runtime/api-test-venv/bin/python -m unittest api.tests.test_ticket03_geometry -q` — **10/10 passed**.

## Issues and next actions

## Product implementation continuation (2026-09-30)

- Ticket 09 provisional evidence fusion and correction persistence are now implemented in `api/services/structured_asset_fusion.py` and `api/routers/structured_assets.py`. Conflict/unknown/resolved claims retain their evidence without cross-model confidence ranking; exact topology-valid user corrections take precedence while preserving displaced model assertions; topology or segmentation changes orphan or mark corrections pending-remap rather than transferring them by guess. Focused CPU suite: **8/8 passed** in `api/tests/test_ticket09_fusion.py`. The route provides fusion inspection and compare-and-swap correction writes. Acceptance remains open and requires workflow-stage integration plus its upstream tickets.
- Ticket 10 provisional content-addressed caching is integrated into `/workflow-runs/from-mesh` in `api/routers/structured_workflow_runs.py`, backed by `api/services/structured_stage_cache.py`. Cache identities include input/topology, pinned adapter, schema, runtime, and parameters; reads verify digests and current-domain identity; writes are locked, atomic, and durable; requested import reruns invalidate the imported stage and its bundled no-op dependent. The route refuses unsupported child-only reruns. Focused CPU suite: **13/13 passed**. Later semantic-stage cache/rerun integration remains required.
- Ticket 11 provisional GLB/glTF exporter and validation are implemented in `api/services/structured_asset_export.py` and registered in `api/main.py` through `api/routers/structured_asset_export.py`. Source topology/accessor ordering is preserved, conversion requires an explicit matrix when needed, unsupported PBR channels are not invented, compatibility reporting retains unknown telemetry, and writes reject path escapes/overwrites. Focused CPU suite: **8/8 passed**. A shared validator defect for historical glTF StageArtifact references with external buffers was fixed: references now use full glTF geometry identity, including local dependencies. Its regression plus all Ticket 09–11 focused suites pass **30/30** together. OpenAPI confirms `/structured-assets/fuse`, `/structured-assets/corrections`, `/structured-assets/export`, and `/workflow-runs/from-mesh` are registered.
- Ticket 13 has a first integrated review surface in Modly's existing viewer. Workflow finalization now carries the resulting Structured Asset sidecar path onto the current mesh job; the panel reads and displays asset/topology state, region mappings, evidence kind, confidence, and provenance, and posts a user correction against the selected valid topology-bound region using sidecar digest compare-and-swap. Corrections are explicitly described as per-asset edits, not model training. `npm run build` and `npm run lint` both pass with a temporary official Node 26.10.0 binary. The system `/usr/bin/node` still fails to launch because its package requires missing `libsimdjson.so.33`; no global packages were changed. Seam/face picking, visible region highlighting, before/after seam preview, undo/redo, invalidation preview, targeted semantic-stage reruns, and runtime telemetry UI remain open.
- The reported shared-GPU issue remains unresolved and `.modly-amd-runtime/GPU_RUNS_PAUSED` remains in force. No GPU or model execution was started in this continuation. Ticket 04 reproducibility/quality and RX 7900 GRE resource gates remain open. The API test run was CPU-only; three older importer integration checks previously failed with `REFERENCE_ADAPTER_PIN_MISMATCH`, while the focused Ticket 09–11 suite passes.
- Current next work: continue Ticket 04 code-level diagnosis without target execution; finish Tickets 07 and 08 through project-owned candidates; extend Ticket 13 to the required surface seam interaction; integrate later semantic stages into Ticket 10 dispatch/cache, and validate exporter artifacts through the shared importer. Do not change acceptance gates or mark Tickets 09–13 accepted from implementation tests.

- Ticket 01 acceptance criteria are all passed. Workspace-safe mesh staging covers standalone glTF dependencies; the pinned importer executes from a verified private snapshot; and legacy image-to-mesh generation passes as an ASGI request with a real temporary Modly extension in Python 3.12. The full-spec saved multi-node graph requirement is tracked for the later end-to-end semantic workflow gate, not added to Ticket 01's audited acceptance list. Share the existing editor graph planner/execution semantics behind host interfaces when that stage is dependency-ready.
- Tickets 01, 02, 03, and 06 meet their audited acceptance criteria. Tickets 07 and 08 are dependency-ready and are being advanced in parallel; Ticket 05 still waits for 04. Preserve the documented upstream 2D-mask dependency and close it in the full headless acceptance workflow.
- New source-only candidate-frontier screens for Tickets 04 and 07 found no eligible alternate; reports are linked from their issue records. Ticket 08's fixture correspondence sidecar integration is complete and retained at `api/runtime/adapters/pbr/evidence/ticket08-three-region-pbr-correspondence-v1.npz` (SHA-256 `cfbfc146f65451137da58887ef969ec3c06b0fef4c893be29486cfd392933474`, topology revision `sha256:77922db4079e22f05d6649baa6483218559ca8ccb18ab16c9d51b32ff03315c2`). Its focused suite passes 21/21 with the isolated optional-SciPy overlay. These advances do not pass Tickets 04, 07, or 08; all frozen gates remain in force. The PartField model remains held until the owner answers the use-scope question recorded during candidate screening.
- Acquire or expose RX 7900 GRE hardware/runtime and complete Ticket 12's headless POC gate before claiming project acceptance.
- Restore writable Git metadata and provide the Codacy analyzer MCP integration if those validation/evidence channels are required for the final handoff.

## Ticket 04 mask-stage diagnostic update (2026-09-27)

- A second same-teapot, same-settings, same-seed opt-in prompt-seed-lift run (`a4921308-3de2-4337-861b-0a8948a64c73`) again produced 104,454 sentinel `999` labels and no final Structured Asset. Its payload-free audit saw 108 propagated object/frame records across 12 views. Each float mask had 1,048,576 nonzero entries but zero positive entries; shrink, stability, and area stages retained all 108 records, while bool conversion produced all-false masks. IoU accumulation was empty and lifting received empty masks. This localizes failure to the predictor output entering propagation, before those later filters; the audit does not establish why the logits have no positive foreground or whether input prompts registered correctly. The accepted proposal count changed from 11 in run `0340842b-df11-4f94-b43b-9a85902fa92c` to 9, and neither run produced a quality score or VRAM peak.
- Added an all-unassigned fail-fast check before GeoSAM2 completion; its regression test passes and ensures the expensive completion routine is not called when every face is `-1`/`999`. It does not fix or alter segmentation output.
- The focused Ticket 04 diagnostic/seed-lift suite passes 21/21. The complete completion-artifact module passes 2/2 in the network-disabled project AMD image (`localhost/modly-amd-migraphx:ticket02`); this exercised the actual hash-pinned upstream completion routine but did not use the GPU. The new guard test also passed 1/1 in the host API venv.
- Exact completion-artifact command: `XDG_RUNTIME_DIR="$PWD/.modly-amd-runtime/run" podman --root "$PWD/.modly-amd-runtime/storage" --runroot "$PWD/.modly-amd-runtime/run" run --rm --network=none --read-only --userns=host --volume "$PWD/api:/modly/api:ro" --volume "$PWD/.modly-amd-runtime/source:/modly/.modly-amd-runtime/source:ro" --volume "$PWD/.modly-amd-runtime/tmp:/tmp:rw" --env PYTHONPATH=/modly:/opt/rocm/lib --env PYTHONDONTWRITEBYTECODE=1 localhost/modly-amd-migraphx:ticket02 python -m unittest api.tests.test_ticket04_geosam2_completion_artifacts -v`.
- Different generated assets may have different face counts and are mapped against their own topology revisions. No cross-asset face-index comparison is made. Same-input repeat checks remain on byte-identical mesh and inference inputs, and current same-mesh failures still block Ticket 04 acceptance.
- The topology interpretation regression was rerun and passes 1/1: `TMPDIR="$PWD/.modly-amd-runtime/tmp" PYTHONDONTWRITEBYTECODE=1 .modly-amd-runtime/api-test-venv/bin/python -c 'import typing_extensions, sys, unittest; sys.path.insert(0, "api"); unittest.main(module="api.tests.test_ticket04_different_topology_mappings", argv=["test_ticket04_different_topology_mappings", "-v"])'`.
- Next diagnostic: capture whether automatic prompts are registered with the predictor, using payload-free point/label counts and coordinate/mask digests, alongside finite/min/max/positive counts for propagated logits. Repeat the exact teapot input only after this telemetry is locked and tested. Do not infer mask-registration cause from current data.
- Dependency-ready source screens: Ticket 07's MINC-SigLIP2 community checkpoint misses its published macro-F1 and glass/metal/plastic recall floors, omits required labels, and has no abstention contract; rejected before acquisition. Ticket 08's MaterialSeg3D requires albedo as input and assigns fixed roughness/metalness values by class, with no independently qualified AMD path or frozen quality evidence; rejected before acquisition. Reports are `api/runtime/adapters/material-identity/evidence/TICKET07_MINC_SIGLIP2_COMMUNITY_CHECKPOINT_SCREEN_2026-09-27.md` and `api/runtime/adapters/pbr/evidence/ticket08-materialseg3d-official-source-screen-2026-09-27.md`. No ticket gates changed.
- Further ready-frontier screens: Ticket 07's CLAMP depends on haptic input and has no qualified image-only region classifier, abstention, complete asset rights/identity, or AMD evidence; rejected before acquisition. Ticket 08's SVBRDF Uncertainty emits the requested map types in its public sample, but its pinned optimizer forces Mitsuba `cuda_ad_rgb`; the documented GPU route is NVIDIA and the alternative `llvm_ad_rgb` is CPU. It is not an RX 7900 GRE candidate. Reports: `api/runtime/adapters/material-identity/evidence/TICKET07_CLAMP_VISUOHAPTIC_SOURCE_SCREEN_2026-09-27.md` and `api/runtime/adapters/pbr/evidence/ticket08-svbrdf-uncertainty-source-screen-2026-09-27.md`. No weights, packages, fixtures, or truth were accessed; gates unchanged.

## Ticket 08 provenance refresh (2026-09-25)

A source-only SuperMat follow-up pinned an official SD 2.1 single-file checkpoint revision and its upstream LFS SHA-256, but this `.ckpt` is not the Diffusers component snapshot used by the candidate and the official full license terms remain inaccessible (HTTP 401). No artifact was acquired or loaded. SuperMat remains blocked pending exact component identities, accessible terms, and owner use-scope clearance; see `api/runtime/adapters/pbr/evidence/supermat-official-assets-license-refresh-2026-09-25.md`. This does not change Ticket 08 status or gates.

## Registered Ticket 08 candidate v2 result (2026-09-25)

The new fixed-geometry v2 candidate consumed the exact fixture correspondence sidecar. Its training-only run used 32x32 output, 25 function evaluations, minimum 3 samples/cell; it covered all 70,688 training pixels and measured forward RGB MAE 0.0729694. The source, input identities, and candidate output were frozen before a single quality scorer invocation. Visible coverage (1.0) and geometric-normal novel-light MAE (0.0395586) passed. Base-color MAE 0.133329, SSIM 0.516634, roughness MAE 0.403555, metallic MAE 0.326869, and conductor/dielectric bias 0.283427 failed their unchanged gates. It remains rejected and was not target-GPU tested or tuned. The report and immutable artifacts are in `api/runtime/adapters/pbr/evidence/ticket08-registered-fixed-geometry-v2-{training-report,quality-score}.json` and `ticket08-registered-fixed-geometry-v2-training-output.npz`. This candidate does not pass Ticket 08.

The complete Ticket08 suite now passes **25/25** in the project Python 3.12 environment with the isolated, checksum-verified SciPy overlay. This includes the registered v2/v3 deterministic, topology, unknown-output and training-closure tests plus the existing fixture, scorer and projection regressions. The registered-v2 and v3 quality failures are documented above and remain rejections; passing regression tests do not change their status.

Evidence-integrity correction for the v2 probe: an exploratory run parsed the broad fixture metadata JSON, which contains material names. That score report is preserved separately and excluded from acceptance. The same candidate and frozen parameters were rerun with the exact-key allowlisted training-scene JSON; its output digest was unchanged, and the valid separated scorer produced the rejection metrics above. No tuning followed the score.

Registered latent-normal Ticket08 v3 was frozen and scored separately. Training-only fit covered all 70,688 samples with forward RGB MAE 0.0591110 (81.0 s CPU); the single scorer run measured base-color MAE 0.161017, SSIM 0.425811, roughness MAE 0.251079, metallic MAE 0.201517, metallic bias 0.520005, and novel-light MAE 0.040401. It fails the fixed map gates and is rejected. No AMD target run followed. Exact evidence is in `api/runtime/adapters/pbr/evidence/ticket08-registered-latent-normal-v3-quality-score.json` and the adjacent training report/output.

## Additional dependency-ready candidate screens (2026-09-25)

- Ticket 04: SAMesh is a relevant multiview mesh-part segmentation approach that uses SAM2 masks and rendered face IDs to lift regions onto mesh faces. It is not ready for a target probe: inspected sources do not pin its code, establish its root-project license or exact checkpoint identities, or provide AMD or Modly-fixture evidence. The SAM2 optional CUDA extension can be disabled, but its official guide states that doing so removes mask postprocessing. Report: `api/runtime/adapters/parts/evidence/ticket04-samesh-alternate-screen-2026-09-25.md`. No gate changed.
- Ticket 07: SiPhy is an object-level material-reasoning lead, but the official workflow lacks pinned assets/complete use terms, requires an API key for its main material proposal path, and does not establish the region-level classifier, abstention or AMD contract. Report: `api/runtime/adapters/material-identity/evidence/SIPHY_OFFICIAL_CANDIDATE_SCREEN_2026-09-25.md`. No assets or fixture truth were accessed.
- Ticket 08: Materialist, MaterialFusion/StableMaterial, and Intrinsic Image Fusion remain ineligible because of topology-preservation, asset provenance/terms, CUDA/runtime, and resource gaps. A separate registered-input review found no defensible third estimator to freeze without truth leakage or unvalidated priors. Reports: `api/runtime/adapters/pbr/evidence/inverse-rendering-candidate-frontier-2026-09-25.md` and `registered-global-candidate-feasibility-screen-2026-09-25.md`. Ticket 08's full focused suite passes 25/25; no candidate passes its PBR gates.

### Qwen3-VL material-identity synthetic preflight (2026-09-25)

Qwen3-VL-2B-Instruct is now pinned at full revision `89644892e4d85e24eaac8bacfd4f463576704203`; all 12 staged snapshot files pass their locked SHA-256 and byte-size checks. A separate CPython 3.14 project overlay pins Transformers 4.57.1 and its dependencies by wheel hash. The synthetic CPU model preflight passed using `PYTHONNOUSERSITE=1`: the pinned processor and model scored one generated neutral-gray image twice, producing five finite next-token logits with exact repeated equality. A–E map to distinct single tokens 32–36 in evaluator class order. Prompt SHA-256: `96869f69dd8db709fb72da2ff79d4bba381ba44f2df15fd1161b43e98fb87835`; result JSON SHA-256: `acfc57427de698f7ee0c33ea4c74ac695da0efb4d5d8ca4f9711abd42f779d9b`. This did not read fixture or truth data. The run used system Python's PyTorch ROCm on CPU; a process-local shim only declares an otherwise-missing torchvision NMS schema for import and is not product inference or target-hardware evidence. An earlier processor-only exploration with the user site enabled is explicitly excluded from acceptance. Full details: `api/runtime/adapters/material-identity/evidence/QWEN3_VL_SYNTHETIC_MODEL_CPU_PREFLIGHT_2026-09-25.md`. Candidate may proceed to the unchanged development-only quality screen; Ticket 07 accuracy/coverage/abstention/AMD gates remain open.

### Qwen3-VL development result (2026-09-25)

The full truth-free evaluation completed all 580 crop views (one forward per crop, batch size 1) and durably committed raw logits before deriving development targets. The 140-row dev screen failed the frozen gates with no feasible threshold pairs: macro-F1 0.2982 (<0.85), minimum recall 0.05 (<0.80), all-region coverage 0.7357 (<0.80), unknown abstention 0.10 (<0.90), ambiguous abstention 0.50 (<0.90). The run stopped before held-out truth and created no held-out report or policy. Raw logits/manifest/dev-report SHA-256: `13a7dd98a9155e201b7c2fc66b9e26dee3d9540d2553682f97af78f187fe50b2` / `85e7d12eef0b448e11ded4963289d7274c1797ee27f51a0cdd61e1358258befb` / `ab45f108003f847b4cdab05507675c2b96c25b64730cee930b8c51ee9e2731ac`. CPU scoring time 918,461 ms (15m18s), 0.6315 crops/s. Qwen candidate is rejected under this frozen contract; no threshold/gate changed. Three dev-protocol tests pass. Full evidence: `api/runtime/adapters/material-identity/evidence/QWEN3_VL_DEVELOPMENT_EVALUATION_2026-09-25.md`.

An additional Ticket 07 source screen found HRIM2021 Material-Based Semantic Segmentation, but its authors state pretrained checkpoints cannot be shared due to confidentiality, and its taxonomy lacks required classes and calibrated OOD behavior. It is rejected before weight access. Evidence: `api/runtime/adapters/material-identity/evidence/HRIM2021_MATERIAL_SEGMENTER_SCREEN_2026-09-25.md`.

A second Qwen candidate used one fixed object-disjoint dual-ridge head over the already-saved five raw logits (normalized object-mean vectors; five folds; lambda 1.0; no parameter search). Its dev metrics also fail: macro-F1 0.4379, min recall 0.05, all-region coverage 0.9929, unknown abstention 0.00, ambiguous abstention 0.05. All 20 unknown and 19/20 ambiguous dev views received forced supported labels. The run notes elevated post-selection bias, no heldout truth access, and unchanged gates. OOF/report hashes: `57d5fea522282bd63ea838f2ad077ee02f08c647d6226a0ebd610424c4ea9bbe` / `b6d2b5d962fd0ba4af5f13433d618c32bffca7a518bf1e79e72c49366b31a0f3`. Evidence: `api/runtime/adapters/material-identity/evidence/QWEN3_FIXED_RIDGE_DEVELOPMENT_2026-09-25.md`.

Ticket 08's new primary-source candidate frontier found no independently license-cleared estimator ready for evaluation. LumiTex includes non-commercial components and lacks a complete immutable weight set/AMD route; MatLat weights are CC BY-NC; Material Palette lacks metallic and an available named upstream checkpoint; MatMart has no located official executable/checkpoint/license/runtime chain. No assets or fixture truth were accessed. Evidence: `api/runtime/adapters/pbr/evidence/license-amd-cleared-pbr-estimator-frontier-2026-09-25.md`. All ready candidates remain blocked; continue with Ticket 04, 07, and 08 only.

Ticket 04's follow-up official-source screen also found no candidate ready for target evaluation. MeshSegmenter remains README-only; CubePart generates new meshes and lacks cleared use scope, checkpoint hash, and AMD proof; SAM3D-Part is interactive generative extraction with an 80 GB CUDA reference setup and SAM-derived licensing. None establishes the frozen input-face mapping or >=0.90 macro-IoU gate. No weights were fetched or run. Evidence: `api/runtime/adapters/parts/evidence/ticket04-late-2026-part-model-frontier-screen-2026-09-25.md`.

### Restored project runtime hardware access (2026-09-25)

The restricted shell hides `/dev/kfd`/`/dev/dri`, but the project-owned `localhost/modly-amd-migraphx:ticket02` container can use the RX 7900 GRE through the managed host Podman path with the project-owned runtime directory. Verified inside the networkless container: PyTorch `2.11.0+rocm7.14.0`, HIP `7.14.60850`, ROCm reports `cuda=True`, device 0 is `AMD Radeon RX 7900 GRE`; device count is 2. The checked image ID is `c96b56796c753d749cb02b31b88da7567c3712861d214bfe26452d21bd7e6b8d`. Exact command/output: `api/runtime/amd/evidence/project-container-rx7900gre-access-2026-09-25.md`. This restores the path for gated AMD probes but does not pass any ticket's model, quality, resource, latency, or parity acceptance. No host packages or user-site custom installation were changed.

The owner has been asked whether PartField's noncommercial research/education use restriction applies to this project. No PartField checkpoint load or probe is authorized until that use-scope question is answered. Continue other eligible source/runtime work while it is pending.

### Follow-up candidate and integrity work (2026-09-25)

Ticket 07 source screening identified SmolVLM2-2.2B-Instruct at immutable revision `482adb537c021c86670beed01cd58990d01e72e4`. Its all-file asset lock and synthetic CPU/AMD preflights are complete. Candidate-specific scoring has now failed the unchanged development quality gates on RX 7900 GRE; see `api/runtime/adapters/material-identity/evidence/SMOLVLM2_DEVELOPMENT_EVALUATION_2026-09-25.md`. Ticket 07 remains blocked and the candidate is not accepted.

The Qwen synthetic preflight verifier now checks the pinned lock identity and all 12 staged file SHA-256 values, not only the checkpoint file. `verify_snapshot()` passed against the local snapshot (lock SHA-256 `24d26f83ecf1fedd91e1709df364a8e8e29703463e72cc06f4b4197814b012a7`; weight SHA-256 `7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0`), and the 8-thread generated-neutral-gray CPU preflight repeated with identical five finite raw scores. The Qwen scoring suite passes 9/9. This verifies snapshot integrity and synthetic scoring determinism only; Qwen's frozen development quality gates already failed, and its result is unchanged.

Ticket 04 and Ticket 08 parallel primary-source follow-up screens found no new candidate ready for artifacts or evaluation. Ticket 04 alternatives remain blocked by interface, immutable asset, rights, AMD, or frozen quality gaps; Ticket 08's ROSA, MaterialSeg3D, and Intrinsic Image Diffusion leads remain incompatible with the full topology-preserving AMD PBR contract. No assets or truth were accessed; no acceptance gates changed.

The earlier GeoSAM2 source/correspondence attempt left half the faces at sentinel 999 and failed the complete-partition gate. That automatic policy failure is retained in its evidence and was superseded by the separately frozen all-rendered-view policy and production acceptance below; no gate was waived.

An independent Ticket 08 screen found that CHORD cannot proceed under current terms and artifact/runtime evidence: research-only copyleft, gated weights without a verified immutable artifact hash, unpinned source, CUDA 12.8 setup without AMD proof, and unproven topology binding. No model/data/truth artifacts were accessed. Ticket 08 remains open. Evidence: `api/runtime/adapters/pbr/evidence/chord-official-source-screen-2026-09-25.md`.

Fresh parallel source-only screens covered Tickets 07 and 08. Ticket 07's Fine-Grained Material Selection and Materialistic candidates are exemplar selectors rather than named identity classifiers; ObjectFolder is object-level with an incompatible taxonomy. Ticket 08's TexGaussian lacks complete pinned assets/terms and AMD evidence; Seed3D 2.0 is hosted-only; ExMesh++ changes topology and lacks deployable source/assets. None qualifies for a frozen probe. No weights, packages, or truth were accessed. Reports: `api/runtime/adapters/material-identity/evidence/ticket07-new-candidate-frontier-2026-09-25.md` and `api/runtime/adapters/pbr/evidence/ticket08-new-candidate-frontier-2026-09-25.md`.

A Ticket 07 development-only multiview feature screen averaged four observations per region with fixed dual ridge and five object-disjoint folds. Macro-F1 passed at 0.8596, but minimum recall was 0.60, all-region coverage 0.7429, unknown abstention 0.60, ambiguous abstention 0.60, and zero of 1,296 threshold pairs met every gate. It stopped before heldout or AMD execution. Evidence: `api/runtime/adapters/material-identity/evidence/ticket07-multiview-pooling-dev-screen-2026-09-25.md`.

A separate Ticket 08 analytic repair review found no justified third inverse-render candidate: the current registered observations lack an independently accepted topology-bound region map and repeat the same three lights across views, so adding region/spatial/normal priors would be unsupported tuning. No prototype or scorer was run and no heldout data was inspected. Reopen after accepted region evidence and training-only fit stability/conditioning checks. Evidence: `api/runtime/adapters/pbr/evidence/ticket08-analytic-research-2026-09-25.md`.

Current dependency-ready work: Ticket 05 semantic attachment and its frozen evaluation contract are implemented; project-rendered fixture generation, blind evaluation, Florence-2 pre-load review/integration, and target quality remain pending. A DINOv2 RBF development candidate for Ticket 07 failed its unchanged gates and stopped before heldout access; additional development-only candidate work is in progress. Ticket 08's latest source screen (Intrinsic Image Diffusion) also failed its source/runtime/topology gates. Tickets 01–04 and 06 are accepted. Tickets 09–13 remain gated by their declared dependencies; do not advance dependent work early.

### GeoSAM2 workflow integration and current execution environment (2026-09-25)

The reusable GeoSAM2 adapter is connected to Modly's `reference-part-segmentation` processor. The default backend is GeoSAM2, while P3-SAM remains an explicit compatibility choice. The workflow stores both canonical partition labels and untouched upstream label IDs, deterministic topology-bound regions, renderer outputs, correspondence, inference manifests, and an updated Structured Asset sidecar. Canonical IDs are derived from sorted face membership because the pinned model's raw integer labels shifted by +1 across two isolated target processes even though the face partition was exactly equal; raw IDs remain preserved as provenance. This normalization policy is now recorded in `api/runtime/adapters/parts/PROBE_RUNBOOK.md`.

The project entry point includes `workflow-geosam2`, which assembles the headless workflow request and mounts only project-owned API, source, checkpoint, package cache, renderer, and dependency paths into the AMD image. The first invocation exposed missing interactive stdin; the runner was corrected with `--interactive` and JSONL completion/error handling. Managed execution then completed current run `cccccccc-cccc-4ccc-8ccc-cccccccccccc`, persisted separate canonical/upstream labels, and returned a processor `done` event. Full validation is in the production workflow evidence. Image identity was subsequently verified after the project runroot repair: `sha256:7e1bd299f4581985b86b7f300c51ade8ac0e4f7e132e190194f60025fd4039d9` with manifest digest `sha256:0e547e279db22d6a1ab22d50a144c1088d7e9c6f555a1571d271e38be0d7ff18`; see `api/runtime/adapters/parts/evidence/ticket04-geosam2-image-identity-2026-09-25.md`.

Ticket 07's SmolVLM2 snapshot is staged in the ignored project model cache and fully locked: 16 files, 8,992,155,310 bytes, every local SHA-256 verified, and both published shard LFS hashes matched. A synthetic-only CPU load/inference pass with `trust_remote_code=False` produced finite, bitwise-identical logits across two forwards (79.80 s / 71.35 s); no material data or fixture was read. Exact evidence: `api/runtime/adapters/material-identity/SMOLVLM2_ASSET_LOCK.json` and `api/runtime/adapters/material-identity/evidence/SMOLVLM2_CPU_PREFLIGHT_2026-09-25.md`. This is not yet a material-label scoring, quality, or AMD acceptance result. A bounded synthetic RX 7900 GRE preflight is underway.

The SmolVLM2 synthetic RX 7900 GRE preflight passed through the project AMD image using PyTorch ROCm FP16. Model/processor load and transfer took 29.13 s; first and repeated forwards took 11.61 s and 0.91 s. Last-token logits were finite and bitwise equal; PyTorch ROCm peak allocated/reserved VRAM was 5,982,042,624 / 6,306,136,064 bytes. The entire 16-file local snapshot revalidated against its immutable lock. No fixture data was mounted or read. Evidence and exact container boundary: `api/runtime/adapters/material-identity/evidence/SMOLVLM2_AMD_SYNTHETIC_PREFLIGHT_2026-09-25.md`; probe source SHA-256 `e9b67159ae3887792b6afaf2cd8c90fcb774dc08fcecd7dc7b5f78d7bb22afb8`; raw result SHA-256 `e4b9e8f05c8d48aea5d22635395462b2dcef10b2441243909712e54e71c50139`. The subsequent frozen label-scoring screen is complete and failed development quality gates; see the report below. No Ticket 07 acceptance gate passed.

### Truth-path integrity correction (2026-09-25)

Review of the shared Ticket 07 loader found that prior generic-loader runs hashed indexed `truth.json` bytes as opaque data during fixture integrity verification. They did not parse heldout truth rows or use heldout labels for selection; Qwen's development targets were derived only after its raw score artifact was durably committed. The earlier broad claim that truth bytes were never opened/read is superseded. `evaluator.load_fixture_inputs()` now validates truth digest/byte-count metadata against the frozen manifest and skips truth before path stat/hash/open. The focused missing-truth-path boundary test passes, as do evaluator and Qwen scoring regressions (14/14 total). Evidence wording in the Qwen report and Ticket 07 issue is corrected. The SmolVLM2 development-only screen is underway and must use this loader and stop on any failed frozen development gate. No acceptance gate changed.


### SmolVLM2 Ticket 07 development gate (2026-09-25)

The candidate scored all 580 frozen crop views once each on RX 7900 GRE/PyTorch ROCm FP16. Its complete label-free raw logits and manifest were durably committed before deriving development-only recipe labels. The unchanged 140-view development gate failed: macro-F1 0.1585382, minimum recall 0.00, coverage 0.75, unknown abstention 0.00, ambiguous abstention 0.40, and 0/6,014 feasible threshold pairs. The runner stopped before heldout access and created no policy/heldout report. The container mounted a truth-absent fixture copy read-only, along with model/source/runtime; only a fresh ignored output directory was writable, and networking was disabled. 9 focused tests pass. Full run command, telemetry, and artifact hashes: `api/runtime/adapters/material-identity/evidence/SMOLVLM2_DEVELOPMENT_EVALUATION_2026-09-25.md`. Ticket 07 remains unaccepted; no gate changed.


### GeoSAM2 all-view candidate status correction (2026-09-25)

The earlier Ticket 04 row describing GeoSAM2 as rejected applies only to the frozen `opposite_views` policy. It was superseded for the separately predeclared `all_rendered_views` policy, whose candidate quality/resource gates passed. Ticket 04 was later marked acceptance-passed after the 2025-09-25 Modly process run and checklist audit; that historical status was itself superseded on 2026-09-26 when the current registered workflow failed on unassigned face 58. The latest failure and revalidation requirement are recorded in the current status row and latest section. No gate changed.

### Ticket 08 bounded primary-source refresh (2026-09-25)

A source-only screen of FlashTex, Uni-Renderer, PBR-NeRF, and Neural-PBIR found no new evaluation-ready candidate. Their blockers are respectively evidence-conditioning/albedo-light leakage plus CUDA stack; unreleased weights and NVIDIA-only documented runtime; OptiX and joint geometry changes; and CC BY-NC terms plus joint shape/material reconstruction. No code, weights, packages, fixture inputs, or truth were accessed, and no gate changed. Ticket 08 remains unaccepted. Per-candidate sources and detail: `api/runtime/adapters/pbr/evidence/ticket08-new-source-screen-flashtex-unirenderer-pbrnerf-neuralpbir-2026-09-25.md`.

An additional screen of Intrinsic Image Diffusion (IID) found its reported albedo/roughness/metallic channels interesting but not eligible for target probing: rights and immutable model identities are unresolved, the documented environment is NVIDIA-only, and the model interface does not establish material maps bound to caller topology. No code, weights, packages, fixture inputs, or truth were accessed; no acceptance criterion changed. Details: `api/runtime/adapters/pbr/evidence/intrinsic-image-diffusion-primary-source-screen-2026-09-25.md`.

### GeoSAM2 production workflow acceptance (2026-09-25)

Historical run `cccccccc-cccc-4ccc-8ccc-cccccccccccc` executed the then-current Modly process extension on RX 7900 GRE/gfx1100 and returned `done`. Its known-truth source mesh matched vertex/face arrays exactly. The unchanged face-level scorer returned macro-IoU 1.0, per-part IoU 1.0/1.0, full coverage, and zero overlap. Current canonical labels and untouched upstream labels were persisted separately. A completed repeat run produced bitwise-identical canonical partitions and upstream labels; render manifests also matched. The run's 45 stage artifact refs, all 38 render files, both label artifacts, manifest hashes, topology mappings, Structured Asset schema, and unknown confidence states were verified. Runtime was PyTorch `2.11.0+rocm7.14.0` / HIP `7.14.60850`; latency 363,734 ms, peak allocated VRAM 10,423,485,440 bytes, and peak reserved VRAM 14,417,920,000 bytes (13.425 GiB). Explicit PyTorch ROCm remains selected because the image-encoder MIGraphX trial failed frozen `1e-4` tensor tolerance on 2/7 outputs. The previously unresolved local image identity is verified at ID `sha256:7e1bd299f4581985b86b7f300c51ade8ac0e4f7e132e190194f60025fd4039d9`, manifest digest `sha256:0e547e279db22d6a1ab22d50a144c1088d7e9c6f555a1571d271e38be0d7ff18`; exact command/evidence: `api/runtime/adapters/parts/evidence/ticket04-geosam2-image-identity-2026-09-25.md`. These remain useful historical evidence, not current acceptance; see the 2026-09-26 failure section above. Ticket 05 is not dependency-ready while Ticket 04 revalidation is open.

### Ticket 07 BFMS Mask2Former source-only screen (2026-09-25)

A new bounded screen found `jinfengxie/BFMS_1014`, a dense 42-class Mask2Former with immutable snapshot revision `32cd86eb4837b870a9a94bd408084da65ab4ac00`, declared MIT model terms, and publisher-listed weight SHA-256. Its labels cover all five required source categories. It is not eligible for fixture scoring: source-level training-image provenance/terms are unresolved, paper-reported class accuracy is 64.19% for clear plastic and 32.93% for rubber/latex, the published data are building-façade focused, and calibrated unknown/ambiguous and RX 7900 GRE behavior are unproven. Generic PyTorch Mask2Former execution is an AMD plausibility signal only. No files were downloaded, model code was run, fixture/development/heldout material was accessed, or gates changed. Evidence: `api/runtime/adapters/material-identity/evidence/BFMS_MASK2FORMER_OFFICIAL_SOURCE_SCREEN_2026-09-25.md`.

The source-only screen also found the inference-code identity incomplete: the HF snapshot binds config and weights, but the published demo is mutable and no Transformers implementation package/source hash was frozen. This remains a pre-staging blocker alongside training-image rights.

### Ticket 07 DINOv2 fixed RBF development screen (2026-09-25)

A new no-sweep object-disjoint candidate used fixed intercept-centered RBF kernel ridge (`lambda=1.0`) over the immutable Meta DINOv2 ViT-B/14 embeddings. Five fixed renderer folds kept all views of each object together; each fold trained only on 20 supported development objects; excluded OOF cohorts kept every object whole, with the frozen unknown-object recipe distribution unbalanced across folds. Only the 140 development feature rows entered fitting and threshold calibration. Frozen dev gates failed: macro-F1 0.7418491, minimum class recall 0.50, all-region coverage 0.7928571, unknown abstention 0.15, ambiguous abstention 0.80, and 0/19,881 feasible threshold pairs. The runner stopped before any heldout score/truth operation; no policy or AMD run was produced. Model/data/source digests, candidate code hash, exact command, output hashes, and limits: `api/runtime/adapters/material-identity/evidence/DINOV2_RBF_KERNEL_RIDGE_DEV_SCREEN_2026-09-25.md`. Ticket 07 remains unaccepted and gates unchanged.

### Ticket 07 DINOv2 + SigLIP2 fusion screen (2026-09-25)

A predeclared two-rule normalized feature fusion evaluated equal-weight modality concatenation using fixed linear ridge and fixed RBF kernel ridge. Both used the existing immutable DINOv2 and SigLIP2 truth-free feature artifacts; only the 140 development rows entered fit/OOF/calibration. Object views stayed together in the frozen folds. Neither candidate passed: linear macro-F1 0.6775857, minimum recall 0.30, coverage 0.7785714, unknown abstention 0.35, ambiguity abstention 0.55; RBF macro-F1 0.7470846, minimum recall 0.40, coverage 0.7285714, unknown abstention 0.20, ambiguity abstention 0.85. Both had zero feasible threshold pairs and stopped before heldout truth; neither emitted a policy/model lock. No AMD run or gate change. Full evidence: `api/runtime/adapters/material-identity/evidence/DINOV2_SIGLIP2_FUSION_DEV_SCREEN_2026-09-25.md`.

### Ticket 08 MatNet and ShadeNet source-only screen (2026-09-25)

Two additional released local image-to-material models were screened against the frozen PBR contract. Neither passes pre-evaluation source/resource/rights/topology gates: MatNet lacks a complete immutable weight/hash and license chain, documents CUDA GPU requirements in its Materialist implementation, and provides no caller-topology-preservation contract; ShadeNet has a pinned revision and local PyTorch/ONNX artifacts but declares CC-BY-NC-4.0 and has no mesh-topology interface or target evidence. No code/assets were fetched or executed, and no fixtures/truth or gates were changed. Ticket 08 remains blocked. Evidence: `api/runtime/adapters/pbr/evidence/ticket08-models-matnet-shadenet-screen-2026-09-25.md`.

### Ticket 08 DualMat official-source screen (2026-09-25)

DualMat is a distinct image-conditioned dual-path diffusion lead with paper-described albedo, metallic, and roughness predictions and a multi-view extension. The official project materials inspected did not expose a separate source repository, immutable code/weight identities, complete license chain, AMD/resource evidence, or caller-topology/region binding. Stop before acquisition and evaluation. No assets, code, fixtures, or truth were accessed; no rubric changed. Evidence: `api/runtime/adapters/pbr/evidence/ticket08-dualmat-official-source-screen-2026-09-25.md`.

### Ticket 05 complete fixture and scorer readiness (2026-09-25)

The earlier statements that the semantic fixture and blind scorer were pending are superseded. The project-pinned Blender 4.0.2 / Modly GeoSAM2 renderer generated 280 objects with four calibrated RGB crops for each topology-bound target part. The frozen split counts are exact: development has 5 examples per supported role plus 20 unknown and 20 ambiguous; heldout has 20 per role plus 20 unknown and 20 ambiguous. Full `verify_fixture()` independently verified all 3,364 indexed files, source and contract pins, object and exact-geometry disjointness, topology face mapping, target-node identity, projection/crop/camera provenance, split support, and input/truth identity. Fixture manifest SHA-256: `dfd29807bc816d479a67baab48d5509e397a560a45990be0422f32d3e99228e4`.

The focused Ticket 05 fixture, contract, semantic attachment, and evaluator suites passed 23/23 in the project Python 3.12 environment; the fixture smoke regenerates a rendered object and passes. The evaluator requires a durable successful development report tied to exact model/source/prompt/policy/input/fixture digests before heldout truth access. The later Florence development result below supersedes the earlier pending-score state: the candidate failed the frozen development gates and stopped before heldout truth. Ticket 05 remains unaccepted.

The earlier pre-load-only Florence statement above is superseded: all nine local asset sizes and SHA-256 values now match `FLORENCE2_ASSET_LOCK.json`, including the published checkpoint digest. Transformers 4.57.1 loaded weights but failed synthetic generation because the custom nested model lacks the new `GenerationMixin` and cache API. The project's separate Transformers 4.51.3 Hunyuan overlay passed `weights_only=True` loading with no missing/unexpected/mismatched keys and completed synthetic generation with caching on the RX 7900 GRE (12.57 s including setup; 1,452,089,856 bytes allocated and 1,560,281,088 reserved). The blank-image response contained a polygon labeled with the entire prompt, which demonstrates that runtime compatibility does not establish semantic quality. See `api/runtime/adapters/parts/FLORENCE2_RUNTIME_LOCK.json` and `api/runtime/adapters/parts/evidence/ticket05-florence2-amd-synthetic-preflight-2026-09-25.md`. No fixture inputs or truth were accessed for this smoke.

Fresh dependency-ready primary-source screens found no Ticket 07 candidate eligible for development scoring: Columbia CAVE's Matador has conceptual taxonomy-aware uncertainty but lacks an immutable inference checkpoint, cleared terms, frozen taxonomy mapping, and AMD route. Ticket 08's newly surfaced LumiTex is incompatible with the 14 GiB/device and rights/identity gates: its documented stack uses CUDA 12.8, its 12B BF16 model exceeds the budget, it includes non-commercial components, and topology invariance is unproven. Reports: `api/runtime/adapters/material-identity/evidence/MATADOR_HIERARCHICAL_MATERIAL_RECOGNITION_SOURCE_SCREEN_2026-09-25.md` and `api/runtime/adapters/pbr/evidence/lumitex-official-source-frontier-2026-09-25.md`. Neither ticket's criteria changed.

New parallel Ticket 07/08 source-only screens also found no eligible candidate. MateViT lacks immutable checkpoint identity/terms, AMD evidence, and unknown/ambiguous calibration; NDJIR changes geometry and omits metallic, while DiffReg-PBIR is noncommercial, CUDA/OptiX-bound, and reconstructs its own mesh. Reports: `api/runtime/adapters/material-identity/evidence/MATEROBOT_OFFICIAL_SOURCE_SCREEN_2026-09-25.md` and `api/runtime/adapters/pbr/evidence/ticket08-ndjir-diffreg-pbir-primary-source-screen-2026-09-25.md`. No assets or fixture truth were accessed; ticket gates remain unchanged.

Ticket 05's independent semantic-node review findings were remediated. The node preserves other adapter assertions and stage artifacts, contains paths, verifies geometry/sidecar/observation/view provenance and installed executable sources around inference, coordinates same-convention per-asset writers, and cleans newly created artifacts after failed commits. Focused node + semantic tests pass 23/23; compilation and JSON checks pass. The independent review caveat remains: host writers that ignore the node's lock could race outside the shared lock convention. These node tests do not pass Florence model quality or any Ticket 05 acceptance criterion.

## Ticket 05 deployment seam update (2026-09-25)

Florence now has a source-digest-addressed, code-only install overlay: `scripts/install-florence-semantic-adapter.sh` builds the `api/pyproject.toml` distribution, registers `microsoft.florence-2-base.v1`, includes the asset/runtime lock files, and verifies those entry points/files before publishing. The node README states how the Modly process-extension runtime must add the emitted overlay to `PYTHONPATH`, while model files remain separately provided through `MODELS_DIR`. Validation: installer wheel build completed; `api.tests.test_ticket05_semantic_adapter_install` passed 2/2 against the installed distribution. This supersedes the earlier statement that no installed entry point/package path existed. No fixture inference or truth access occurred, and no semantic quality gate has passed; Ticket 05 remains unaccepted. The installed-package check does not itself establish a full Modly workflow GPU run.

### Ticket 07 cross-view variability candidate (2026-09-25)

One preregistered, four-view physical-appearance candidate concatenated the mean and population standard deviation of 30 topology-masked cues, with a fixed dual-ridge classifier and existing object-disjoint development folds. All 580 truth-free feature rows were written before loading the development plan; heldout truth was never read. It failed the unchanged development gates: macro-F1 0.7183261 (<0.85), minimum class recall 0.20 (<0.80), coverage 0.60 (<0.80), unknown abstention 0.60 (<0.90), ambiguous abstention 0.40 (<0.90), and 0/1,296 feasible threshold pairs. Stopped before heldout or GPU evaluation. Preregistration, source, features, OOF predictions, and report are retained under `api/runtime/adapters/material-identity/`; hashes and detailed results are in `evidence/TICKET07_VIEW_VARIABILITY_DEV_SCREEN_2026-09-25.md`. Ticket 07 remains unaccepted; no criteria changed.

### Shared ROCm stage telemetry and Florence wiring (2026-09-25)

`AMDInferenceRuntime.profile_stage` now profiles an already-selected PyTorch ROCm closure exactly once, rejects CPU execution, and persists bounded failure diagnostics plus device/runtime identity, latency, peak allocated VRAM, and peak reserved VRAM. Florence records each part-view inference profile in assertion provenance and the shared runtime JSONL log. AMD runtime tests pass 17/17; combined Florence adapter/development/install and AMD runtime suites pass 35/35. No candidate fixture inference has yet been performed against this updated source digest.

### Ticket 05 API dependency repair (2026-09-25)

`api/runtime/adapters/parts/semantic_evidence.py` imports Pillow for required evidence-image processing, but the API requirements omitted it; `Pillow==12.3.0` is now pinned, matching the project AMD runtime image. Focused semantic evidence pytest coverage passes 9/9; the project API unittest suite passes 64 cases, with its pytest-only evidence module reported separately. Ticket 05 remains unaccepted pending the one-shot semantic development gate and full Modly GPU workflow evidence.

### Ticket 05 Florence AMD development result (2026-09-25)

The candidate's single frozen development-only run completed on RX 7900 GRE in `localhost/modly-amd-migraphx:ticket02` with networking disabled and heldout input/truth paths masked. The runner committed 80 raw development examples before scoring. It failed with supported coverage 0.0, all eight role recalls 0.0, unknown abstention recall 0.0, and ambiguous abstention recall 1.0; 5/5 non-ambiguous gates failed. Generated strings reproduced the ontology prompt with corrupted labels, leading to exact-policy ambiguous abstentions. The candidate stopped with no retry, tuning, or heldout score. The durable commit, predictions, score, and 320 per-view profiles are under `api/runtime/adapters/parts/evidence/ticket05-florence-development-2026-09-25/`. Backend PyTorch ROCm; median/p95/max per-view latency 506.62/525.38/6,763.36 ms; peak allocated/reserved VRAM 753,033,728/849,346,560 bytes. This is real candidate GPU and telemetry evidence, but Ticket 05 remains unaccepted. The exact locally pinned `timm==1.0.30` wheel was installed without dependency resolution or network access into the isolated Florence runtime overlay before the run. Shared stage profiling plus Florence evidence provenance tests pass within the combined focused 35-test suite.

### Ticket 05 OWLv2 alternative candidate (2026-09-25)

OWLv2-base-patch16-ensemble at pinned revision `57beb61adb5abda3de4a9796bc35ae60bc4b9802` passed the official-source screen and its exact 8-file Apache-2.0 safetensors snapshot was staged/hash-verified on the work drive. A distinct four-view consensus policy was preregistered before weights were staged. Synthetic-only, network-disabled target-GPU preflight then rejected this candidate: exact ontology-definition query token lengths were 19–28, above the checkpoint's 16-token text limit; registered Torch-MIGraphX 1.2 forward failed with a 28-versus-16 tensor-size error. No prompt truncation, shorter-query substitution, retry, fixture input, or truth access occurred. No development prediction or score exists for OWLv2. See `api/runtime/adapters/parts/evidence/TICKET05_OWLV2_SYNTHETIC_PREFLIGHT_REJECTION_2026-09-25.md` and `.modly-amd-runtime/results/owlv2-migraphx-synthetic-probe.json`. The candidate was rejected before development evaluation; Ticket 05 remains unaccepted.

### Ticket 07 RF-MatID official-source screen (2026-09-25)

RF-MatID is a distinct material-identification research system, but its official contract uses RF frequency/time-domain signals rather than imagery or topology-bound RGB material regions. The repository declares BSD-3-Clause for code, while the inspected sources expose no immutable pretrained inference checkpoint or distinct weight terms. PyTorch use is not RX 7900 GRE qualification; the repo documents a `cuda` training-device option and no AMD/MIGraphX/resource evidence. This candidate cannot enter the unchanged image-based Ticket 07 screen. No code, package, weights, fixture input, or truth was accessed. Evidence: `api/runtime/adapters/material-identity/evidence/RF_MATID_OFFICIAL_SOURCE_SCREEN_2026-09-25.md`. Ticket 07 remains unaccepted; no gates changed.


### Ticket 05 Qwen3-VL-2B conditional candidate (2026-09-25)

Official-source screen found Qwen3-VL-2B-Instruct at immutable snapshot `89644892e4d85e24eaac8bacfd4f463576704203`, Apache-2.0 and safetensors; its 4.26 GB checkpoint and image-plus-text generation interface are a plausible fit for frozen definition prompts, but file size does not prove RX 7900 GRE memory fit. Official Qwen instructions require Transformers >=4.57.0, versus the accepted AMD image's 4.51.3; a digest-locked local overlay and exact AMD preflight are mandatory. No Qwen weights were newly staged specifically for Ticket 05, and no Ticket 05 fixture inputs/truth were accessed. The exact snapshot had already been staged and used by Ticket 07 before this preregistration; see the later prior-access conflict record. One-shot candidate prompt, output parsing, four-view decision, frozen acceptance gates, and stop rule are recorded before asset acquisition in `api/runtime/adapters/parts/evidence/ticket05-qwen3-vl-development-preregistration-2026-09-25.md`. Candidate is conditional, not accepted; no gates changed.

The work-drive Transformers 4.57.1 overlay passed a network-disabled import-only check inside the accepted ROCm image on RX 7900 GRE: Python 3.12.3, PyTorch `2.11.0+rocm7.14.0`, HIP `7.14.60850`, Torch-MIGraphX 1.2 and Qwen's `AutoModelForImageTextToText`/`AutoProcessor` import successfully. This is not a model or inference preflight; Qwen weights remain unstaged and no fixture was mounted. See `api/runtime/adapters/parts/evidence/TICKET05_QWEN3_RUNTIME_IMPORT_PREFLIGHT_2026-09-25.md`.

### Ticket 08 SVBRDF Uncertainty source-only screen (2026-09-25)

The SIGGRAPH 2025 frequency-analysis SVBRDF optimizer documents base-color, metallic, roughness, and entropy outputs for multi-view captures, but is not evaluation-ready: the current upstream branch could not be resolved to an immutable commit in this network environment; `nvdiffrast` is an unpinned Git dependency with no documented AMD backend; caller-mesh topology/region binding and separate benchmark-data terms remain unverified. Rejected before acquisition or evaluation. No code, packages, weights, fixture data, or truth were accessed; no gates changed. Evidence: `api/runtime/adapters/pbr/evidence/ticket08-svbrdf-uncertainty-official-source-screen-2026-09-25.md`. The source tree is not a Git repository in this mount, so no local Git diff/status or commit is available.

### Ticket 07 ConCLIP construction-waste source screen (2026-09-25)

Screened the distinct ConCLIP multimodal construction-waste material classifier against the frozen Ticket 07 contract. Its publisher-described VL-Concrete/VL-Metal image-text classification tasks do not establish topology-bound region outputs or the five required material labels; no immutable inference checkpoint, checkpoint rights, calibrated unknown/ambiguous behavior, frozen development metrics, or RX 7900 GRE ROCm/MIGraphX evidence was identified. Rejected before acquisition or scoring; no code/assets, fixture inputs, or truth were accessed and no gates changed. Evidence: `api/runtime/adapters/material-identity/evidence/CONCLIP_CONSTRUCTION_WASTE_SOURCE_SCREEN_2026-09-25.md`.


### Ticket 05 Qwen3-VL prior-access conflict (2026-09-25)

The exact Qwen3-VL-2B-Instruct snapshot at `89644892e4d85e24eaac8bacfd4f463576704203` was staged and used by the separate Ticket 07 material-identity candidate workflow before the Ticket 05 Qwen preregistration. Ticket 07 used a distinct fixture; no Ticket 05 fixture access is claimed. The Ticket 05 preregistration's claim that it preceded candidate-weight access is invalid for this checkpoint; do not score this model as a preregistered Ticket 05 one-shot. This does not alter the frozen Ticket 05 gates. The adapter and its synthetic unit tests remain software work only. No additional Qwen inference, fixture access, or truth access was performed for Ticket 05. Evidence: `api/runtime/adapters/parts/evidence/TICKET05_QWEN3_VL_PRIOR_ACCESS_CONFLICT_2026-09-25.md`.


### Additional dependency-ready source screens (2026-09-25)

Ticket 07 screened the publisher's RGB-only `akde/mwc-rgbt-waste-sorting` conveyor tracklet classifier. It lacks required material labels, unknown/ambiguous calibration, topology-bound material-region output, complete checkpoint rights/identity, and RX 7900 GRE evidence. No candidate assets or fixtures/truth were accessed. Report: `api/runtime/adapters/material-identity/evidence/AKDE_MWC_RGBT_WASTE_SORTING_SOURCE_SCREEN_2026-09-25.md`.

Ticket 08 screened NI-Tex and MatE from official sources. NI-Tex's documented CUDA/H100-H200 stack and 39.5 GB mutable-main listing do not qualify for the target; MatE emits tileable textures without metallic or a caller-mesh topology interface, and its code is unreleased. Neither entered evaluation. No assets or fixture/truth were accessed and thresholds remain unchanged. Report: `api/runtime/adapters/pbr/evidence/NI_TEX_MATE_SOURCE_SCREEN_2026-09-25.md`.

Ticket 05's Qwen adapter has 7/7 focused contract and asset-lock unit tests passing in the work-drive Python 3.12 API environment. Transformers 4.57.1 and `Qwen3VLForConditionalGeneration` import in the pinned overlay. This is adapter-contract evidence only; the checkpoint was previously used under Ticket 07, so it cannot be treated as a valid preregistered Ticket 05 model candidate. No Ticket 05 model inference or fixture/truth access occurred.


### Ticket 05 Qwen3 progress-note correction (2026-09-25)

Earlier interim wording that Qwen3-VL weights remained unstaged is superseded: the exact snapshot was staged and loaded in the separate Ticket 07 workflow before Ticket 05's preregistration. Ticket 07 used its own fixture; no Ticket 05 fixture input/truth access is claimed. The checkpoint is process-rejected as a preregistered Ticket 05 candidate. The Ticket 05 Qwen adapter and 7 passing focused contract tests are software evidence only. Do not use that checkpoint for Ticket 05 scoring. See `api/runtime/adapters/parts/evidence/TICKET05_QWEN3_VL_PRIOR_ACCESS_CONFLICT_2026-09-25.md`.

Ticket 05 screened `OpenGVLab/InternVL3-2B` at pinned revision `899155015275a9b7338c7f4677e19c784e0e5a21` from official sources. It is only conditionally eligible for exact source retrieval and local static/license review; no source or weights were downloaded or executed, and no Ticket 05 fixture/truth was accessed. Require exact local source hashes and review before any `trust_remote_code=True` load, then preregister prompt/template, preprocessing, resources and one-shot acceptance policy before weights or fixture access. Evidence: `api/runtime/adapters/parts/evidence/TICKET05_INTERNVL3_2B_SOURCE_SCREEN_2026-09-25.md`.


### Ticket 04 historical status-note reconciliation (2026-09-25)

At the time, the accepted-ticket audit found stale pre-GeoSAM2 text in older Ticket04 candidate-frontier paragraphs and reconciled status to acceptance passed based on the then-current Modly process-extension run. This historical reconciliation was superseded on 2026-09-26 by the current-code registered run failure and the revalidation status recorded at the top of the ticket and in the latest ledger section. The earlier run's command used the script's pinned `localhost/modly-amd-geosam2:ticket04` tag; its local image ID and manifest were read-only inspected after runroot recovery, but the production result did not record an in-run image digest attestation.


### Ticket 05 Qwen2-VL candidate screen (2026-09-25)

**Superseded by the user's four-month model-release constraint and ineligible for further Ticket05 model work.** The historical Qwen2-VL source screen and adapter/tests remain in the workdrive, but no Qwen2 weights or Ticket05 fixture/truth were accessed and no AMD/MIGraphX/resource/semantic claim was made. Do not use or score Qwen2 for Ticket05. This change does not alter frozen evaluation thresholds or ticket gates. Evidence: `api/runtime/adapters/parts/evidence/TICKET05_QWEN2_VL_2B_SOURCE_SCREEN_2026-09-25.md`.

InternVL3-2B is held at its inconsistent license/lineage metadata; its 12 MB source-review snapshot is on the workdrive but its weights were not fetched. SmolVLM-500M's pinned model repo has no explicit license in its metadata or README and was not advanced; no weights were acquired.


### Ticket 05 North Micro Vision source screen and provider constraint (2026-09-25)

`CohereLabs/North-Micro-Vision-Instruct` at pinned Hub revision `46b719694e3bad142f3e931774f4622f1024009e` is a conditional local candidate: release Aug 12, 2026 (44 days before this screen), 2.4B, Apache-2.0, bf16. Its Hub tree contains no custom model Python; config has no `auto_map`; official Transformers 5.16.0 owns the built-in CohereCompass model/processor. The accepted project image with the existing Transformers 4.57.1 overlay lacks the required model class. No North Micro weights or Ticket05 fixture/truth were read. No model GPU, MIGraphX, VRAM, or semantic test passed; no acceptance thresholds changed. Before any fixture access, stage and hash-lock an isolated Transformers 5.16.0/dependency overlay and run a synthetic-only target-GPU loader/generation and eager-vs-MIGraphX qualification. Evidence: `api/runtime/adapters/parts/evidence/TICKET05_NORTH_MICRO_VISION_SOURCE_SCREEN_2026-09-25.md`.

The provider implementation is now wired through the existing semantic workflow node: the local North Micro option is selected by default but fails clearly as not-ready until its adapter passes qualification; it does not auto-switch. GPT-6 Luna is a distinct remote OpenAI Responses API option and is never a silent fallback. Immediately before the remote semantic process launches, a main-process native dialog discloses image transfer; decline prevents process launch. Renderer consent claims are stripped. The Python runner uses its constructor-bound extension ID and only the built-in semantic extension can receive the environment key after main-process consent. Provider/model/endpoint/locality and input/output provenance are retained. North Micro runtime/AMD and Ticket05 semantic gates remain open.


### Current product provider implementation status (2026-09-25)

The coding implementation remains on GPT-6 Luna. At the product level, the node defaults to the local North Micro VLM candidate, currently not ready, with GPT-6 Luna (`gpt-6-luna`) as a separately selected remote OpenAI Responses API option. A main-process native dialog discloses image transfer immediately before remote inference; decline aborts before process launch. Missing credentials, provider errors, and unready local resolution fail visibly; there is no fallback. The local candidate is not accepted until its unchanged Ticket 05 gates pass.

The product provider seam is implemented and checked with 8 synthetic/mock Python tests, 3 main-process authorization tests, 3 process credential-isolation tests, 6 network-free workflow preflight tests, and TypeScript no-emit typechecking. JS extension workers never inherit `OPENAI_API_KEY`; the constructor-bound built-in semantic Python runner gets it only after main-process confirmation. Renderer-supplied consent is stripped. The requested GPT-6 Luna adapter model remains `model_id`; the API response's `model` is required and retained separately as `served_model_id`, alongside the complete raw response. Requests set `store: false` to disable Responses object storage, without claiming zero retention. The rebuilt code-only adapter wheel SHA-256 is `74841ac7e58371a04994fff4c18ec7d9597238f6a23c0c7873c2a1d232c1c692`; its source-digest-addressed overlay is `.modly-amd-runtime/package-cache/florence/3c152d2fcb72405fe757c712c172d4953b504a2ab1ab530bca4330c7085916ac/site-packages`. No API request or real credentials were used. North Micro has only passed the source screen; its hash-locked Transformers 5.16 runtime overlay, candidate-specific preregistration, synthetic AMD/MIGraphX qualification, and Ticket 05 evaluation remain unimplemented/unaccepted. No Ticket 05 fixture/truth or North Micro weights were touched.

### Ticket 05 North Micro Vision synthetic preflight stop (2026-09-25)

This supersedes the earlier North Micro source-screen note saying weights were untouched. Following a candidate-specific preregistration, the exact checkpoint at `CohereLabs/North-Micro-Vision-Instruct@46b719694e3bad142f3e931774f4622f1024009e` was downloaded to the work-drive cache and verified at 4,969,765,248 bytes, SHA-256 `cd9a9ed867111bbc6b6e062bf0b3b9c88af183eb41003083501a76c84e912046`. Transformers 5.16.0 loaded 577/577 entries using the isolated work-drive overlay. The synthetic-only probe failed before generation when the processor's PNG decoder reported that torchvision was not compiled with libPNG. Generation calls: 0; MIGraphX parity: not attempted; no decoder substitution or retry; no Ticket 05 fixture/truth access. North Micro is not qualified and Ticket 05 remains unaccepted. Raw traceback is `.modly-amd-runtime/results/north-micro-vision-v1/synthetic-preflight-failure.log`; exact preregistration, locks and report are retained in `api/runtime/adapters/parts/` and its `evidence/` directory.

### Work-drive recovery of task-created boot-drive artifacts (2026-09-25)

Task-created Modly temporary artifacts found under `/tmp` were copied to `.modly-amd-runtime/boot-drive-recovered-20260925/tmp/`, byte-verified against their originals (212 files; 0 symlinks), and indexed in `BOOT_DRIVE_RECOVERY_MANIFEST.json`; the exact source files were then removed. Task-created npm cache/log entries and the task-created pip ephemeral wheel cache were also removed from their original boot-drive locations after preserving the relevant recovered artifacts. Unrelated pre-existing caches and files were left untouched. All subsequent project activity and caches are directed to the work drive.

The first recovery note counted the initial 212 files only. A bounded follow-up found eleven additional Modly-specific roots under `/tmp` (including 1,142 files/directories, 3,171,224 bytes) that were not in the initial manifest. They have now been copied and file-hash verified under `.modly-amd-runtime/boot-drive-recovered-20260925/additional-tmp/`, recorded in `ADDITIONAL_RECOVERY_MANIFEST.json`, and the exact source roots removed. Combined recovered evidence remains on the work drive. Unrelated temporary files and pre-existing user caches were not touched.

### Ticket 05 LFM2.5-VL-3B source-only eligibility screen (2026-09-25)

A bounded official-source screen found `LiquidAI/LFM2.5-VL-3B` at revision `35a118d938ce6d123ac2d371649f24a8efb69058`, released Aug 12, 2026. It has built-in Transformers 5.16.0 model/processor classes and a 6,247,065,472-byte BF16 checkpoint with published SHA-256 `8413ba08bc7490552f3cf7611809927a825067e4375f7c39dfcb74df95301908`; no weights were fetched. Technical entry is conditional, but rights are on hold: LFM Open License v1.0 does not grant commercial use to entities with annual revenue >= USD 10M, and the intended legal-entity/use facts are unresolved. It also shares the Transformers 5.16 torchvision image-decoder path that stopped North Micro, so any future synthetic protocol must freeze its allowed input representation before a one-shot probe. No assets/code were executed; no fixture/truth or gates/status changed. Report: `api/runtime/adapters/parts/evidence/TICKET05_LFM25_VL_3B_ELIGIBILITY_SCREEN_2026-09-25.md`.

### Ticket 05 VisionPsy-Nano-460M eligibility screen (2026-09-25)

The official-source screen identifies `qvac/VisionPsy-Nano-460M` at immutable revision `a779cb695f7627c36ded60a82a8c3cc73f03fa24`, released July 29, 2026. Its reported FP32 safetensors checkpoint is 2,029,498,200 bytes with LFS SHA-256 `5b5ddf088789717db7d7bc25d8ddd88ee4bf678ff36dfe28ca5d760f79d1e3fa`; weights were not fetched. This is a text-generating vision role-classifier candidate, not a segmenter. Static screen blockers before preregistration: its config requires custom model/processor code, initialization makes an unpinned lookup of `HuggingFaceTB/SmolLM2-360M-Instruct`, and its card identifies training-data subsets with noncommercial licenses. Model metadata declares Apache-2.0, but rights and attribution review remains open. No AMD runtime evidence exists. Report: `api/runtime/adapters/parts/evidence/TICKET05_VISIONPSY_NANO_460M_ELIGIBILITY_SCREEN_2026-09-25.md`. No code execution, weights, fixture/truth access, or gate/status changes occurred in the eligibility screen.

VisionPsy's exact pinned repository Python source was then retrieved from the official Hub revision to `.modly-amd-runtime/candidate-source/visionpsy-nano-460m/a779cb695f7627c36ded60a82a8c3cc73f03fa24/`; all 12 `.py` files were checked against the Hub commit/byte metadata, SHA-256 recorded, and AST-parsed without importing or executing them. Review does not clear preregistration: the active model constructor reaches `AutoTokenizer.from_pretrained(HuggingFaceTB/SmolLM2-360M-Instruct)` without an immutable revision or local-only setting; legacy base loaders also retain unpinned Hub fetch paths. The exact report/hash manifest is `api/runtime/adapters/parts/evidence/TICKET05_VISIONPSY_SOURCE_REVIEW_2026-09-25.md`. No weights, fixture/truth, or acceptance gates were touched.

### Parallel critical-path fast-path triage (2026-09-25)

To avoid repeating low-yield serial screens, Tickets 05, 07, and 08 were assessed in parallel with no threshold changes and no fixture/model runs. Ticket 05's quickest local VLM lead remains VisionPsy-Nano; its exact source is statically reviewed, and the unpinned tokenizer lookup appears avoidable by separately pinning/staging the tokenizer and passing its local path, but the disclosed noncommercial training-data subsets remain a rights-policy hold before weight acquisition. Ticket 07 is no-go for another inference run: all previously advanced rights-cleared candidates failed fixed quality/OOD gates; DMS46 is only a conditional route after Apple's missing acknowledgements/checkpoint terms are cleared. Ticket 08 is no-go for SuperMat's one-shot test until official SD 2.1 base provenance/license and Modly intended-use clearance are resolved; Material Anything's PyTorch3D/Kaolin path is not a quicker AMD route. Reports: `api/runtime/adapters/parts/evidence/TICKET05_VISIONPSY_SOURCE_REVIEW_2026-09-25.md`, `api/runtime/adapters/material-identity/evidence/TICKET07_FASTPATH_DECISION_2026-09-25.md`, and `api/runtime/adapters/pbr/evidence/ticket08-supermat-fast-path-decision-2026-09-25.md`. No inference or fixture scoring was run in these triage lanes.

### Ticket 05 VisionPsy tokenizer reproducibility fast path (2026-09-25)

The pinned SmolLM2 base-tokenizer snapshot was acquired without model weights: `HuggingFaceTB/SmolLM2-360M-Instruct@a10cc1512eabd3dde888204e902eca88bddb4951`, five files (3,376,028 bytes), verified locally against immutable Hub commit, exact sizes, and SHA-256 manifest under `.modly-amd-runtime/candidate-tokenizers/`. A network-disabled, read-only tokenizer-only smoke in the project AMD image passed using the project-local Podman store/runroot: GPT2Tokenizer, vocab 49,152, EOS/pad `<|im_end|>` ID 2, chat template present. It mounted no VisionPsy checkpoint/source or Ticket 05 fixture/truth and does not establish model runtime. The candidate preregistration now keeps the base `TOKENIZER_DIR` separate from VisionPsy `MODEL_DIR`; exact report is `api/runtime/adapters/parts/evidence/TICKET05_VISIONPSY_TOKENIZER_SMOKE_2026-09-25.md`. The protocol freezes one synthetic-only run but blocks checkpoint acquisition pending affirmative documented rights-policy resolution; asset/runtime lock remains explicitly incomplete. No acceptance gate changed.

The VisionPsy local-only tokenizer source overlay has also passed independent static review: all 12 pristine-source and 12 overlay-source hashes were independently recomputed, and `processors.py` is the only changed file. Its single change forces `local_files_only=True` in `get_tokenizer`. The review clears only a code-review prerequisite; no candidate source was imported/executed, no checkpoint was acquired, and the unresolved rights/runtime gates remain. Evidence: `api/runtime/adapters/parts/evidence/TICKET05_VISIONPSY_LOCAL_ONLY_SOURCE_OVERLAY_2026-09-25.md`.

### AMD Runtime real-GPU profiler smoke (2026-09-25)

The `AMDInferenceRuntime.profile_stage` telemetry path was exercised inside the existing pinned `localhost/modly-amd-migraphx:ticket02` image with `/dev/kfd` and `/dev/dri` available. One synthetic FP16 1024x1024 matrix-multiplication closure completed on RX 7900 GRE/gfx1100; no model weights, candidate adapter, or inference ran. The profile reported PyTorch ROCm, Torch 2.11.0+rocm7.14.0, ROCm 7.14.60850, MIGraphX 2.16.0.dev+20250912-17-575-g4bcfe75b2, torch-migraphx 1.2, latency 1181.6702369978884 ms, peak allocated VRAM 39,845,888 bytes, and peak reserved VRAM 54,525,952 bytes. Persisted JSON and JSONL run records agree; focused evidence: `api/runtime/amd/evidence/profile-stage-rocm-smoke-2026-09-25.md`. This validates live profiler telemetry only and does not change ticket acceptance.

### Ticket 05 recent local VLM alternatives screen (2026-09-25)

A bounded official-source screen found no new local VLM that clears the user's four-month release constraint and initial eligibility bar. Jina-VLM is both outside the window and CC-BY-NC; Gemma 4 E2B, Villanova-2B-VL-2603, and Qwen3.5 small models are March releases and outside the window; Cortex-Mini preview is a 27B Qwen3.8 derivative; LFM2.5-VL-DSpark is a 279.5M drafter that requires its separate 3B target. The in-window North Micro, LFM2.5-VL, and VisionPsy paths remain governed by their existing stop/rights evidence. No model source, weights, dependencies, fixture inputs, or truth were accessed and no criterion changed. Detail and primary links: `api/runtime/adapters/parts/evidence/TICKET05_RECENT_LOCAL_VLM_ALTERNATIVE_SCREEN_2026-09-25.md`.

### Ticket 05 Decider 2B Vision GGUF synthetic HIP smoke (2026-09-25)

The user-supplied `mindchain/decider-2b-vision-GGUF` candidate is now hash-locked at immutable repo/upstream revisions; both text and mmproj assets match published sizes and SHA-256. Exact llama.cpp conversion-pinned source `9575389609d6f8437de0b205561a4824d217c409` built successfully for HIP/gfx1100 using the already installed host ROCm 7.2.53211. A native harness mirrored the pinned upstream `Answer: (` letter-logit protocol. It jointly processed four generated synthetic images on the RX 7900 GRE and produced byte-identical raw A-J slot scores over two runs. This is a conditional technical pass for a Decider-shaped harness, with CPU buffers/compute and image-position warnings disclosed. It is not a quality or acceptance result; no Ticket 05 fixture input/truth was used. The host ROCm version differs from the project-pinned ROCm 7.14 image. Next: verify source-vs-GGUF slot-score parity, resolve closed-choice assertion representation in the candidate schema, then preregister candidate-specific development scoring and qualify through Modly's extension/workflow path. Rights provenance remains partially disclosed. Details: `api/runtime/adapters/parts/evidence/TICKET05_DECIDER_2B_SYNTHETIC_HIP_SMOKE_2026-09-25.md`. Ticket 05 criteria and thresholds unchanged.

### Ticket 05 Decider llama.cpp/HIP on the project-pinned ROCm runtime (2026-09-26)

The conversion-pinned llama.cpp source `9575389609d6f8437de0b205561a4824d217c409` compiled inside the existing `localhost/modly-amd-migraphx:ticket02` image against its pinned ROCm/HIP `7.14.60850` SDK for `gfx1100`; the image exposed the RX 7900 GRE without any ROCm installation or upgrade. The custom native `mtmd` probe jointly processed four generated synthetic views and read the ten restricted answer-slot logits. Two runs were byte-identical (SHA-256 `a579bd95cb97e15dc1943abfac47976c5f32de7e54d9aa44dc0bbe8f8963db93`). This establishes target-runtime technical feasibility for the local resolver harness, not semantic quality, source/GGUF numerical parity, Modly provider integration, or Ticket 05 acceptance. `find_slot: non-consecutive token position` warnings remain, and exact upstream PyTorch checkpoint/tokenizer/processor assets are not currently available for paired parity. No fixture/truth was accessed; Ticket 05 thresholds and status remain unchanged. Evidence: `api/runtime/adapters/parts/evidence/TICKET05_DECIDER_ROCM714_HIP_QUALIFICATION_2026-09-26.md` and `api/runtime/adapters/parts/evidence/TICKET05_DECIDER_UPSTREAM_GGUF_SYNTHETIC_PARITY_2026-09-26.md`.

### Resolver provider selection correction (2026-09-26)

Per the user's direction, Decider 2B Vision is the only resolver exposed by the semantic workflow, is its default, and is the only provider registered for that workflow. The workflow processor rejects unsupported provider values with `UNKNOWN_SEMANTIC_PROVIDER`; it does not route to North Micro or GPT-6 Luna. The GPT-6 Luna resolver implementation file itself was not changed. The installer and distribution metadata are being aligned to Decider-only registration. This only changes resolver selection and does not pass the open parity or Ticket 05 acceptance gates.

Completed the Decider-only resolver/package seam: only `mindchain.decider-2b-vision.gguf.v1` is registered; the installer verifies the Decider asset and harness locks and installs to a work-drive digest-addressed overlay. GPT-6 Luna source was left unchanged. The typed `closed-vocabulary-choice.v1` source assertion preserves the selected option/text and prompt/table digests separate from the normalized label, with mapping validation in the node. Package wheel SHA-256 `8f5b9766706bd15969c65a26654d5b94b4fe698f373c1b2b457b9b2a897e7a98`; focused installation/semantic/node/Decider suites: 42 passed, 2 skipped for missing optional semantic-evidence dependencies.

Paired upstream/GGUF descriptive comparison ran on the preregistered synthetic contact sheet and prompt. Both ranked J first, but full order differed (upstream C/E tie; GGUF C>E); mean absolute logit delta 0.222838 and maximum 0.5352 (F). Effective image grids align after upstream 2x merge, but preprocessing and hidden-state equivalence are unproven. Six native recurrent-position warnings remain. No numeric tolerance was preregistered, so parity remains unaccepted. Evidence: `api/runtime/adapters/parts/evidence/TICKET05_DECIDER_UPSTREAM_GGUF_DESCRIPTIVE_PARITY_2026-09-26.md`. No fixture or truth was accessed; Ticket 05 quality and acceptance remain open.

An independent post-parity authorization audit confirmed the reference-only preregistration explicitly forbids treating that synthetic comparison as authorization for Ticket 05 development scoring. It also identifies unresolved recurrent-state metadata risk and the four-view equivalence gap. No development input or truth was opened. Evidence: `api/runtime/adapters/parts/evidence/TICKET05_DECIDER_POST_PARITY_AUTHORIZATION_AUDIT_2026-09-26.md` (SHA-256 `58224d3a40d46e35bd15bb6832d55c8983be2df49400c4724bb5dcfb986581cb`). A proposed candidate-specific protocol amendment was prepared at `api/runtime/adapters/parts/evidence/PROPOSED_TICKET05_DECIDER_DEVELOPMENT_PROTOCOL_AMENDMENT_2026-09-26.md` (SHA-256 `95115951d9784a4133176183ac995193dc91ae9627114571a9a2303a88f96cb9`); it proposes four independent one-image calls matching the pinned upstream interface and a fixed equal-weight aggregate into one part-level decision. It preserves the frozen thresholds, calibration objective, and heldout controls but is not active and does not authorize fixture scoring. Owner approval or replacement is required before freezing the candidate protocol and accessing development inputs.

An additional static lifecycle audit confirms the current native harness clears llama.cpp hybrid memory with `llama_memory_clear(..., true)` before each per-part request, runs one complete multimodal prompt, and does not use sequence removal/shift/division or state save/restore APIs. This limits the observed cache-management risk for the currently supported clear-per-request path, but does not prove recurrent-state correctness within a prompt or answer-slot numerical correctness. Report: `api/runtime/adapters/parts/evidence/TICKET05_DECIDER_ONE_SHOT_MEMORY_LIFECYCLE_AUDIT_2026-09-26.md` (SHA-256 `729240ac382c89b26d81cef8f5ecb22cb2b8653c9b6c202382fe8ce93eda597b`). No inference or fixture access occurred; the freeze and acceptance remain unchanged.

### Ticket 08 source frontier refresh (2026-09-26)

A bounded official-source search found no PBR estimator that clears all current immutable-identity, complete-rights, unchanged-topology/material-region mapping, local AMD/14 GiB, and frozen quality prerequisites. Marigold IID Appearance v1.1 is the strongest new channel-fit lead (sRGB albedo, linear roughness/metallicity; official local Diffusers inference), but cannot advance: its linked official model license has an unfilled `[insert use restrictions]` attachment, its linked Stable Diffusion model license returned HTTP 401, and the reviewed Hub listing did not expose a complete immutable revision/file-hash lock. Its ROCm/RX 7900 GRE path, projection integration, preregistration, and quality remain unmeasured. No assets, packages, fixture inputs, development predictions, or truth were accessed. Highest-value next step is to obtain complete model-specific binding terms and an immutable selected-file manifest from the publisher/upstream, then—only if those gates pass—freeze a candidate-specific protocol before any target runtime or fixture evaluation. Ticket 08 acceptance remains blocked; all gates and thresholds remain unchanged. Evidence: `api/runtime/adapters/pbr/evidence/marigold-iid-appearance-v1-1-source-gate-2026-09-26.md`.

### Ticket 08 IDArb/LINO source-screen refresh (2026-09-26)

A bounded primary-source screen found no candidate that clears Ticket 08 source/weight identity, complete rights, caller-topology/material-region projection, and local AMD/14 GiB gates. IDArb is a relevant multiview/multilight PBR research system, but its official README still lists inference/checkpoint release as pending and documents CUDA 11.8/A100. LINO-UniPS added PBR evaluation code but expects a separate `lino_pbr.pth`; the identified official Hub checkpoint is tagged normal-estimation only, while declared dependencies include CUDA 12.4 wheels, `spconv_cu124`, and xformers. MatForge-App lacks an albedo head in the inspected source summary and notes a noncommercial ImageNet-1K backbone; the surfaced Materia wrapper reports 24 GB minimum and is not the model publisher. No code/assets/packages/datasets/fixture/truth were accessed. Ticket 08 remains blocked. Highest-value next action is an official immutable PBR checkpoint plus complete terms from a publisher, followed by AMD-compatible dependency qualification and topology mapping before preregistration. Evidence: `api/runtime/adapters/pbr/evidence/ticket08-idarb-lino-unips-primary-source-screen-2026-09-26.md`.
- **Ticket 08 contract audit (2026-09-26):** read-only review of the experimental registered fixed-geometry and latent-normal estimators found no Modly runtime integration and no material-region ID/map in `RegisteredPbrInputs`; outputs are coarse UV-cell maps with topology/correspondence provenance, not assertions bound to a material region. Existing v2/v3 score artifacts show rejected frozen-gate candidates (v3 passes novel-light only and fails base-color, SSIM, roughness, metallic MAE, and metallic-bias thresholds); no new scoring, fixture/truth access, estimator execution, implementation edits, or gate changes occurred. The latest IDArb/LINO-UniPS source screen remains pre-evaluation blocked. Ticket 08 stays acceptance-blocked. Audit: `api/runtime/adapters/pbr/evidence/ticket08-estimator-contract-audit-2026-09-26.md`.
### Ticket 08 2026 frontier candidate refresh (2026-09-26)

A primary-source-only screen covered MatLat (CVPR 2026), NeAR (CVPR 2026), TextureSplat (3DV 2026), and MatSpray. None clears the existing pre-acquisition contract: MatLat explicitly places its checkpoint under CC BY-NC 4.0 and documents CUDA/nvdiffrast; NeAR's available interface is a coupled neural asset/renderer and its documented CUDA 12.x stack explicitly says HIP/ROCm is unsupported; TextureSplat and MatSpray target Gaussian splats rather than caller-topology/material-region-bound PBR maps and document CUDA/OptiX/NVIDIA routes. Immutable full weight/component identity, applicable rights, topology mapping, RX 7900 GRE <=14 GiB, and frozen fixture quality remain unproven as applicable. No repos, packages, weights, data, fixtures, development inputs/predictions, or truth were downloaded/read/run; no scoring occurred. Ticket 08 gates remain unchanged and blocked. Evidence: `api/runtime/adapters/pbr/evidence/ticket08-2026-frontier-candidate-refresh-2026-09-26.md`.

### Ticket 07 Apple-DMS SegFormer-B5 v2 official-source screen (2026-09-26)

A primary-source-only refresh found a recent 57-class dense material classifier with a broadly relevant taxonomy and explicit ambiguity-like labels. It does not clear source eligibility: the author's SegFormer-B5 v2-run2 model card reports Plastic clear 0.3886, Rubber/latex 0.4111, and Metal 0.5541 class accuracy/recall proxies, below the frozen 0.80 per-class floor; these are publisher-set results, not Modly fixture scores. Metal is generic, required abstention behavior is not established, checkpoint/code/dataset rights and immutable file digest remain unresolved, and target ROCm/MIGraphX plus <=14 GiB performance are unmeasured. No code, weights, data samples, fixture inputs, development labels, or held-out truth were accessed. The source screen is `api/runtime/adapters/material-identity/evidence/APPLE_DMS_SEGFORMER_B5_V2_OFFICIAL_SOURCE_SCREEN_2026-09-26.md`. Ticket 07 remains acceptance-blocked; gates unchanged.


### Ticket 05 Decider M-RoPE warning static source diagnosis (2026-09-26)

A static review of the pinned llama.cpp/mtmd source confirms `find_slot: non-consecutive token position` is a diagnostic emitted because M-RoPE image batches process many patch embeddings while advancing a compressed scalar position span. `find_slot()` logs and stores position metadata without clearing recurrent tensors or aborting one-shot decode; its prepare/commit calls explain paired warnings. This does not prove one-shot answer-state correctness. Position-addressed recurrent cache operations remain unverified and potentially unsafe because no M-RoPE span contract is threaded through the scalar cache metadata. No inference/tests, fixture/truth access, policy edits, or warning suppression occurred; Ticket 05 acceptance remains unchanged. Evidence: `api/runtime/adapters/parts/evidence/TICKET05_DECIDER_MROPE_WARNING_SOURCE_DIAGNOSIS_2026-09-26.md`.

### Ticket 07 DMS46 probe-plan correction (2026-09-26)

The exact pinned Apple source contains `ACKNOWLEDGMENTS.txt` (362 lines); the prior report's 401-line count and initial missing-file conclusion are corrected. This project uses the pretrained checkpoint and does not use Apple's DMS training dataset, so the dataset's CC-BY-NC license is not treated as a restriction on this inference use. The checkpoint has a separate Apple model license; whether its personal, non-exclusive grant covers the intended deployment or distribution remains unresolved. Rights to source images remain separate. The probe plan now records this distinction. No weights were loaded, fixture/truth accessed, or gates changed. Evidence: `api/runtime/adapters/material-identity/evidence/DMS46_PINNED_ACKNOWLEDGMENT_RECHECK_2026-09-26.md` and `api/runtime/adapters/material-identity/PROBE_PLAN.md`.

### Ticket 07 DMS46 owner scope and evaluator blocker (2026-09-26)

The owner approved local inference with the pretrained DMS46 model and clarified that the project does not use Apple's DMS training dataset; no DMS dataset-license restriction is applied to this local model evaluation, and checkpoint weights will not be redistributed in this task. Original source-image rights remain separate. The earlier statement that model-use scope was awaiting review is superseded. A candidate-specific preregistration is recorded at `api/runtime/adapters/material-identity/evidence/DMS46_EVALUATION_PREREGISTRATION_2026-09-26.md`. Fixture scoring and truth access remain blocked because the current evaluator is SigLIP2-specific and cannot consume DMS46 dense, topology-bound outputs; target AMD runtime and quality gates are also still unmeasured. No threshold or acceptance criterion changed.

### Ticket 05 owner-approved Decider per-view protocol and native contract (2026-09-26)

The owner approved four independent one-image Decider calls per part and an equal-weight arithmetic mean of their restricted ten-choice probability vectors; no quality threshold or heldout gate changed. Native HIP harness v2 and adapter now bind each ordered view to its own score row and evidence artifact, then emit one part-level assertion. The exact one-view prompt digest is `sha256:df2343bc03b112324e18450c5983ba32f0d52393e480ca9aa7c8c6a215827d2e`. Project API test-venv focus ran 32 tests: 30 passed and 2 skipped because optional semantic evidence runtime dependencies are unavailable. A generated-only four-image native run on AMD Radeon RX 7900 GRE emitted one header plus four ordered score records, passed adapter score validation and probability normalization, and repeated byte-identically (`201a95d9c97e2aca2817a9a3058c791c983fefaa7cb8783d97eec3359ea2fb3b`). The M-RoPE position warnings are retained; this is protocol/runtime evidence only. No semantic fixture or truth was accessed. Ticket 05 acceptance remains open pending development quality and remaining gates.

### DMS46 evaluator integration and gate blocker (2026-09-26)

Independent review found and corrected three evaluator integration issues before fixture access: Modly stage digests cover the exact persisted bytes including newline; the frozen label-blind input manifest has no split field, so development/heldout membership must come from deterministic renderer IDs; and Modly emits one stage per asset, requiring separate 35-stage development and 110-stage heldout batches. The evaluator now has exact stage-byte/geometry/topology checks, canonical durable batch files with SHA-256 sidecars, and a heldout path that verifies both batches and the passing development report before opening truth. The final focused evaluator/process rerun ran 17 tests: 16 passed and one optional-dependency test skipped. Scratch files were confined to `.modly-amd-runtime/tmp` on the work drive. No fixture files, images, truth, model, dataset, or GPU inference were accessed. Evaluator SHA-256 `7a3effb82f264c28fa3e6de452c32e9ec4ce0344de9dbdb4bd5d35b74ac2f8f3`; focused evaluator test SHA-256 `724afa5fec9517b26740dc8a612bc641d840dd742beccde315a3b7da1426e7f8`.

The owner approved the development-only supported-region coverage gate >=0.87, derived from the unchanged heldout conjunction (352 required minus at most two accepted unknown and two accepted ambiguous, over 400 supported). The heldout >=0.80 all-region coverage and >=0.90 OOD abstention gates remain unchanged. A perfect synthetic development contract batch passes this screen (100/100 supported accepted, 100/140 all-region diagnostic); no fixture data or model inference was used.


### Ticket 08 Material Anything target-operation preflight (2026-09-26)

A no-network probe in the immutable project AMD image confirmed RX 7900 GRE access: Python 3.12.3, PyTorch 2.11.0+rocm7.14.0, HIP 7.14.60850, `torch.cuda.is_available() == True`, device zero AMD Radeon RX 7900 GRE, and generated tensor allocation succeeded. Material Anything's active dependencies are absent: `pytorch3d.renderer.TexturesUV` fails import with `ModuleNotFoundError: No module named 'pytorch3d'`; `kaolin` fails with `ModuleNotFoundError: No module named 'kaolin'`. Thus active `TexturesUV`, `kal.ops.mesh.index_vertices_by_faces`, and `kal.render.mesh.rasterize` operations were not reached or executed. No image change, installation, weight/model/data/fixture/truth access, or inference occurred. This confirms the current image is dependency-blocked, not that a ROCm build/port is impossible; further operation proof requires an approved dependency/image change. Ticket 08 status and frozen gates remain unchanged. Evidence: `api/runtime/adapters/pbr/evidence/material-anything-target-ops-preflight-2026-09-26.md`; generated-only run log SHA-256 `7be7a5272a5b8f81ef790b31f5d78fdaf51456b3004301ff778b206a20ba089e`.

### Ticket 05 Decider calibration implementation and process-binding blocker (2026-09-26)

The owner-approved threshold rule is implemented as pure development-only code: candidate thresholds are unique observed aggregate pI/pJ values plus the reject-all boundary; I/J eligibility is inclusive; eligible I/J selection uses highest probability with I winning ties; otherwise A-H uses highest probability with letter-order ties. Feasible pairs must meet both existing OOD abstention floors, then rank by selective accuracy, supported coverage, macro recall, and ascending threshold pair. The policy and generated-only tests do not change the frozen quality gates.

The CPU candidate preparation, registered-process input assembly, and approved Decider-only calibration policy are implemented. The owner approved source-authored masks as label-blind semantic target inputs, with GeoSAM2 segmentation quality independently evaluated and genuine GeoSAM2 provenance/render artifacts required. The actual r4 registered GeoSAM2 batch remains blocked: one RX 7900 GRE attempt failed the strict complete-face partition (face 534 unassigned); two subsequent attempts failed with `No input points or masks are provided for any object`. The serialized batch stopped after those two attempts pending source diagnosis; none generated accepted segmentation output. This is process/inference failure evidence, not semantic quality scoring. Raw Decider outputs have not been committed and development truth has not been accessed. Details and hashes: `api/runtime/adapters/parts/evidence/TICKET05_GEOSAM2_SOURCE_MASK_BATCH_FAILURES_2026-09-26.md`; current batch index SHA-256 `168fc722d9089ac4c74bd7cf022ef00d7ffc76c81271f86acbae13b0def2163a`. No threshold or acceptance criterion changed.

## Ticket 08 Material Anything ROCm source feasibility update (2026-09-26)

A source-only follow-up reviewed the upstream PyTorch3D ROCm build workflow against the pinned Material Anything path. PyTorch3D main now has a build-only _C build/import workflow with PyTorch 2.11.0 + ROCm 7.2.3 on a CPU-only runner; it explicitly runs no tests. This improves source-build feasibility but does not establish GPU operator execution, gfx1100 behavior, or compatibility with the target image's ROCm 7.14. Material Anything remains unpinned to a PyTorch3D commit and omits Kaolin from requirements. Its active path uses PyTorch3D mesh rasterization/interpolation plus Kaolin index_vertices_by_faces and UV rasterize. Kaolin official docs describe full support as NVIDIA CUDA-only; its rasterize implementation dispatches to CUDA extension or nvdiffrast. A torch gather replacement for face indexing appears feasible in principle; a faithful UV-rasterizer bridge remains unimplemented and requires output parity evidence, while PyTorch3D renderer operations still need target validation. No operators, packages, weights, fixtures, observations, or truth were run/accessed; Ticket 08 remains blocked and gates are unchanged. Evidence: api/runtime/adapters/pbr/evidence/material-anything-rocm-source-feasibility-2026-09-26.md.

## Ticket 07 DMS46 parity audit follow-up (2026-09-26)

A truth-free source-to-adapter audit found no confirmed preprocessing mismatch in RGB conversion, max-side-512 LANCZOS resize, ImageNet normalization, dense output interpretation, or face-map canvas binding. A CPU-only synthetic RGB PNG matched exactly between Pillow and OpenCV+BGR-to-RGB for the tested path. The audit corrected stale “shortest side to 512” wording in `api/runtime/adapters/material-identity/PROBE_PLAN.md` and `SELECTION_AND_GATES.md`. It does not overturn the severe failed development score: DMS46 remains rejected under unchanged gates; GPU/ROCm parity and actual rendered face-map reprojection remain unverified; heldout was not accessed. Evidence: `api/runtime/adapters/material-identity/evidence/DMS46_ADAPTER_PREPROCESSING_PARITY_AUDIT_2026-09-26.md` (SHA-256 `ae9e826eb3dcb80f1e437aaa9859f366d76e7ec69b1b73e800707df8449a5334`).

## Ticket 05 empty-proposal fix and scorer ready for hardware retry (2026-09-26)

Modly now hash-pins an AST-checked guard for GeoSAM2's empty automatic-proposal seed views, writes per-view proposal provenance, rejects target masks at the segmenter prompt boundary, and leaves complete-face/topology gates unchanged. A separate development scorer is implemented and source-locked; it validates all 80 candidate/process records before it opens development truth and never accesses heldout. The focused truth-free suite passed 20 tests and 8 subtests; `verify_development_lock()` verified 28 sources. A one-object registered AMD retry did not reach container startup because fresh `newuidmap` capability setup failed (`Could not set caps`); the preexisting stale-boot runroot and both startup-failure logs are preserved on the work drive. No post-fix GPU result, raw semantic commit, or development score exists. Exact SHAs and commands: `api/runtime/adapters/parts/evidence/TICKET05_GEOSAM2_EMPTY_PROPOSAL_FIX_AND_SCORER_2026-09-26.md`.

## Ticket 07 and Ticket 08 candidate-source refresh (2026-09-26)

Ticket 07 source screening found LFM2.5-VL-3B as a new plausible classification lead, but commercial-use license applicability is unresolved and target AMD, <=14 GiB, deterministic protocol, and quality gates remain unmeasured. No checkpoint/code/package or fixture/truth access occurred. Evidence: `api/runtime/adapters/material-identity/evidence/TICKET07_LFM25_VL3B_SOURCE_REFRESH_2026-09-26.md` (SHA-256 `efe176a0fb32f7482a686d5c84270af7d1ef11e994b8954bae78cf3abedd6294`).

Ticket 08's ReLi3D source screen found a non-eligible candidate: the documented pipeline reconstructs new topology instead of binding outputs to caller topology/material regions and documents CUDA-specific native dependencies without an AMD route. No assets/packages/weights/fixture/truth were accessed; no inference occurred. Evidence: `api/runtime/adapters/pbr/evidence/ticket08-reli3d-primary-source-screen-2026-09-26.md` (SHA-256 `a9955a3018d90bc17a24fd10087ee1d3132f058431fb6014039473cc947e5b91`). Neither report changes acceptance gates or advances the acceptance status of 07 or 08.

A follow-up rootless-Podman diagnosis confirmed the current tool process has `CapEff=0`, `NoNewPrivs=1`, and seccomp enabled. The `newuidmap` binary has a setuid capability on disk, but it cannot set capabilities in this process. Nested user-namespace Podman probes also failed on overlay mount permissions and `/run/lock/netavark.lock`. No workflow container/GPU was started and no system privileges or image-store data were changed. A supported rootless Podman execution context is still required for the registered AMD retry; details: `api/runtime/adapters/parts/evidence/TICKET05_GEOSAM2_EMPTY_PROPOSAL_FIX_AND_SCORER_2026-09-26.md`.

## Ticket 07 process import-order correction (2026-09-26)

The registered material-identity processor now preloads installed `typing_extensions` before adding the API root, preventing the empty source marker from shadowing the dependency. The previously failing missing-weight-pin/out-of-workspace test passed with Python 3.12 and host Python 3.14.7. Processor SHA-256: `b38c650702f48f3b6c77862cf1573e1fcb15e43828ec172103a6a9021703fa15`. No quality or heldout gate changed; no heldout data was accessed; Ticket 07 acceptance status is unchanged.

Ticket 08 additional primary-source SVBRDF screening found no candidate that clears the frozen channel, rights, caller-topology/material-region, and AMD runtime gates. No model/package/checkpoint or fixture/truth was accessed; no GPU was used; no estimator code or acceptance gate was changed. Evidence: `api/runtime/adapters/pbr/evidence/ticket08-additional-svbrdf-candidate-screen-2026-09-26.md` (SHA-256 `1ada055b7cbc7c075d7af9597ad32b1fe5b642bc424d2599fc7c4d5620ccb7cb`). Existing Ticket 08 status is preserved.

## Ticket 05 GeoSAM2 failure telemetry follow-up (2026-09-26)

The adapter now atomically records per-seed generated/accepted proposal counts and successful point-registration call counts plus the predictor's actual object count immediately before propagation, preserving diagnostics if propagation fails. Focused synthetic tests: 17 passed; development lock verification: 28 source files passed. The expensive registered GPU retry remains blocked and was not run: the preceding approved retry failed before container startup because `newuidmap` could not set capabilities. This does not qualify inference or candidate acceptance. Exact hashes and commands: `api/runtime/adapters/parts/evidence/TICKET05_GEOSAM2_FAILURE_TELEMETRY_2026-09-26.md`.

## Ticket 05 Decider package and target-runtime recheck (2026-09-26)

The Decider-only local distribution builds and installs in an isolated Python 3.12 venv with Modly's exact pip arguments (`--no-deps --no-build-isolation --disable-pip-version-check`). Exactly one registered semantic-adapter entry point resolves to the installed RECORD-indexed module; its source hash passes the Modly node loader check. The focused Python installation suite passed 4/4 and Electron setup helper tests passed 2/2. Separately, the project-owned ROCm 7.14 container launched with the RX 7900 GRE visible (`torch.cuda.is_available() == True`, AMD Radeon RX 7900 GRE), so the earlier user-namespace startup error is not currently preventing this project-local `--userns=host` runtime path. These are packaging and startup proofs only. Ticket 05 remains open: registered GeoSAM2 segmentation outputs and assembled development workflow bindings must pass, then the Decider must pass its frozen development quality gate before heldout access. Tickets 07 and 08 remain open with no candidate clearing all frozen gates; Tickets 09–13 remain blocked by their dependencies.

## Ticket 05 development-process binding checks (2026-09-26)

The development runner now validates the assembled candidate's seven workflow references before dispatch: StructuredAsset sidecar, geometry, registered topology map, render manifest, camera metadata, part-scoped observation manifest, and segment mapping. It verifies current geometry/topology, registered artifact digests, source-target metadata, and all four selected view bytes/order. Synthetic dispatch/isolation checks passed 8/8, and the v2 frozen-source-lock test passed 1/1. The v2 lock preserves the archived v1 lock byte-for-byte (SHA-256 `92a842da29c81dc5a41b347f180e4ea261954f534ebf8a97490704e4a3c3798c`), retains the candidate and all protocol/cohort/threshold/heldout fields exactly, and binds 28 current source identities with zero mismatch. Commands: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=api .modly-amd-runtime/api-test-venv/bin/python -m unittest api.tests.test_ticket05_runner_dispatch api.tests.test_ticket05_decider_development_runner.DevelopmentRunnerIsolationTests.test_exact_dev_allowlist_selected_before_any_crop_open api.tests.test_ticket05_decider_development_runner.DevelopmentRunnerIsolationTests.test_runner_fails_closed_before_inference_when_process_bindings_are_missing -v`; and `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=api .modly-amd-runtime/api-test-venv/bin/python -m unittest api.tests.test_ticket05_decider_development_runner.DevelopmentRunnerIsolationTests.test_frozen_policy_source_lock_matches -v`. No candidate inference, actual fixture inputs, truth, or heldout data were accessed. The assembled candidate still requires real registered GeoSAM2 outputs and a successful processor completion before semantic development scoring is authorized.

## Ticket 04 current-code registered-run failure (2026-09-26)

The fresh project-local AMD workflow started successfully and completed its registered GeoSAM2 processing attempt for run `a4e6c8f0-1357-4b9d-8f02-2468ace0bdf1`, but failed at final label validation with `GEOSAM2_UNASSIGNED_FACES`: canonical face 58 retained upstream sentinel `999`. The run produced all 12 seed-view proposal/propagation output pairs and a proposal audit, but no segmentation manifest, canonical labels, or committed Structured Asset segmentation update. No Decider inference, development truth, or heldout data was accessed. This supersedes the earlier startup-blocked retry note for this run path. The adapter's strict full-face check remains intact. A source audit found that pinned GeoSAM2 `utils/inference_utils.py::complete_labels()` fills label `0` through adjacency/nearest-three completion but does not treat sentinel `999` as unlabeled. Any proposed repair must be a separately versioned and locked Modly-side derivation, preserve raw labels and unknown confidence, and pass Ticket 04's unchanged frozen-fixture quality, repeatability, topology, provenance, and RX 7900 GRE resource gates before acceptance is current again. Ticket 05 remains blocked on successful registered segmentation; Tickets 07 and 08 remain unaccepted; Tickets 09–13 remain dependency-blocked.

## Ticket 04 unassigned-label policy implementation (2026-09-26)

A separate locked Modly policy now translates only GeoSAM2's raw `-1`/`999` sentinels to the pinned routine's documented unlabeled value, invokes the AST-verified upstream `complete_labels(smooth_type="adjacent", PA=0.02)`, and copies derived values back only at original sentinel positions. Existing assigned labels are preserved; incomplete outputs still fail strict validation. Raw upstream labels, derived labels, and a per-face fill mask are separately hashed in the segmentation manifest and registered as Structured Asset intermediate artifacts. Confidence remains `unknown`. The versioned policy lock is packaged in `api/pyproject.toml`; package inspection confirmed it in the wheel (`.modly-amd-runtime/results/ticket04-unassigned-policy-wheel/modly_decider_semantic_adapter-1.0.0-py3-none-any.whl`, SHA-256 `164ea983a8ec4b58ef74eb07edfbc5b7b6cef36402c15c9236caca3df3edc081`). Independently rerun focused tests passed 8/8 with `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=api .modly-amd-runtime/api-test-venv/bin/python -m unittest api.tests.test_ticket04_geosam2_adapter api.tests.test_ticket04_geosam2_completion_policy -v`. The runtime policy verifier returned lock SHA-256 `8464af562a698ee529b6d7b606ce1fdaab3886b732f569b153dd7e8f7165c177` and module SHA-256 `8cb50369081fe43412b6c2867104745f3e304c002e4cfc6925c30e015fd06779`; the pinned upstream AST digest independently matched `68c0019ce02cc78e1350e1cee1966d1cbd56ba5a28b7d7660fb12b22f455cd80`. This is synthetic/software/package evidence only. No model inference or fixture/truth scoring occurred. Ticket 04 target qualification remains open.

## Ticket 04 current-code repeatability revalidation failure (2026-09-26)

The full frozen known-truth workflow input was copied unchanged into two new isolated workspaces (geometry SHA-256 `71ba339507486660bc8e4733dfadfc1575f0c6079732b6bdc3fe22e0c47f833b`; source sidecar SHA-256 `c577a31bfe2063a7612b38f48a21d6f533bbb295aa86126c33dc2cbcc635925f`) and processed in separate RX 7900 GRE workflows: `scripts/modly-amd-runtime.sh workflow-geosam2 .modly-amd-runtime/results/ticket04-requal-20260926-run1 fixture.glb StructuredAssets/8c7f9bd6-bd57-4839-ab19-53fdf452922e.structured-asset.json 73b2d9ce-59ee-4d4f-9cc3-61a67b5d36a1` and the same command with workspace `...-run2` and run ID `e1a3e501-9e2b-4580-86d2-a6ec59536a24`. Both emitted registered complete face partitions with valid artifact digests, RX 7900 GRE/gfx1100 identity, unknown confidence, zero overlap by strict partition validation, and no sentinel; both raw fill masks are zero. Their render manifests are byte-identical (SHA-256 `94f17e25ecb343fcb444ebd98e0a1187b1011074924633c6cbbda707ceb6fe26`) but the raw labels differ on 768/1,536 faces and canonical membership differs on 46/1,536 faces. Run 1 produced three regions (768/722/46 faces; 15,068,037,120 peak reserved bytes); run 2 produced two (768/768; 14,417,920,000 peak reserved bytes). Proposal audit differs: run 1 accepted proposal counts were `[2,5,3,4,2,4,1,3,3,5,1,3]`; run 2 `[2,5,3,4,2,3,1,2,0,0,0,3]` with empty seed views 8, 9, and 10. Both target peaks were below the reported 17,163,091,968-byte board memory. Because repeatability failed, the known-truth arrays were not opened and no quality score was produced. The current target completion branch remains unexercised because these inputs contained zero sentinels. Ticket 04 and dependent Ticket 05 remain blocked; do not update the Ticket 05 v2 source lock until Ticket 04 passes unchanged acceptance gates.

An independent read-only audit confirmed the repeat pair had identical geometry, source sidecar, source/target mesh, face-correspondence digest, topology revision, and render manifest. All 48 stage-artifact references in each output sidecar resolve and match their recorded digests; each partition mapping covers all 1,536 faces exactly once at the same topology revision. Run 1 and run 2 used 14.033 GiB and 13.428 GiB peak reserved, respectively, on the reported 16 GiB RX 7900 GRE. The mismatch is therefore an inference repeatability failure, not an input or artifact-integrity mismatch. An opt-in digest/count-only first-divergence trace and a synthetic sentinel-branch test are implemented; they do not read truth data or alter Ticket 04 acceptance thresholds.

## Ticket 04 diagnostics and synthetic completion coverage (2026-09-27)

The sentinel-completion branch now has a synthetic artifact test using the actual AST- and hash-pinned GeoSAM2 completion routine. `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=api .modly-amd-runtime/api-test-venv/bin/python -m unittest api.tests.test_ticket04_geosam2_completion_artifacts -v` passed 1 test in the project Ticket 04 container. It verifies raw-label preservation, derived labels/fill mask, unknown confidence and provenance, and serialized manifest digests; it does not exercise target hardware. Opt-in diagnostics are implemented through `MODLY_GEOSAM2_DIAGNOSTICS=1` and add only per-view mask/coordinate digests and counts to the proposal audit. The default path is unchanged. The adapter and telemetry regressions passed 7 tests with `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=api .modly-amd-runtime/api-test-venv/bin/python -m unittest api.tests.test_ticket04_geosam2_adapter api.tests.test_ticket04_geosam2_diagnostic_telemetry -q`. NMS score/box/kept-index digests and exact IoU/stability subfilter counts are not captured; use the available per-view trace first and add safer pinned-source instrumentation only if it cannot locate the first divergence. No target rerun or quality scoring has occurred with this telemetry, and Ticket 04 remains unaccepted.

### Ticket 04 larger-mesh repeat study (2026-09-27)

Two completed runs (`b6f57f83-6dd1-4d6d-bfb2-14e0d8033cad`, `5aa83b9a-f86a-437e-8a11-344e9f8f4b22`) used the same 46,180-face TripoSR Flamingo mesh, face map, render manifest, GeoSAM2 code digest, weights, settings, seed 42, and RX 7900 GRE runtime. Both emitted `done`, saved valid 10-region partitions covering every face, and passed integrity checks for all 48 stage artifacts. On common raw-assigned faces, regions match on 99.9895% (28,628/28,631; ARI 0.999875); upstream unassigned faces were excluded from this raw comparison. After unassigned-face completion, membership-matched agreement is 98.1052% (45,305/46,180; ARI 0.937262), with 875 face assignments different and 10 fill-mask faces different. All 12 accelerated-mask hashes and coordinate-sample signatures match. Accepted proposals differ only by one in views 0 and 6 (184 versus 186 total). No fixture truth was accessed and no quality score was produced.

A third same-input/code/settings/seed attempt (`fa73c52c-eceb-4411-a289-4b1a87088da5`) completed proposal collection but failed `GEOSAM2_NO_FACE_LABELS`: it accepted 32 proposals in view 0 and zero in views 1–11, so it has no final face labels or segmentation manifest and cannot enter the face-agreement metric. The view-0 image-mask and sampled-coordinate digests match the successful baseline, but the generator returned 34 masks versus 35 and 36; the saved evidence first detects a result-count difference, not the internal score/rejection cause. Its accelerated-mask digests match the successful baseline through views 0–2 and coordinate signatures through views 0–1. A further same-input/code/settings/seed attempt (`c1a37e09-84bc-46ee-9c4d-f0ddae5ba303`) also failed `GEOSAM2_NO_FACE_LABELS`, with 35 accepted proposals in view 0 and none in views 1–11; its view-0 mask and coordinates match the completed baseline although the generator returned 37 masks. Both failed runs used the same adapter digest and have no final face labels for a valid agreement comparison. The prior `d7a0140c-f0fe-49ce-a038-214a9922aaa2` attempt also failed before final labels; its observed proposal counts differ sharply, and its old error message does not preserve the cause. The completed run `f8c6a77e-7090-4713-bb7d-3ddcedadeb1b` is exploratory only because its adapter-code digest differs; against `b6f57f83`, completed-label agreement was 86.9251% (ARI 0.867152).

Each completed same-code workflow emitted a valid result and registered all output files but the wrapper returned status 1; no persisted run log explains this discrepancy. Both completed runs reported 14,790,649,344 bytes peak allocated and 16,418,603,008 / 16,651,386,880 bytes peak reserved (roughly 15.3 / 15.5 GiB) and took about 10.4 / 11.1 minutes. The completion-failure diagnostic now includes the raw unassigned-face count and a chained cause capped at 512 characters; its focused regression test passed. Full evidence and explicit study limits are recorded in `api/runtime/adapters/parts/evidence/TICKET04_FLAMINGO_REPEATABILITY_PANEL_2026-09-27.md` and `.modly-amd-runtime/results/ticket04-variation-panel/`. Two completed outputs and multiple failures on this one mesh still do not estimate typical variation across shapes, and Ticket 04's exact-label gate still fails.

### Ticket 04 distinct-shape pilot (2026-09-27)

A first current-code chair attempt (`ecd8b8c1-3d42-4a28-8fae-a5c662f24b3e`) used the registered 83,732-face TripoSR chair mesh (SHA-256 `2efaf77b9d4eaef7342badd4110bf23e3a5c85af0301b0b2359f81f1619e7e3d`), fixed seed 42, and the same RX 7900 GRE runtime/model route. All 12 views completed, but only the starting view had 17 accepted proposals; the other 11 had none. The run failed with `GEOSAM2_NO_FACE_LABELS` and produced no final labels, manifest, or quality score. A second same-input/code/settings/seed chair run (`3188b1f9-26b7-46fa-8573-cb261058e68d`) found proposals in six views but still left all 83,732 raw labels at sentinel `999`; completion failed because there were no region labels to extend. The proposal patterns differ between the two attempts, but neither has face labels for an agreement comparison. Full attempt records: `.modly-amd-runtime/results/ticket04-variation-panel/chair-run-ecd8b8c1-3d42-4a28-8fae-a5c662f24b3e.json` and `.modly-amd-runtime/results/ticket04-variation-panel/chair-run-3188b1f9-26b7-46fa-8573-cb261058e68d.json`.

Flamingo and chair are separate generated assets with different geometry and face counts. Each is considered on its own topology revision; no labels or face indices are compared across those assets. The frozen repeatability gate applies when the same mesh bytes are segmented again with the same adapter, weights, parameters, and seed.

### Ticket 04 teapot topology run (2026-09-27)

The registered 104,454-face teapot (`658a5fbd…d6c2ab2`) completed with a valid four-region partition, all faces assigned, mappings bound to its own topology revision, and all 48 registered artifact digests verified. Its raw upstream output contained 235 unassigned sentinels; the separate fill mask records those 235 completed faces. RX 7900 GRE peaks were 14,602,411,008 allocated and 16,691,232,768 reserved bytes (13.60/15.55 GiB); model load plus inference took 354.3 seconds. Confidence is `unknown`; no truth or quality score was accessed. Two further runs of the identical mesh/code/settings/seed failed before final labels: one failed with `GEOSAM2_NO_FACE_LABELS` after proposals only in its starting view; the third (`0b670b96-9416-4fcb-979b-5b299f2ec448`) also had proposals only in view 0 (54 accepted at Modly audit, 55 returned by generator) and the upstream path returned no `face_label`. No failed attempt enters a face-label comparison. The latest proposal audit is under `.modly-amd-runtime/results/ticket04-variation-panel/teapot-repeat-03/`; prior attempt records are `.modly-amd-runtime/results/ticket04-variation-panel/teapot-run-9e00daf7-d664-4a7f-82e7-404fdd94a8a3.json` and `.modly-amd-runtime/results/ticket04-variation-panel/teapot-run-0b16cd6a-043c-4f8d-a2bc-93c5cf2b95fb.json`.

Opt-in adapter diagnostics record total, assigned, distinct, and `-1`/`999` face-label counts if the pinned inference returns labels, plus a payload-free explicit no-label state and the accepted non-prompt seed views on failure. The pinned upstream source is available read-only in the project source cache; inspection plus the third teapot repeat shows that only starting view 0 had accepted proposals, and upstream skips that prompt-seed view during final face lifting while the empty non-prompt views are skipped by Modly's frozen policy. No seed reached lifting in that run. The source and output audit still do not expose per-object post-filter survivors or projected point-to-face samples. Focused telemetry and audit tests passed 5/5 in the project Python 3.12 environment. Segmentation behavior and acceptance gates are unchanged; current-code fixture quality/coverage, same-input repeatability, completion-branch target proof, and resource-recovery evidence remain open.

Ticket 04 checklist wording was corrected to match FINAL_AUDITED_SPEC.md: on changed topology, mappings/IDs are invalidated unless proven correspondence establishes the same region, in which case identity may be retained. Independent generated meshes may differ in face count and are evaluated on their own topology.

A subsequent exploratory teapot workflow (`446fdb8a-b304-456e-92af-05e2c6839be4`) was stopped after six seed views because the adapter file changed while its process was running, so the imported-code identity and recorded digest could not be reconciled. It emitted no final event/Structured Asset and is excluded from every acceptance or repeatability comparison. Its partial files are retained under `.modly-amd-runtime/results/ticket04-variation-panel/teapot-repeat-04/` as interrupted diagnostic artifacts only.

The pinned inference source skips final lifting for its automatic prompt-seed view. An additive, unselected candidate policy now AST-matches that guard and allows that view to reach the first lift only for automatically generated, unprompted seed masks. It is opt-in through `MODLY_GEOSAM2_PROMPT_SEED_LIFT=1`; default behavior is unchanged. The versioned lock/module are `api/runtime/adapters/parts/GEOSAM2_PROMPT_SEED_LIFT_POLICY_LOCK.v1.json` and `geosam2_prompt_seed_lift_policy.py`; synthetic plus adapter-lock tests pass 6/6, and the combined transform compiles the actual pinned inference source. No target inference has used the candidate; current thresholds and acceptance remain unchanged. Before selection it requires target workflow, known-truth quality/coverage/overlap, exact same-input repeat, and VRAM proof.

Latest combined focused run: the GeoSAM2 diagnostic/count/no-label tests and the opt-in prompt-seed candidate tests passed 12/12 in the project Python 3.12 environment. Exact command: `TMPDIR="$PWD/.modly-amd-runtime/tmp" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=api .modly-amd-runtime/api-test-venv/bin/python -m unittest api.tests.test_ticket04_geosam2_diagnostic_telemetry api.tests.test_ticket04_geosam2_diagnostic_audit_return api.tests.test_ticket04_geosam2_prompt_seed_lift_policy -v`. A separate source check matched and compiled the composed policy against pinned `inference.py` (29,354 characters). This is focused contract evidence, not Ticket 04 acceptance.

## Ticket 08 MVInverse candidate screen (2026-09-26)

MVInverse is a new multiview intrinsic-decomposition lead with albedo, metallic, roughness, normal, and shading outputs, but it fails Ticket 08's pre-acquisition gates: official inference documents CUDA 11.8 and `cuda`/`cpu` devices without AMD/RX 7900 GRE evidence; its per-view image-map interface does not establish caller-topology/material-region or UV binding; immutable code/weight identity and complete dependency rights remain unresolved. Rejected before acquisition or evaluation. No package, model, fixture, or truth was accessed; no gates changed. Evidence and official primary-source links: `api/runtime/adapters/pbr/evidence/ticket08-mvinverse-primary-source-screen-2026-09-26.md` (SHA-256 `34bd20f7b620c59ce692edc6bcb388d4974ecb038c4f846749e19eab6f465300`).

## Ticket 07 VLMaterial source screen (2026-09-27)

One distinct official-source lead, VLMaterial, was screened without acquiring or running assets. The official project is for procedural material-program generation and rendering rather than topology-bound identity classification; its docs do not establish the frozen taxonomy, calibrated unknown/ambiguous outputs, or bare-versus-painted-metal abstention. Its checkpoint ZIP has no immutable file identity in the inspected page, and the documented LLaVA-NeXT 8B / 48 GB VRAM recommendation, 8×H100 scripts, and CUDA setup do not support an RX 7900 GRE <=14 GiB route. The project lists MIT code/pretrained-weight terms and separately CC BY-NC 4.0 dataset terms; provenance still needs complete review for any reconsideration. Rejected before acquisition; no fixture/truth access and no gates changed. Evidence: `api/runtime/adapters/material-identity/evidence/TICKET07_VLMATERIAL_OFFICIAL_SOURCE_SCREEN_2026-09-27.md`.

## Ticket 08 PBRnxt official-source screen (2026-09-27)

PBRnxt is a distinct 2D texture enhancement lead, but its documented input is one existing flat diffuse texture and its outputs omit metallic. The source has no caller-mesh/material-region correspondence, and its checkpoint has no pinned digest or weight-specific terms in the reviewed interface. The repository documents a CUDA-oriented PyTorch environment; ROCm execution and RX 7900 GRE memory/latency remain unknown. Normal and displacement were not relabeled as bump/height. It was rejected before acquisition or evaluation; no code, packages, weights, Modly assets, fixture, or truth were accessed, and no gate changed. Evidence: `api/runtime/adapters/pbr/evidence/ticket08-pbrnxt-official-source-screen-2026-09-27.md`.

## Ticket 04 different-topology mapping regression (2026-09-27)

Added `api/tests/test_ticket04_different_topology_mappings.py`: it imports generated one-face and two-face GLBs independently, creates valid mappings bound to each asset's own topology revision, and verifies the distinct face counts/revisions without comparing cross-asset labels. The focused test passed (1 test). Exact command: `TMPDIR="$PWD/.modly-amd-runtime/tmp" PYTHONDONTWRITEBYTECODE=1 .modly-amd-runtime/api-test-venv/bin/python -c 'import typing_extensions, sys, unittest; sys.path.insert(0, "api"); unittest.main(module="api.tests.test_ticket04_different_topology_mappings", argv=["test_ticket04_different_topology_mappings", "-v"])'`. This confirms separate topology handling only; it does not pass the model's Ticket 04 quality, coverage, same-input repeatability, or RX 7900 GRE gates.

## Additional dependency-ready candidate screens (2026-09-27)

Ticket 07 screened Columbia CAVE's hierarchical material-recognition system from its official project page, paper and author profile. Its coarse-label confidence behavior does not establish the frozen five-role taxonomy or calibrated `unknown`/`ambiguous` abstention; the reported Matador-C1 evaluation omits glass, plastics and paint and combines metals. No immutable executable checkpoint/terms, caller-topology interface, or RX 7900 GRE route were established. Rejected before acquisition. Report: `api/runtime/adapters/material-identity/evidence/TICKET07_HIERARCHICAL_MATERIAL_RECOGNITION_SOURCE_SCREEN_2026-09-27.md`.

Ticket 08 screened MatForge-App's official repository and v1.0 release. Its published outputs omit required base color/albedo, its single-image interface has no topology/material-region binding, checkpoint identities/complete terms are unresolved, and it documents CUDA rather than a qualified AMD path. Rejected before acquisition; no assets or truth accessed. Report: `api/runtime/adapters/pbr/evidence/TICKET08_MATFORGE_APP_SOURCE_SCREEN_2026-09-27.md`.

Ticket 07 screened Ultralytics YOLOE from official model documentation and licensing sources. Its instance masks make material-crop classification structurally plausible, but it is a general object segmenter; published LVIS mAP is not evidence for any frozen material/abstention gate. No topology-binding, immutable checkpoint/terms, or RX 7900 GRE route was established; official deployment guidance names NVIDIA 4–8 GB VRAM. Rejected before acquisition/evaluation; no assets, fixtures or truth accessed. Report: `api/runtime/adapters/material-identity/evidence/TICKET07_YOLOE_OFFICIAL_SOURCE_SCREEN_2026-09-27.md`.

Ticket 08 screened the official Learned Gradient Descent single-image SVBRDF estimator. Its diffuse/specular/roughness/normal output has no metallic estimate, and its single-image output is not bound to Modly material regions or topology. The model checkpoint has no immutable digest/terms and no AMD, <=14 GiB, or frozen-quality evidence. Rejected before acquisition. Report: `api/runtime/adapters/pbr/evidence/ticket08-learned-gradient-svbrdf-screen-2026-09-27.md`.

Ticket 08's additional official-source refresh found RGB-X and TextureWorks ineligible before acquisition. RGB-X has mutable weight IDs without established checkpoint terms, an NVIDIA-only environment, and no caller-topology contract; TextureWorks uses NVIDIA PTX image heuristics and does not estimate albedo. Both lack RX 7900 GRE and frozen channel/novel-light evidence. Full source findings: `api/runtime/adapters/pbr/evidence/ticket08-rgbx-textureworks-source-screen-2026-09-27.md`. No assets or truth were accessed and no thresholds changed.

## Additional dependency-frontier rechecks (2026-09-27)

Ticket 07's Molmo2-4B lead remains uncleared on intended use. Ai2 describes the checkpoint as Apache-2.0 while stating research/education intent and noting third-party training data with academic/non-commercial research restrictions. Official wording does not resolve whether those data restrictions attach to downstream checkpoint use; Modly's general public product intent is not established as sole-purpose scientific research or education. This is uncertain, not a claim of express prohibition, and does not resolve quality or AMD eligibility. The Ticket 07 issue records the reviewed official sources. No model or truth data was accessed.

Ticket 08's Material Anything dependency path was rechecked against the exact project image without acquiring weights or occupying the target GPU. PyTorch3D and Kaolin are absent. The active model source calls PyTorch3D mesh loading/rasterization/attribute interpolation and Kaolin UV indexing/rasterization. Kaolin documents full support around NVIDIA CUDA and no route for this PyTorch 2.11/ROCm 7.14 image. PyTorch3D has an upstream ROCm 7.2.3/PyTorch 2.11 build-only workflow but no target operation or ROCm 7.14 result. This leaves a safe isolated PyTorch3D build/test as a possible next feasibility check, while Kaolin rasterization still needs a separately validated AMD replacement or the candidate must be rejected. No weights, packages, or target GPU tests were used. The Ticket 08 issue records the primary sources and next allowed check.

The opt-in Ticket 04 prompt-seed-lift feasibility run (`0340842b-df11-4f94-b43b-9a85902fa92c`) completed on the RX 7900 GRE and failed closed. Its locked audit shows 11 accepted proposals from view 0 and none in views 1–11. The upstream result had 104,454 faces and every label remained sentinel `999`; Modly returned `GEOSAM2_UNASSIGNED_COMPLETION_FAILED` because there was no region label for the completion step to extend (`ValueError: max() iterable argument is empty`). No final manifest or Structured Asset was produced. The candidate changed registration behavior but failed the label/coverage gate; there is no quality, repeatability, or VRAM peak acceptance evidence. Exact module and policy-lock hashes match the frozen candidate snapshot; the audit is at `.modly-amd-runtime/results/ticket04-variation-panel/teapot-prompt-lift-candidate-01/StructuredAssets/runs/0340842b-df11-4f94-b43b-9a85902fa92c/geosam2-inference/proposal-audit.json`. The candidate remains unselected and all acceptance gates remain unchanged.

Ticket 07 screened OpenMR tactile material recognition, Microsoft RegionCLIP, and MaskTerial; all three were rejected before acquisition. OpenMR uses touch sensors, RegionCLIP's published work is object detection, and MaskTerial targets microscopy of exfoliated 2D materials. None establishes Ticket 07's frozen material taxonomy, calibrated abstention behavior, and AMD gates. Ticket 08 screened DiffusionRenderer and rejected it before acquisition: its image-space outputs are not bound to caller topology/material regions, official memory use exceeds the 14 GiB gate, and no AMD route is documented. Full reports and primary-source links are recorded in the respective ticket issue files. No thresholds changed and no model/data artifacts or truth were accessed.

Ticket 08 also screened DiffMat. Although it emits PBR channels, it optimizes a supplied Substance graph to match a texture image rather than recovering calibrated caller-mesh/material-region properties. The official source warns complex scenes may need at least 16 GB VRAM, documents CUDA without AMD qualification, and states noncommercial licensing with Adobe Substance dependencies. Rejected before acquisition/evaluation. No files, packages, graphs, fixtures, truth, or GPU were accessed and no gates changed; report is linked in the Ticket 08 issue.

Ticket 08 screened Meshy AI Texturing's hosted service as a direct-mesh alternative. The official web app supports common mesh uploads and emits PBR texture channels, but no local weights/runtime, immutable model identity, topology/material-region provenance, or AMD memory evidence is published. The hosted-only path fails the audited local/no-required-cloud route. Rejected before account or asset use; no upload or gate change occurred. Report is linked in the Ticket 08 issue.

### Ticket 04 prompt registration follow-up (2026-09-27)

The v2 completion lock and prompt-diagnostics package data are now verified by project-runtime tests. One run stopped before inference because the verifier still expected the v1 schema; that check has been corrected and tested. The subsequent opt-in teapot run (`9f76dc6b-0645-4d9c-9a88-92bd854e1173`) registered 12 starting-view point prompts and none in views 1–11. Across 156 propagated-logit summaries, all values were finite and exactly `-1024.0`; every summary had zero positive pixels. All 104,454 face labels remained `999`, so completion failed closed without a final manifest or Structured Asset. This locates foreground absence before 3D lifting but does not explain tracker behavior or validate prompt locations. Its locked audit is under `.modly-amd-runtime/results/ticket04-variation-panel/teapot-prompt-lift-diagnostic-02/StructuredAssets/runs/9f76dc6b-0645-4d9c-9a88-92bd854e1173/geosam2-inference/proposal-audit.json`. The project GeoSAM2 image passes 17/17 completion/prompt-policy/audit tests; the focused diagnostic and seed-lift regressions pass 31/31 in the project API test environment. Ticket 04 remains acceptance-open, Ticket 05 waits on it, and no same-mesh repeat-label comparison, fixture-quality score, or memory gate passed. Different generated assets continue to be evaluated on their own topologies; their face counts and labels are not compared.

Packaging follow-up: `python -m pip wheel --no-deps --no-build-isolation --wheel-dir .modly-amd-runtime/package-check ./api` built wheel `modly_decider_semantic_adapter-1.0.0-py3-none-any.whl` (SHA-256 `5e153648c56a7ebcc70b2249868e6dda3a3f549fb8a6e5140185a8e415578dd4`). Direct archive inspection confirmed both completion policy lock versions and the prompt-registration diagnostics lock are present. This validates packaging only, not installation/runtime acceptance.

### Ticket 04 run follow-up and explicit topology handling (2026-09-27)

User clarified that generated assets may vary, including in face count. This matches the audited contract: each generated model is evaluated on its own topology revision; face-index comparison is only meaningful for identical mesh inputs. No cross-asset face comparison or shared face-count assumption is used. The same-mesh Flamingo pair remains descriptive evidence of stochastic variation (875 of 46,180 final face assignments differed); it does not pass the fixed exact-repeat acceptance check.

The opt-in diagnostic run `f072ca48-09d9-4b7b-91c0-9f5979ae8b2a` produced a valid one-region Structured Asset on the 104,454-face teapot topology. All 48 registered artifact digests verified. GeoSAM2 directly assigned 8,081 faces; the separate completion stage filled 96,373 sentinels, and confidence is `unknown`. No truth was scored. RX 7900 GRE peak allocated/reserved memory was 13,958,460,928 / 16,271,802,368 bytes; load plus inference took 727,954.1 ms. This single run does not estimate variation and does not pass Ticket 04's quality/repeat gates. Full artifacts are under `.modly-amd-runtime/results/ticket04-variation-panel/teapot-prompt-lift-diagnostic-02/StructuredAssets/runs/f072ca48-09d9-4b7b-91c0-9f5979ae8b2a/`.

The video-index policy regression module now covers the pinned inference-mode decorator, earliest committed or temporary conditioned seed, preflight-before-remap behavior, and fail-closed missing-seed handling. The combined video-index, diagnostic-audit, and prompt-registration suite passes 28/28 with the wrapper-level regressions included. Ticket 04 remains acceptance-open; Ticket 05 remains blocked behind it.

The same 28-test focused regression suite also passed inside the project-owned `localhost/modly-amd-geosam2:ticket04` image with networking disabled and without GPU device mounts. Exact command: `XDG_RUNTIME_DIR="$PWD/.modly-amd-runtime/run" podman --root "$PWD/.modly-amd-runtime/storage" --runroot "$PWD/.modly-amd-runtime/run" run --rm --network=none --read-only --userns=host --volume "$PWD/api:/modly/api:ro" --volume "$PWD/.modly-amd-runtime/source:/modly/.modly-amd-runtime/source:ro" --volume "$PWD/.modly-amd-runtime/tmp:/tmp:rw" --env PYTHONPATH=/modly/api:/opt/rocm/lib --env PYTHONDONTWRITEBYTECODE=1 --workdir /modly localhost/modly-amd-geosam2:ticket04 python -m unittest api.tests.test_ticket04_geosam2_video_index_policy api.tests.test_ticket04_geosam2_diagnostic_audit_return api.tests.test_ticket04_geosam2_prompt_registration_diagnostics -v`.

An additional teapot diagnostic run (`6d4a24ab-e1ab-47fa-9301-89f6c493b6c8`) emitted a valid Structured Asset on the same 104,454-face topology; all 48 registered artifacts verified. It assigned 9,892 faces upstream and completed the remaining 94,562, yielding one final region covering the mesh with `unknown` confidence. The prior `f072...` run assigned 8,081 faces upstream and also completed to one full-mesh region. Canonical output label bytes match, but raw labels/fill masks differ. This pair is not a controlled same-code repeat because the adapter-code digest changed between the runs (`8ba80e…` vs `ffc5b3…`); mesh, render manifest, weights, seed, and listed thresholds match. No truth was scored. The second peak was 14,624,537,088 allocated / 16,670,261,248 reserved bytes, and load plus inference took 609,426 ms. The workflow emitted a `done` event with a valid sidecar but returned exit code 1; the discrepancy is unresolved.

That current-code same-mesh rerun finished without labels: raw output had all 104,454 faces at sentinel `999`, and no foreground survived the final stability filter, so completion failed closed with `GEOSAM2_UNASSIGNED_COMPLETION_FAILED`. The immediately prior run `6d4a...` used the same `geosam2.py` digest (`ffc5b3…`), exact same inference input files (2/2), byte-identical render bundle (39/39 files), topology, frame-index policy, prompt-lift/completion policies, model source and seed/settings, and emitted a valid one-region Structured Asset after filling 94,562 raw sentinels. This is a same-code, same-input success-versus-failure pair on the diagnostic teapot, not a quality score; no truth was read. The failed run's audit has 12 view records with proposals on all views, but proposal registration alone did not produce a label partition. Its failure record did not preserve a final sidecar manifest with the adapter-code digest or full numeric settings. Ticket 04 remains unaccepted.

A third attempt with the same current adapter, input, model, seed, and settings (`439c5d8f-c871-48b5-8a11-9e27f2cdf0a3`) matched both prior runs' two input and all 39 render-file digests, then failed with all 104,454 labels at sentinel `999`; only view zero had accepted proposals. In the controlled three-run teapot sample, one run completed a single-region result by filling most faces and two failed before final labels. The sample confirms current-path reliability is a blocker; it does not establish expected quality or a broad variation rate.

The fourth same-code, same-input teapot attempt (`a7248a91-2363-442a-bde2-ae03ea546e08`) also failed with all 104,454 labels unassigned. Its two input-file and all 39 render-file digests match the other attempts; proposals were accepted only in the starting view. The fifth run is now complete and also failed. Independent hashing confirmed all five attempts used byte-identical input and render files, and the current adapter source digest matches the one completed run's manifest. The five-run sample is one completed one-region result (filled from 94,562 unassigned faces) and four failures before a final partition; proposal patterns differed across these same-input attempts. This is useful reliability evidence for one teapot and one opt-in candidate path, not a general expected failure rate or quality score. None of the outputs were truth-scored.

Ticket 07 also screened Factral's RGB-D spectral segmentation and Seeing Through Touch. The former does not establish the frozen identity labels or calibrated abstention behavior; the latter requires tactile inputs and produces correspondence/localization rather than named material identity, with CUDA-only setup and incomplete model pin/terms. Ticket 08 also screened NFPLight; like EBREnv it requires specialized near/far flat-surface captures, lacks metallic and height outputs, and does not bind maps to Modly topology. All candidates were rejected or held before acquisition; no fixture/truth access or gate changes occurred. Reports: `api/runtime/adapters/material-identity/evidence/TICKET07_FGMATSELECTION_OFFICIAL_SOURCE_SCREEN_2026-09-27.md`, `api/runtime/adapters/material-identity/evidence/TICKET07_BEYOND_APPEARANCES_RGBD_SPECTRAL_SOURCE_SCREEN_2026-09-27.md`, `api/runtime/adapters/material-identity/evidence/TICKET07_SEEING_THROUGH_TOUCH_OFFICIAL_SOURCE_SCREEN_2026-09-27.md`, `api/runtime/adapters/pbr/evidence/ticket08-dgrsise-source-screen-2026-09-27.md`, `api/runtime/adapters/pbr/evidence/ticket08-ebrenv-official-source-screen-2026-09-27.md`, and `api/runtime/adapters/pbr/evidence/ticket08-nfplight-official-source-screen-2026-09-27.md`.

### Ticket 04 predictor-output diagnostics and ready-frontier research (2026-09-27)

A code/evidence audit of the all-sentinel GeoSAM2 attempts found two different failures: some runs already returned all-negative logits from registration/propagation, while another produced positive propagated masks that later filters removed. For start view 0, the locked frame-index remap is a no-op. Current diagnostics do not summarize numeric registration outputs/object scores when keys are present, and can silently leave values null for unsupported tensor types. No inference threshold or completion behavior is justified for change. Next targeted source work is to make payload-free registration/preflight summaries robust (including bfloat16) and explicitly record summary failure reasons, then use those fields to separate prompt rejection from tracker loss before another rerun. Evidence was read from saved payload-free audit files and source; no truth or fixture labels were opened.

Ticket 08's CPU synthetic light-excitation probe finite-differenced the existing GGX forward model at one arbitrary parameter point. Across four synthetic views, rotating a three-light rig per view reduced the local Jacobian condition number from 19.19 to 12.27 (rank 5 in both scenarios). This is a forward-model identifiability hint only, not an estimator quality score or permission to re-score rejected candidates. The registered path accepts one shared training-light set; using this idea requires genuine calibrated per-view light inputs. Script/report: `api/runtime/adapters/pbr/research/ticket08_light_excitation_conditioning.py` and `api/runtime/adapters/pbr/research/ticket08-light-excitation-conditioning-2026-09-27.md`. No fixture, truth, fit, scorer, model, or GPU was used.

Ticket 07's SMARC source screen found its reported 85.10% accuracy is for Touch-and-Go categories that do not match Modly's five labels. The inspected sources do not establish calibrated unknown/ambiguous outputs, dense topology-bound material-region assertions, pinned deployable weights/terms, or AMD memory qualification. Rejected before acquisition/evaluation; no model/data/fixture/truth was accessed. Report: `api/runtime/adapters/material-identity/evidence/TICKET07_SMARC_MINIMAL_CUE_OFFICIAL_SOURCE_SCREEN_2026-09-27.md`. No ticket gates changed.

### Ticket 04 numeric predictor diagnostics repair and Ticket 07 source follow-up (2026-09-27)

Ticket 04 diagnostic-only code now reduces registered SAM output and stored per-object mask/object-score values to numeric summaries, retries unsupported NumPy tensor dtypes through transient float32 conversion, and records bounded type-only failure reasons. It also records preflight output summaries before and after predictor preflight without persisting IDs, prompt coordinates, tensors, or exception text. No model inputs, thresholds, labels, or completion behavior changed. Focused project test passed 13/13; the prompt-registration diagnostics v3 lock verifier passed. Module SHA-256 `92b9a8413c4f698800407c5b037c54d25a8ab3d44569d0c639b2cf9eef9cd710`; lock SHA-256 `fee5214efa4f670f212b799bc349cbb7f814213712c6247d55cc53c713459bec`. Changes are diagnostic support only and do not pass Ticket 04 acceptance.

Ticket 07's second independent source audit reviewed the existing MINC fine-tune and generic SigLIP2 route. The fine-tune lacks a required class and calibrated abstention/dense region outputs, reports macro-F1 below Modly's unchanged floors, has unresolved source-photo rights, and has no target AMD resource proof; generic SigLIP2 already failed development gates on CPU. Neither warrants target-GPU use. No weights, dataset, fixture/truth, or GPU were used. Report: `api/runtime/adapters/material-identity/evidence/TICKET07_SIGLIP2_MINC_RUNTIME_PROVENANCE_AUDIT_2026-09-27.md`.

Ticket 07 also screened OvarNet as an open-vocabulary region-attribute route. Its box-crop attribute head and generic `metal` label do not establish the frozen five surface identities, bare-versus-painted metal, calibrated abstention, or topology-bound mask evidence. No immutable inference bundle/license or AMD target evidence was found; bounding-box crops may include out-of-region pixels. Rejected before acquisition/evaluation; no assets, data, fixtures, truth or GPU accessed. Report: `api/runtime/adapters/material-identity/evidence/TICKET07_OVARNET_OPEN_VOCAB_REGION_ATTRIBUTE_SOURCE_SCREEN_2026-09-27.md`.

Ticket 08 metadata audit confirms the frozen scene inputs include camera transforms and a known shared three-light rig; every fixture view uses the same rig. That supports the existing synthetic probe but does not establish varied-light behavior. Current real-image observations have no measured camera/light calibration, and the inspected teapot PNG carries no metadata. The audited spec requires missing capture values to remain unknown. Any real calibrated-light path needs per-view light direction/radiance, coordinate frame/units, and calibration provenance bound to the observation digest; known camera/exposure/white-balance data should also be retained. No fixture truth, scorer, model, or GPU was used. This is an acquisition/schema prerequisite only; Ticket 08 gates remain open.

### Active RX 7900 GRE Flamingo diagnostic run (2026-09-27)

Run `8bb46be6-1d5a-46e0-a023-7ddb8449257a` is still active in the project-owned `localhost/modly-amd-geosam2:ticket04` runtime, using the Flamingo geometry and sidecar documented above, seed 42, `MODLY_GEOSAM2_DIAGNOSTICS=1`, and the opt-in unselected prompt-seed-lift path. It is advancing through the 12 starting views and has written intermediate per-view postprocessed GLBs; there is not yet a final Structured Asset, complete run record, failure record, latency/peak measurement, or acceptance result. The run began before the diagnostic bounds patch and must be treated as exploratory. Do not score it as a pass or compare its face labels against a different mesh. Its running process is held by active shell session 57490 and must be polled before any rerun.

### Ticket 04 bounded diagnostic telemetry (2026-09-27)

The earlier opt-in diagnostic copied whole registration batches to CPU, retained uncapped rows, and rewrote the audit after every record. All three telemetry collectors and their persistence path are now bounded. Prompt/logit records, preflight objects/frames/output entries, mask-stage frames/masks, and generator view/candidate arrays have explicit caps plus omission counts. Propagated tensor metrics reduce on device and transfer scalar summaries only; audit writes are batched; generator wrappers forward close. Inference outputs and acceptance thresholds are unchanged. The project Python CPU suite independently passed 36 tests with 1 skip because PyTorch is absent, including all three exact policy-lock checks. Torch-side reductions still need an actual target runtime invocation. Module hashes: prompt `f494429b39ca38924882bd8c116feaa455910bc062db5848ac861873f1535b68`, mask `0646580b926b4172a6394acc45e4e25dc525b81b57198c885e3b650516ac9eee`, generator telemetry `642f13c8b0a45506b9908c360aeb0b408220894580ff7dd2e5fe1dd6fa4093a2`; locks: `f9c24d92237b1dac0531a4bbb68582ba44fe044e7144de1c8bbe8390c04394ef`, `5c359c8074d2efb4c311e37114eea59705e66f404dc47f3c09aee995980faed4`, `c369c9bcb0ba13f67306df1e3b6fee19418217405bc44403dfd6201fa34079f8`. No GPU work or fixture/truth access occurred for this telemetry change.

The active RX 7900 GRE Flamingo run `8bb46be6-1d5a-46e0-a023-7ddb8449257a` has recorded seven of twelve starting views and emitted intermediate postprocessed meshes through view 06. Its audit has grown to 16.6 MB under the old unbounded diagnostics, confirming that this run cannot qualify the revised logger. The process remains active in session 57490; at the latest check GPU use was 70% and current VRAM use 14,384,816,128 bytes (about 13.4 GiB of 16 GiB), which is not the final peak. There is no final Structured Asset/result yet. Poll this session and inspect its terminal state before any retry.

### Ticket 01 §15 source-capture metadata revalidation (2026-09-27)

A spec-to-implementation audit found that `StructuredAsset.source_observations` only carried artifact identity/path/digest/media type, while FINAL_AUDITED_SPEC §15 requires retaining camera pose/intrinsics, exposure, white balance, lighting, and capture order when known. Ticket 01 was reopened and then reaccepted after adding an optional digest-bound capture metadata schema, richer measured-light metadata, omission of absent values, and a real headless worker round-trip over a seeded legacy observation plus a measured observation. Validation passed: the Structured Asset/headless suites passed 38/38; the compatibility suite including Tickets 05–08 passed 89 tests with 3 environment-gated skips. Existing `ArtifactReference` callers remain accepted. This closes only Ticket 01; it does not claim that source metadata is automatically extracted from arbitrary files or images.

At the last pre-reboot poll, the earlier Flamingo diagnostic run `8bb46be6-1d5a-46e0-a023-7ddb8449257a` was reported active in shell session 57490. It later completed; the terminal outcome and limitations are recorded immediately below. This run predates bounded telemetry and cannot qualify the revised logger.

The run is now terminal. It emitted a Structured Asset with 16 valid, disjoint regions covering all 46,180 Flamingo faces; all 48 registered artifact hashes validated. Reported inference/load latency was 3,642,435.9 ms, peak allocated VRAM 12,869,627,904 bytes, and peak reserved VRAM 14,824,767,488 bytes on the RX 7900 GRE. It is not acceptance evidence: this old diagnostic path wrote a 545,638,320-byte sidecar by copying the 13.4 MB proposal audit into all 16 assertion provenance records, and the workflow command exited 1 after emitting `done`. No truth was scored and no quality claim is made. `api/runtime/adapters/parts/geosam2.py` now records only the audit digest and byte size in assertion provenance while retaining the complete audit once as a stage artifact. Regression tests verify this compact reference; a target rerun is needed to establish serialization/runtime exit behavior after the fix.

### Ticket 08 SuperMat base-model source gate (2026-09-27)

The pinned SuperMat code and publisher weight metadata identify MIT, but its required SD 2.1 base is an unaffiliated mirror pointing to CreativeML Open RAIL++-M terms whose authoritative linked endpoint returned HTTP 401. The mirror itself states research-purpose direct use; Modly commercial and redistribution scope is not established. The source-only review rejected acquisition/inference on current evidence and records the license/revision chain, known component hashes, and required clearance in `api/runtime/adapters/pbr/evidence/SUPERMAT_LICENSE_AND_RELEASE_SOURCE_GATE_2026-09-27.md`. No weights, code, fixtures, or truth were accessed. Ticket 08 remains open with all model-quality, novel-light, and target-runtime gates unchanged.


### Ticket 07 Material Palette source screen and Ticket 04 rerun state (2026-09-28)

Ticket 07's official-source screen found Material Palette is a PBR appearance-map generation/decomposition workflow, not a fixed material-identity classifier, and its 2D-mask input does not establish topology-bound regions. Immutable executable/weight identities, complete deployable rights, AMD support/resource evidence, and frozen fixture metrics are also absent. Rejected before acquisition/evaluation; no code, weights, datasets, fixtures, truth, or GPU were accessed. Report: `api/runtime/adapters/material-identity/evidence/TICKET07_MATERIAL_PALETTE_OFFICIAL_SOURCE_SCREEN_2026-09-27.md`. Ticket 07 gates remain unchanged.

The compact-provenance Flamingo rerun `a1c25070-3ee3-42a5-9bf8-c98d72fd9401` is no longer active. Its retained audit is still `inference_in_progress` with zero seed-view registrations/proposal counts; its output directory has render/input artifacts and an incomplete audit, but no final Structured Asset, failure record, runtime metrics, or terminal command evidence. The old shell session is no longer observable. Treat this rerun as interrupted/unqualified, not a model failure or acceptance result; do not infer success from the audit. The earlier completed run `8bb46be6-1d5a-46e0-a023-7ddb8449257a` and its non-acceptance status remain as recorded above.


Ticket 07's MatForge screen found a related dense PBR/material-group approach but no evidence for the frozen five-class, topology-bound identity and abstention requirements, deployable weight/terms pinning, target memory, or fixture quality. Rejected before acquisition; no model/data/fixture/truth/GPU accessed. Report: `api/runtime/adapters/material-identity/evidence/TICKET07_MATFORGE_OFFICIAL_SOURCE_SCREEN_2026-09-28.md`. Acceptance gates remain unchanged.

The earlier compact-provenance Flamingo rerun `a1c25070-3ee3-42a5-9bf8-c98d72fd9401` has only an incomplete `inference_in_progress` audit with zero proposals and no final asset. Podman now reports a machine reboot invalidated project run-state; that disposable run directory was cleared while retaining the project image/model storage. Running the project-owned status check requires a host user namespace because the restricted shell cannot grant `newuidmap` capabilities; the escalated project status command is pending. This interrupted attempt has no inference or acceptance result.


Ticket 08's ARM source screen reviewed a relightable 3D appearance-reconstruction research lead. Its maps are coupled to internally generated geometry/UVs; no official deployable model release/pins/terms, Modly existing-topology/material-region interface, or AMD runtime proof was found. Rejected before acquisition/evaluation; no code, weights, fixtures, truth, or GPU accessed. Report: `api/runtime/adapters/pbr/evidence/ticket08-arm-appearance-official-source-screen-2026-09-28.md`. Ticket 08 criteria remain unchanged.

After a reboot invalidated ephemeral project Podman state, only `.modly-amd-runtime/run` was reset; the stored images and model files were preserved. The project image inventory then passed. The interrupted compact-provenance Flamingo attempt had no output beyond an incomplete render/audit. A fresh same-input RX 7900 GRE diagnostic run `f2c25070-3ee3-42a5-9bf8-c98d72fd9401` has now been launched with bounded diagnostics; terminal output and artifact verification are pending.


Ticket 07's MatForge screen found a related dense PBR/material-group approach but no evidence for the frozen five-class, topology-bound identity and abstention requirements, deployable weight/terms pinning, target memory, or fixture quality. Rejected before acquisition; no model/data/fixture/truth/GPU accessed. Report: `api/runtime/adapters/material-identity/evidence/TICKET07_MATFORGE_OFFICIAL_SOURCE_SCREEN_2026-09-28.md`. Acceptance gates remain unchanged.

Ticket 04 compact-provenance and empty-proposal regression modules passed 18/18 in the project API test environment after the audit-pointer fix. Command: `TMPDIR="$PWD/.modly-amd-runtime/tmp" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=api .modly-amd-runtime/api-test-venv/bin/python -m unittest api.tests.test_ticket04_geosam2_diagnostic_audit_return api.tests.test_ticket04_geosam2_empty_proposal_policy -v`. This is CPU/support evidence only. The fresh Flamingo diagnostic `f2c25070-3ee3-42a5-9bf8-c98d72fd9401` advanced through validation into the RX 7900 GRE PyTorch (`Using device: cuda`); its GPU process remains live and its terminal result is pending.


Ticket 08's Redner/SVBRDF Estimation source screen found useful differentiable-rendering research, but no documented Modly topology/material-region and metallic estimator, AMD GPU path, target resource evidence, or frozen quality results. Rejected before acquisition/evaluation; no code, weights, fixtures, truth, or GPU accessed. Report: `api/runtime/adapters/pbr/evidence/ticket08-redner-official-inverse-render-screen-2026-09-28.md`. Gates unchanged.


Ticket 07's VLMAT screen reviewed camera-radar material identification. Its radar input and public evidence do not establish the fixed topology-bound image-region classes, calibrated abstention, or eligible AMD runtime. Rejected before acquisition/evaluation; no code, model, fixtures, truth, or GPU accessed. Report: `api/runtime/adapters/material-identity/evidence/TICKET07_VLMATERIAL_CAMERA_RADAR_SOURCE_SCREEN_2026-09-28.md`. Gates unchanged.

The active compact-audit run has now recorded four starting-view registration summaries. Its audit is 1,566,680 bytes with bounded telemetry; only one intermediate postprocessed mesh exists so far. The zero registration counts in views 1–3 are diagnostic observations, not a segmentation result. Continue polling run `f2c25070-3ee3-42a5-9bf8-c98d72fd9401` to terminal status before assessing sidecar size, artifact hashes, completion, or exit code.

Ticket 08 full test discovery passed 30/30 after installing the already-cached project SciPy wheel (`scipy-1.16.1`, SHA-256 `f965bbf3235b01c776115ab18f092a95aa74c271a52577bcb0563e85738fd618`) into `.modly-amd-runtime/api-test-venv`; no network access was needed. Command: `TMPDIR="$PWD/.modly-amd-runtime/tmp" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/.modly-amd-runtime/api-test-venv/lib/python3.12/site-packages:api" .modly-amd-runtime/api-test-venv/bin/python -m unittest discover -s api/tests -p 'test_ticket08_*.py' -v`. This confirms fixture/evaluator/projection contract behavior. The CPU fixed-geometry candidate evaluation reports base-color MAE 0.1236 vs <=0.08, SSIM 0.6408 vs >=0.85, roughness MAE 0.4181 vs <=0.10, metallic MAE 0.3175 vs <=0.10; it fails the frozen quality gates and is not selected. Heldout novel-light MAE 0.0545 is within threshold but cannot compensate for failed channels. PBR acceptance remains open.

The RX 7900 GRE Flamingo diagnostic `f2c25070-3ee3-42a5-9bf8-c98d72fd9401` is terminal with exit 1: opt-in, unselected prompt-seed lift failed closed as `GEOSAM2_UNASSIGNED_COMPLETION_FAILED`, leaving all 46,180 faces at sentinel 999. The audit shows 34 accepted starting-view proposals, but every registered predictor output returned all `-1024` logits; 192 propagated summaries covering 427,819,008 values contain zero positives. Views 1–11 registered none. The audit is 2,190,083 bytes, confirming bounded audit telemetry, but no final Structured Asset/sidecar was emitted, so the compact provenance reference and successful workflow exit remain unverified. This diagnoses the opt-in candidate only and does not attribute the failure to the selected default path. No truth was accessed or scored. Ticket 04 acceptance remains open and unchanged.

A new current-default-path Flamingo diagnostic `c3d25070-3ee3-42a5-9bf8-c98d72fd9401` is running on the exact same mesh and source sidecar without the opt-in prompt-seed-lift variable. It uses diagnostics only. The goal is to check successful output/launcher behavior and compact audit provenance on the selected path; this Flamingo asset has no truth labels, so the run cannot qualify face quality. Poll this existing process to terminal status and do not duplicate it.


Ticket 07's MaterialSeg3D/MIO++ screen found a relevant dense multi-view 3D material lead but no published evidence for Modly's full label taxonomy/abstention, topology-bound provenance, pinned usable weights and terms, RX 7900 GRE resource compliance, or frozen fixture quality. Rejected before acquisition/evaluation; no assets, fixture, truth, inference, or GPU accessed. Report: `api/runtime/adapters/material-identity/evidence/TICKET07_MATERIALSEG3D_FIXED_DENSE_3D_SOURCE_SCREEN_2026-09-28.md`. Gates unchanged.


Ticket 08's fixed-geometry inverse-render estimator now bilinearly expands fitted UV-cell-center values. The CPU metrics improve base-color MAE/SSIM to 0.1031/0.7203 but still fail unchanged limits (0.08/0.85); roughness and metallic errors remain 0.4152 and 0.3176 against <=0.10 and metallic bias is 0.2471 against <=0.08. Novel-light MAE is 0.0416 against <=0.08. The candidate is still rejected because fixed mesh normals cannot represent the fixture's height-derived shading normals; no thresholds, fixture, or scorer changed and no AMD inference was run. Report: `api/runtime/adapters/pbr/evidence/fixed-geometry-bilinear-expansion-cpu-development-2026-09-28.md`. After this change, full Ticket 08 CPU discovery passed 30/30; test success does not pass model quality.

An additional Ticket 08 primary-source review of Sony NDJIR's official Dockerfile strengthens the existing 2026-09-25 source screen: the image is CUDA 11.0, installs NNabla's CUDA extension and builds native components; the official custom-data instructions estimate about 3.5 hours to train on 100 images using an A100. It does not qualify AMD execution. Existing topology/material-region and metallic-channel blockers remain. No code, weights, fixture inputs, held-out truth, or GPU accessed; no gates changed. Updated evidence: `api/runtime/adapters/pbr/evidence/ticket08-ndjir-diffreg-pbir-primary-source-screen-2026-09-25.md`.

The current-code selected default-path Flamingo diagnostic `c3d25070-3ee3-42a5-9bf8-c98d72fd9401` is terminal with exit 1 and `GEOSAM2_NO_FACE_LABELS`. It used the same mesh SHA-256 `e099909f…33aa99` and exact render-manifest digest `b2845065…1e7ab` as prior Flamingo runs, seed 42, and no prompt-seed-lift option. It reports 28 accepted starting-view proposals and none in views 1–11, but outputs no face labels, final asset, or sidecar. Because this version includes newer bounded diagnostic/provenance code, it is not an exact same-code repeat of earlier successful runs. It confirms the current path remains unreliable on this unchanged geometry; no truth was accessed or quality scored. Ticket 04 remains open.

Ticket 07 full project-venv discovery is not green: 71 tests ran, 6 optional-Pillow skips, one failure, and one import error. The `test_owner_approved_fixed_gate_source_digest_is_pinned` failure reports current `SELECTION_AND_GATES.md` SHA-256 `bbdb9e384fcc96f7abbbb33d92d09f37f092f18b4fe32f8a5744aa1eb85c5ca0`, while the runner and DMS46 preregistration pin `6998d86185bf1d91d3af0fe2e3643f232cb0d6765c554109a962f07b8e14caea`. I made no change to that source or digest; reconcile the discrepancy against the frozen preregistration before any dependent scoring. `test_ticket07_qwen3_scoring` also fails to import because Torch is absent from the CPU venv and needs a project Torch environment. These test issues do not change Ticket 07 gates.

The isolated Torch-only Ticket07 Qwen3 scoring contract passed 9/9 inside the project GeoSAM2 image with no GPU devices attached and the image's installed `typing_extensions` preloaded. The CPU venv suite's Qwen import error is an environment limitation rather than a scorer-test failure. Its separate frozen gate-source digest mismatch remains unreconciled; no pin or criteria were modified.

Read-only Ticket04 audit localized the first confirmed fault to the initial prompted SAM2 frame for both the opt-in and default paths. Temporary `object_score_logits` are all non-finite BF16; `pred_masks` contain only the `-1024` no-object sentinel; propagation and mask filtering therefore receive no positive pixels. This is before face lifting and disproves prompt-seed lift as the cause. The upstream path enables BF16 autocast when the CUDA PyTorch API is available (also used by ROCm), but this remains a hypothesis, not a verified root cause. No precision policy has changed. A bounded, numeric-only per-layer trace is being implemented before any precision experiment; acceptance gates and inference output are unchanged.

Ticket 04 added a bounded, opt-in prompted-inference trace at image-feature output, raw mask-decoder output, and post-object-score-gating output. It keeps numeric shape/dtype/finite/range/sentinel summaries only, finds the first nonfinite stage, caps records, suppresses trace failures, and restores wrapped functions. No inference outputs or thresholds changed. The v4 diagnostic lock and `geosam2.py` verifier seam are pinned; focused tests pass 26 (1 optional skip), including lock tampering. The new instrumented default-path RX 7900 GRE run `d4e25070-3ee3-42a5-9bf8-c98d72fd9401` has started on the same truth-free Flamingo input; first-nonfinite-stage evidence is pending.

An independent read-only Ticket07 gate-pin audit confirmed current selection source SHA-256 `bbdb9e384fcc96f7abbbb33d92d09f37f092f18b4fe32f8a5744aa1eb85c5ca0` differs from frozen DMS46 preregistration/runner SHA-256 `6998d86185bf1d91d3af0fe2e3643f232cb0d6765c554109a962f07b8e14caea`. Current held-out thresholds, cohorts and runtime limits, plus the owner-approved development 0.87 supported-coverage floor, align semantically with the preregistration. The likely differences are added candidate evidence and corrected DMS46 resize wording, but there is no original byte copy or usable Git history to prove the full diff. The strict pin test remains a real blocker; no source or hash was changed. Recover the exact frozen source or prepare a reviewed versioned preregistration with unchanged thresholds before more candidate scoring. DMS46 itself has already failed development and is not eligible for heldout access.

Initial instrumented RX 7900 GRE trace from `d4e25070-3ee3-42a5-9bf8-c98d72fd9401` shows the first observed non-finite stage is `image_features`, before the raw mask decoder: in all first 16 bounded prompt traces, FPN level 1 (`output[1].backbone_fpn[1]`) has 655,872/1,048,576 finite BF16 values, while FPN level 2 (`output[1].backbone_fpn[2]`) has 0/1,048,576 finite FP32 values. Flattened feature forms match. This rules out mask decoding and face lifting as the first *observed* fault, but does not yet distinguish an earlier image-encoder op, model weights, input preprocessing, or ROCm precision. Prompt payloads remain absent; run remains active until terminal output.

The `d4e...` coarse-trace Flamingo run is terminal. It produced a segmentation manifest but failed raw full-face coverage: 44,661/46,180 faces assigned and 1,519 sentinel `999`; completion produced a derived 15-label artifact. This Flamingo has no truth labels, so no quality claim is available. The audit is 11,649,051 bytes with no prompt/image/tensor payload, and all 5 manifest artifact references validated for path, size, and SHA-256. RX 7900 GRE recorded 13,141,639,680 bytes peak allocated and 16,145,973,248 bytes peak reserved out of 17,163,091,968 total; load plus inference was 787,313.21 ms. This run imported the earlier coarse v4 instrumentation before the expanded per-branch diagnostics were written, so it does not qualify that trace. Workflow exit was 1; no final sidecar was emitted, leaving compact assertion provenance and successful workflow completion unverified. Ticket 04 remains open with unchanged acceptance criteria. The expanded trace was independently revalidated (28 focused tests passed, 1 Torch-dependent test skipped in the CPU venv; modified Python files compile) and is running on the same Flamingo input as `e5e25070-3ee3-42a5-9bf8-c98d72fd9401`; await that process before any retry.

Independent code review and the live `e5e...` audit exposed a timing gap in the expanded instrumentation: pinned `init_state()` prewarms/caches frame-0 features after wrappers are installed, but outside the active prompt trace. The completed weight/buffer scan shows finite loaded state values, but the per-layer encoder/position/fusion activation hooks are not active during that prewarm in the old trace. Therefore `e5e...` can confirm only that the cached image-feature bundle is already non-finite; it cannot identify which internal layer first produced it. A corrected diagnostic must observe the actual prewarm forward without changing inference outputs. No precision cause or fix is established.

The timing gap is corrected in the pinned v4 collector: it now brackets the real `_get_image_feature` cache-miss call, which is where `init_state()` performs its frame-0 prewarm, and the existing module/forward wrappers summarize that same activation flow. The first later prompt remains a cache hit; a CPU stand-in verifies no duplicate forward, ordered branch summaries, and restoration. Focused tests pass 29 with one PyTorch-only skip, modified Python files compile, and the exact module/lock hashes pass the adapter verifier. The schema shape remains v4 because no fields or acceptance semantics changed. This patch has not run on the RX 7900 GRE. It does not prove which layer is responsible; a fresh target trace after `e5` terminates is still required, with no inference precision or gate changes.

Ticket 07's frozen-source recovery recheck found no exact baseline in project or `/tmp` backup/name/digest searches. `git rev-parse` and `git cat-file` cannot recover it because this checkout has no Git metadata; the old SHA appears only in preregistration/audit/runner references, not recoverable source bytes. Current selection source remains SHA-256 `bbdb9e384fcc96f7abbbb33d92d09f37f092f18b4fe32f8a5744aa1eb85c5ca0`; the frozen pin remains `6998d86185bf1d91d3af0fe2e3643f232cb0d6765c554109a962f07b8e14caea`. No file, threshold, or pin was altered and no fixture/truth/inference was touched. Do not resume DMS46 preregistered scoring until the exact authorized baseline is located or an unchanged-threshold versioned preregistration receives review. Full search disposition is in the Ticket 07 issue's “Frozen-source recovery recheck” entry.

Target `e5e25070-3ee3-42a5-9bf8-c98d72fd9401` is terminal: exit 1, `GEOSAM2_NO_FACE_LABELS`; 34 registrations; 192 propagation records; all 427,819,008 propagated values finite `-1024`, zero positive pixels. Temporary object-score logits are non-finite BF16 and prompt masks are `-1024`. The loaded-state scan checked 810 encoder/position/fusion tensors and found all finite. However the trace only observed the cached image-feature bundle plus decoder and gating because the frame-0 warmup happened before activation tracing; the run cannot identify the first bad layer. Only the 2,193,264-byte payload-free audit was emitted; no labels, segmentation manifest, sidecar, truth access, or quality scoring. The corrected prewarm hook passed 29 focused tests with one Torch-only skip, compile, and lock verification. Corrected target run `f6e25070-3ee3-42a5-9bf8-c98d72fd9401` is active on the same Flamingo input. This supersedes earlier pending status for `e5`; await the `f6` process and do not infer layer causality from `e5`.

The corrected `f6e...` prewarm trace has now localized the earliest non-finite module output: RGB/position/normal inputs are finite; image encoder output is finite; `pos_map_encoder_output` contains no finite values in `vision_features` or FPN levels, with one positional encoding only partly finite. All 810 inspected encoder/position/fusion state tensors are finite. This establishes the position-map encoder as the first observed bad module boundary; it does not identify a specific child layer, primitive operation, or precision cause. Feature fusion, decoder, and gating receive corrupted values downstream. A separate bounded named-child trace is now the next source-only diagnostic patch; do not launch it on GPU until the current `f6` process is terminal. No inference behavior, thresholds, or ticket gates changed.

`f6e25070-3ee3-42a5-9bf8-c98d72fd9401` is terminal. It emitted `done`, but shell exit was 1 because the event filter mistook known pinned upstream stdout messages for failure. A narrowly allowlisted filter now accepts only exact source-identified device/object/progress messages and a strict generated-mesh path; unrecognized output, error events, and absent `done` still fail. Its synthetic parser suite passes 4/4 and shell syntax passes. The run emitted an 870,373-byte sidecar with 14 assertions that all point to the single 11,662,546-byte audit by digest/size; the five manifest artifacts all match recorded bytes and SHA-256. Raw labels assigned 44,662 of 46,180 faces; 1,518 sentinel faces were filled only by the locked derived completion routine, confidence remains unknown, and this truth-free mesh supplies no quality score. RX 7900 GRE peak allocated 13,140,590,592 bytes, reserved 15,118,368,768/17,163,091,968 total (above the 14 GiB gate); load/inference latency was 795,084.78 ms. It identifies the position-map encoder output as the earliest bad module boundary but predates child-level tracing. The new child-module trace passed 31 focused tests with one Torch-only skip, plus compile and lock checks. Launch a fresh GPU trace with the current code; no acceptance gate has passed from this run.

`f7e25070-3ee3-42a5-9bf8-c98d72fd9401` is terminal with explicit `GEOSAM2_NO_FACE_LABELS`, exit 1, and no manifest/sidecar; its 1,624,423-byte audit records 32 registrations and 192 propagated summaries. Its coarse trace found partial non-finite values already in `image_encoder_output` and then partial non-finites in `pos_map_encoder_output`, unlike f6 where the image encoder outputs were finite. All 810 model state tensors are finite. The child trace covered 429 position-encoder child modules with no hook truncation, but some child counts were mathematically impossible for the tensor shapes, so its claimed first bad child `linear_a_q` is not trustworthy. Add hard counter validation/fail-closed behavior and trace children under both encoders before another target run. This is evidence of run-to-run variance and instrumentation limits, not a root-cause or quality result; Ticket 04 gates remain unchanged.

The diagnostics are now schema v5. Torch finite/positive/sentinel counts use float64 accumulation and must be exact integers in `[0, numel]`; impossible values yield type-only `diagnostic_error` and do not mark a tensor or stage non-finite. Bounded named-child summaries now cover both image and position-map encoders in their true forward call order, retaining deterministic module paths, per-encoder hook coverage, explicit omissions and summary failures. Limits are 1,024 hooked child modules per encoder, 2,048 child-output records per run, and 8 summarized tensor leaves per output. Temporary forwards restore after the trace. CPU tests cover count bounds/error semantics, both-encoder order and localization, caps, and restoration. Focused suite: 33 tests passed, one skipped because Torch is unavailable; Python compilation and v5 module/lock verification passed. No GPU run used v5, and f7's `linear_a_q` result remains unqualified; first internal cause is unknown. The coarse target evidence differs between f6 and f7, so a fresh RX 7900 GRE run after f7 terminates must collect both branches before any root-cause claim. No precision, inference, output, or acceptance criteria changed.

The latest whole-module target trace counted 402 loaded state tensors each for image and position-map encoders, and 6 for feature fusion; all are finite. Inputs and image-encoder outputs are finite, while `pos_map_encoder_output` has non-finite `vision_features` and all three FPN values, with positional encoding partly finite. This is the first observed bad module boundary only. The opt-in v4 source now adds temporary named-child forward wrappers under `pos_map_encoder` during the actual frame-0 cache-miss prewarm. It stores deterministic module paths plus numeric summaries only, with caps of 512 child hooks, 512 output records per run, and 8 numeric leaves per output; partial coverage, omitted records, and summary errors are explicit and fail closed. Existing whole-module traces remain intact and hook restoration preserves original forwards. CPU diagnostic/audit-return tests pass 31 with one PyTorch-only skip; Python compilation and exact lock verification pass. The `f6` process imported its source before this child-hook patch and cannot qualify it. Wait for `f6` to terminate, then use a fresh same-input RX 7900 GRE run to locate the first non-finite child; no child layer, kernel, or precision cause has been established, and no inference/output/acceptance criteria changed.

Ticket 07 official-source screen added for the 3DCoMPaT++ point-cloud material-segmentation winner. It is a relevant RGB point-cloud/per-point-label route, but the documented coarse taxonomy lacks the frozen clear-plastic and paint/plaster/enamel classes; the documented output has no unknown/ambiguous or bare/painted-metal abstention; and the winner checkpoint is still marked “will be uploaded soon” / `TODO`. Modly topology-bound evidence and RX 7900 GRE <=14 GiB execution are not established. Rejected before acquisition/evaluation. No assets, fixtures, truth, inference, or GPU accessed and no gate/pin changed. Full source-backed report: `api/runtime/adapters/material-identity/evidence/TICKET07_3DCOMPATPP_POINTCLOUD_SOURCE_SCREEN_2026-09-28.md`.

Ticket 07's API contract module rerun found the documented `PYTHONPATH=api` invocation is broken in this environment: `api/typing_extensions.py` shadows the installed package before Pydantic imports. Preloading installed `typing_extensions` before adding `api` to `sys.path` resolves that import-order issue. Exact rerun completed 15 tests: 14 passed, one model-root integration test skipped because the CPU environment has no configured SigLIP2 model root. This is API contract evidence only; candidate quality and RX 7900 GRE acceptance remain open, and the DMS46 frozen gate-source pin mismatch is unchanged. No fixture truth or model inference was accessed. See the Ticket 07 issue for the exact command and result.
# Ticket 08 Spec-Gloss Surfels source screen (2026-09-28)

An official-source-only screen found Spec-Gloss Surfels relevant to multi-view glossy-object reconstruction and novel-light relighting, but not eligible for Ticket 08 evaluation. Its 2D Gaussian-surfel reconstruction does not preserve an existing Modly topology/material-region map; its published G-buffer includes albedo, roughness, and F0 but not an explicit metallic map. The official environment includes NVIDIA packaging, `nvdiffrast`, and custom rendering submodules; no AMD target path or RX 7900 GRE resource evidence is documented. MIT applies to the code, while linked prior-model identities/terms and a complete immutable component lock are unresolved. No artifacts, weights, data, fixture/truth, inference, or GPU were accessed. No quality claims, thresholds, or acceptance criteria changed. Report: `api/runtime/adapters/pbr/evidence/ticket08-specgloss-surfels-official-source-screen-2026-09-28.md`. Ticket 08 remains open.

### Ticket 04 v5 target trace (2026-09-28)

Fresh truth-free RX 7900 GRE run `f8e25071-3ee3-42a5-9bf8-c98d72fd9401` used the current schema-v5 both-encoder child trace. It terminated with `GEOSAM2_NO_FACE_LABELS`, exit 1, and no final labels, manifest, or sidecar. The 429 child modules under each encoder were fully inventoried; 866 output summaries were recorded without omission. The model-state scan found all 810 tensors finite, and all recorded whole-module boundaries (both encoders, fusion, cached features, and inputs) were finite. One position-map block-7 `attn.qkv.linear_b_v` child summary showed 2,126,638/2,195,200 finite values; the immediately preceding query projection summary failed closed with a type-only error. The parent `attn.qkv` output and full encoder output report all values finite, although pinned source adds the V child result into that parent tensor. This contradiction means the child count is unqualified and no NaN root cause is established. The 32 first-view prompt registrations, stored masks, and object scores were finite, and six of 192 propagated summaries had no positive values. Mask snapshots show 384 propagated objects / 4,196,618 positive pixels, reduced to 36 objects / 214,279 pixels after area filtering, and then zero objects at the final IoU accumulator. Only the starting view retained accepted proposals (32); all other 11 views had zero. These observations point to proposal/filter survival as the immediate area to investigate, not an established encoder NaN. No truth or quality score was used. GPU use returned from a sampled 10.25 GB while active to 772,972,544 bytes after exit versus a 783,028,224-byte pre-run sample; peak allocated/reserved and latency were not captured. Full evidence: `api/runtime/adapters/parts/evidence/TICKET04_GEOSAM2_V5_DIAGNOSTIC_F8E25071_2026-09-28.md`. Ticket 04 acceptance remains open.

Ticket 04 memory wording reconciliation: `api/runtime/adapters/parts/SELECTION_GATES.md` fixes the peak VRAM ceiling at 16 GiB. The f6 value 15,118,368,768 bytes reserved is about 14.08 GiB and is below that ceiling; the older phrase “above the 14 GiB acceptance limit” in the f6 note was incorrect and is superseded in the ticket issue. Memory recovery remains a separate acceptance requirement.

Ticket 07 additional official-source screens found no new frozen-gate candidate. Material Magic Wand provides exemplar-based part/material similarity rather than named, calibrated material identity; MaterialSeg3D++ produces/fuses material maps and has no demonstrated frozen ontology/abstention or AMD target route; OpenScene and CUA-O3D are scene-level open-vocabulary systems without the required topology-bound material assertions or target evidence. No weights, code, fixtures, or truth were accessed. Ticket 07 remains open; no candidate gate or threshold changed.

### Ticket 08 PyTorch3D synthetic GPU operation preflight (2026-09-28)

After Ticket 04 run `f8e25071-3ee3-42a5-9bf8-c98d72fd9401` became terminal, one bounded generated-input PyTorch3D rasterization/interpolation operation ran on the RX 7900 GRE. The 16×16 face mask matched an independent CPU reference exactly; maximum barycentric and interpolated-attribute errors were 1.0523e-7 and 1.0448e-7. Peak allocated/reserved memory was 15,872 B / 2 MiB and returned to zero after cleanup. PyTorch 2.11.0+rocm7.14.0 / HIP 7.14.60850, Python 3.12.3, PyTorch3D 0.7.9, pinned source `b73d735ecf194c31de812feffef3a55cc3726128`; source checkout is clean. This qualifies only the isolated PyTorch3D primitive: Material Anything's UV-atlas/Kaolin route, model identity/terms, quality, and Ticket 08 acceptance remain unresolved. No truth, fixture, source observation, or model weights were accessed. Evidence: `api/runtime/adapters/pbr/evidence/material-anything-pytorch3d-rocm-synthetic-op-preflight-2026-09-28.md`; JSON and log hashes are recorded there.

### Ticket 08 Material Anything UV/Kaolin parity source review (2026-09-28)

Read-only review of pinned Material Anything commit `be3d6b32a195f968540abc2ee106dc02d4b07479` maps `uv_to_3d`: Kaolin gathers per-face 3D vertex features, rasterizes separately indexed UV triangles at zero depth, and supplies an interpolated XYZ atlas plus face IDs; later code clamps/rescales the map and writes `(1,1,1)` at `face_idx == -1`. PyTorch3D face IDs, barycentrics, and face-attribute interpolation are a plausible mapping while preserving original face rows and UV seams, but no evidence yet establishes Kaolin pixel/edge rules, atlas orientation, out-of-range behavior, equal-depth overlap winner, or converted/dilated sentinel parity. The active helper also uses PyTorch3D camera rendering and UV projection paths beyond the one synthetic operation already qualified. Material Anything declares no Kaolin dependency/version and does not pin PyTorch3D; its estimator/refiner files remain unhashed/unfetched and model/data rights are not fully qualified. Recommended next step is a pinned Kaolin/reference decision followed by the synthetic seam/edge/out-of-range/uncovered/equal-depth-overlap parity matrix described in `api/runtime/adapters/pbr/evidence/material-anything-uv-raster-parity-source-review-2026-09-28.md`. No GPU, weights, fixture, observation, or truth was accessed; candidate selection, thresholds, and acceptance remain unchanged.

Ticket 08's follow-up source/runtime screen identified immutable Kaolin `v0.18.0` at `06ffb7d955ca26b608c60a9e862327c56b226921` (Apache-2.0 library), but its default rasterizer calls Kaolin's custom CUDA extension and the project PyTorch 2.11 ROCm image has no Kaolin package. The optional nvdiffrast route is a different backend, unavailable in the current image, and would need parity against the default before serving as a reference. Kaolin's official docs also list tested PyTorch versions ending at 2.8.0 / requirements through 2.5.1, excluding the project runtime. Therefore exact Material Anything default raster parity cannot be run in the current AMD image. No install/build/download, GPU, weights, fixtures, observations, or truth were used; Ticket 08 gates remain unchanged. Evidence: `api/runtime/adapters/pbr/evidence/material-anything-kaolin-reference-runtime-screen-2026-09-28.md`.

### Ticket 04 seed-to-lift control-flow review (2026-09-28)

Read-only interpretation of truth-free run `f8e25071-3ee3-42a5-9bf8-c98d72fd9401` and the exact pinned GeoSAM2 inference source identifies a concrete current failure path: view 0 had 32 accepted proposals and positive propagated masks; after filtering, 24 candidate masks remained. Pinned `inference()` adds these to accumulated masks but deliberately skips 3D lifting on the automatic prompt-seed view when opposite-view segmentation is enabled. Views 1–11 had no accepted proposals, so no later view produced a lift and the adapter failed with no face labels. This narrows the evidence for this run to seed-to-lift handoff, not encoder corruption; it is not proven as the cause of all failures. The separate prompt-seed-lift candidate has already failed other target runs with all-sentinel labels. A new opt-in policy candidate, if pursued, must separately prove seed masks reach lifting on target, retain strict coverage/quality/topology/resource gates, and pass the frozen known-truth fixture before selection. No code, thresholds, truth, or acceptance status changed. Details: `api/runtime/adapters/parts/evidence/TICKET04_F8_SEED_TO_LIFT_CONTROL_FLOW_REVIEW_2026-09-28.md`.

### Ticket 08 Material Anything source-reference contract (2026-09-28)

Added eight generated CPU-only tests for source-visible Material Anything `uv_to_3d` semantics: independent UV-vs-geometry face-index gathering with a seam, face-row alignment, literal clamp/scale/uncovered-sentinel postprocessing, uint8 conversion, and a bounded binary dilation footprint. Focused suite passed 8/8. Broader Ticket 08 regression discovery passed 38/38 and includes the separate existing synthetic fixture/known-map tests. This is not a raster parity test: no Kaolin revision or executable is pinned or staged, so atlas orientation, pixel centers, exact/near edge inclusion, partly out-of-range coverage, overlap/tie order, Kaolin barycentrics, and full color/mask dilation output remain unknown. The exact source transform plus uint8 cast also wraps endpoint values outside [0,1]; this is documented but not modified pending source-parity evidence. Full report: `api/runtime/adapters/pbr/evidence/material-anything-uv-reference-contract-2026-09-28.md`. The new UV probe uses no model weights, fixture, source observations, or truth; no GPU, candidate selection, or gate/threshold changes.

### Ticket 04 block-7 LoRA Q/V target cross-check (2026-09-28)

A bounded one-frame cache-prewarm diagnostic now independently confirms actual non-finite BF16 outputs on RX 7900 GRE in position-map encoder block-7 LoRA `linear_b_q` (1,291,520/2,195,200 finite) and `linear_b_v` (681,856/2,195,200 finite); parent Q/V slices after LoRA addition are also partially non-finite. The prior malformed Q-child counter and finite parent summary are therefore resolved as an instrumentation contradiction for this execution. The probe copied only the four tensor outputs temporarily to CPU to recompute exact counts; it saved scalar metadata only, no labels, and performed no segmentation. Source shows each B projection consumes its corresponding A projection applied to the shared block input, but this probe did not inspect the block input, base QKV, or A outputs, so first-bad operation and precision cause remain unknown. It completed in 17.55 s through predictor construction/cache warmup. A wrong Podman ID-format check was corrected before the successful invocation; immutable image ID is recorded. Ticket 04 remains open; the result cannot pass segmentation quality, repeatability, complete coverage, or full-workflow gates. Full details: `api/runtime/adapters/parts/evidence/TICKET04_GEOSAM2_BLOCK7_QV_TARGET_CROSSCHECK_2026-09-28.md`.

The reviewed eight-boundary follow-up then captured block-7 QKV input, base projection, LoRA A/B Q/V branches, and final parent slices. The QKV input was already non-finite: 1,013,376/2,195,200 FP32 values finite. The same count remained in each BF16 B output and parent Q/V slice; A outputs had 9,048/19,600 finite values, and the base QKV output had exactly 3× the input finite count. This locates the first bad boundary *observed by that probe* before block-7 QKV, without attributing the earlier source. The command completed in 9.68 s with payload-free scalar output. No labels, segmentation, truth or quality scoring occurred; Ticket 04 remains open. Full report: `api/runtime/adapters/parts/evidence/TICKET04_GEOSAM2_BLOCK7_QV_BOUNDARY_CROSSCHECK_2026-09-28.md`.

### Ticket 04 Hiera boundary diagnostic correction (2026-09-28)

Independent review caught an invalid v3 assumption that block-7's post-window QKV input would have the same shape as its unpartitioned norm output. The corrected schema v2 records the pinned block-7 window size and validates the real partition contract: block6/norm1/block7 outputs retain the same feature-map shape, while QKV input has `[B*ceil(H/w)*ceil(W/w), w, w, C]`. The synthetic report fixture exercises a non-divisible 64x64 map with 14-pixel windows and the resulting padded 50x14x14x448 shape for batch 2; rejection checks cover invalid window count, channel width, and window size. Independent review cleared the correction; the focused validator suite passes 18/18, Python compilation and shell syntax checks pass. Fifteen fresh target prewarms then used the same Flamingo input, pinned source/weights/image, and seed. All fifteen host-validated schema-v2 reports found every one of the eleven measured boundaries finite, including block-7 QKV input (2,195,200/2,195,200) and block7 output (1,835,008/1,835,008). The earlier v2 QKV report on the same locked input found 1,013,376/2,195,200 finite there; its diagnostic source differed and it did not capture intervening Hiera boundaries. The 15-run screen does not reproduce the earlier signal and cannot establish its cause or variation rate under the earlier harness. Per-run build/prewarm time ranged from 8,900.347 to 11,538.856 ms (mean 9,325.712 ms); no segmentation or labels ran. Full evidence, identities, and all report hashes: `api/runtime/adapters/parts/evidence/TICKET04_GEOSAM2_HIERA_15_RUN_PREFLIGHT_VARIATION_SCREEN_2026-09-28.md`. `rocm-smi` found no KFD PIDs after the batch, but sampled VRAM rose from 1,619,185,664 to 1,734,553,600 bytes; these are not peak readings or proof of allocator recovery. Ticket 04 acceptance remains open and unchanged.

The independently reviewed schema-v3 follow-up added the patch-embedding and positional-embedding outputs plus read-only precision/determinism/runtime flags, with exact f8 stage-shape enforcement. One fresh RX 7900 GRE prewarm completed with all 13 captured events finite; autocast was enabled with BF16, deterministic algorithms were disabled, both TF32 flags were enabled, and the trunk was in evaluation mode. These are measurements, not an identified cause. The earlier v2 non-finite observation remains unexplained; this run did not segment or produce labels. The scalar report SHA-256 is `0ab51020b2405030581dff311cc790f07b9aeaebcc21080e73b543149d9ee675`; full execution identity and limitations: `api/runtime/adapters/parts/evidence/TICKET04_GEOSAM2_HIERA_PREFIX_RUNTIME_STATE_CROSSCHECK_2026-09-28.md`. Ticket 04 remains unaccepted with all criteria unchanged.

Ticket 07's bounded official-source screen added SSUF. Its four outdoor terrain classes, single-image/single-class output, absent checkpoint/source identity, missing topology/material-region interface, and lack of required abstention behavior do not satisfy frozen Ticket 07 entry gates. No assets, fixtures, truth, or GPU were accessed. Report: `api/runtime/adapters/material-identity/evidence/TICKET07_SSUF_OFFICIAL_SOURCE_SCREEN_2026-09-28.md`.

Ticket 08's bounded official-source screen added MatPredict. Its README describes image-space albedo, roughness, and metallic outputs, but the dataset remains forthcoming, no pretrained checkpoint identity is documented, and no supplied-mesh/topology/material-region binding is described. Rights, AMD resource, provenance/confidence, frozen map metrics, and held-out novel-light gates remain unresolved; no assets, fixtures, or truth were accessed. Report: `api/runtime/adapters/pbr/evidence/ticket08-matpredict-official-source-screen-2026-09-28.md`.

Ticket 07's additional primary-source discovery refresh found no new eligible classifier. FMMC, Apple-DMS, and MaterialSeg3D++ leads were already screened; DenseVLM does not establish the required material-specific labels and uncertainty behavior. This bounded screen accessed no model/data/fixture/truth and changed no gates. Report: `api/runtime/adapters/material-identity/evidence/TICKET07_NEW_SOURCE_DISCOVERY_REFRESH_2026-09-28.md`.

Ticket 08 screened VideoNeuMat from its official project source. Code is marked
forthcoming; the described Wan 2.1 14B / Wan 1.3B pipeline does not establish
separate topology- and region-bound PBR maps, immutable checkpoint identity and
rights, AMD qualification, or frozen-fixture results. Rejected before any
acquisition/evaluation; no gate changed. Report:
`api/runtime/adapters/pbr/evidence/ticket08-videoneumat-official-source-screen-2026-09-28.md`.

Another Ticket 08 refresh screened PM-PMVS. It estimates a uniform material in
a CUDA/NVIDIA reconstruction flow producing a new point cloud, not separate
PBR maps attached to Modly's supplied regions and topology. Its terms also
require permission for commercial use. Rejected before acquisition/evaluation;
no gates changed. Report:
`api/runtime/adapters/pbr/evidence/ticket08-pmpmvs-official-source-screen-2026-09-28.md`.

Ticket 07's independent source refresh screened MatSeg. Its class-agnostic
query-point method does not provide the frozen material identity labels,
unknown/ambiguous behavior, or topology-bound region contract. Checkpoint
identity/terms are incomplete and the inspected inference path is CUDA-only.
Rejected before acquisition/evaluation; no gates changed. Report:
`api/runtime/adapters/material-identity/evidence/TICKET07_MATSEG_ZERO_SHOT_SOURCE_SCREEN_2026-09-28.md`.


### Ticket 08 fresh primary-source discovery (2026-09-28)

A bounded search found no new estimator to advance. The official MaterialSeg3D++
article describes 30 material labels with a separately assigned roughness and
metallic pair per class, confirming the already recorded class-to-preset
limitation. M-XR I2M and LumiTex were also surfaced but already have source
screens, so no duplicate reports were added. No code, weights, packages,
fixtures, source observations, or truth were accessed; no gate changed. Details
and direct official sources: `api/runtime/adapters/pbr/evidence/ticket08-fresh-primary-source-discovery-2026-09-28.md`.

### Ticket 08 MatLat official-source screen (2026-09-28)

MatLat is a mesh-conditioned PBR map generator with base-color, roughness,
and metallic outputs, but its official checkpoint terms are CC BY-NC 4.0, its
documented inference path uses nvdiffrast/CV-CUDA without ROCm/RX 7900 GRE or
<=14 GiB evidence, and the source does not establish preservation of caller
topology, UVs, or material-region IDs. Code and complete consumed asset
identities also remain unpinned. Rejected before acquisition/evaluation; no
weights, packages, fixtures, observations, or truth were accessed and no gate
changed. Evidence: `api/runtime/adapters/pbr/evidence/ticket08-matlat-official-source-screen-2026-09-28.md`.

### Ticket 04 same-mesh proposal variation follow-up (2026-09-28)

The earlier exact-input search omitted
`.modly-amd-runtime/results/ticket04-variation-panel`. The exact Flamingo
bundle is now recovered and identity-verified at
`.modly-amd-runtime/results/ticket04-variation-panel/flamingo/StructuredAssets/runs/f8c6a77e-7090-4713-bb7d-3ddcedadeb1b/`; its render-manifest,
face-map, and mesh hashes match the schema-v3 report. Retained repeat-render
fixture copies are different assets. Ticket 04 remains open and acceptance
criteria are unchanged. Evidence:
`api/runtime/adapters/parts/evidence/TICKET04_EXACT_INPUT_RECOVERY_SEARCH_2026-09-28.md`.

Read-only source audit found an upstream `np.random.choice` pixel sampler and Modly's one-time NumPy seed before model/generator construction. This is a real global-RNG dependency but not an established cause: saved same-input runs had matching sampled-coordinate signatures while proposal counts differed. The audit points to the missing intermediate evidence: proposal scores/margins and kept candidate indices around the fixed IoU, stability, crop-edge and NMS stages. Full source anchors and interpretation: `api/runtime/adapters/parts/evidence/TICKET04_REPEATABILITY_SOURCE_AUDIT_2026-09-28.md`.

The opt-in proposal-filter trace is wired to the RX 7900 GRE probe and verified against the exact recovered Flamingo render/mesh/face-map bundle. Run 6 completed all 12 views and recorded 780 filter events, but left 17,549/46,180 faces unassigned. Run 7 used the same locked input in a fresh process, diverged at the first predicted-IoU event, and failed after view 1 produced no proposals. A 20-call, same-process proposal-only run returned 0 proposals nine times, 34 proposals ten times, and one proposal once. Its output regime changed sharply after repeat 10 alongside a VRAM-use change; cause is unknown. This is the same existing mesh/view, not a comparison of generated assets with different face counts. It does not score semantic quality or pass Ticket 04. Next investigate prompt-point, image-feature, raw-score, and synchronized-memory evidence before any inference-policy or threshold change. Details: `api/runtime/adapters/parts/evidence/TICKET04_PROPOSAL_FILTER_TRACE_F8_SAME_INPUT_2026-09-28.md` and `api/runtime/adapters/parts/evidence/TICKET04_PROPOSAL_ONLY_REPEAT_F8_2026-09-28.md`. Ticket 04 acceptance remains open and unchanged.

### Frozen-fixture score eligibility recheck (2026-09-28)

The two current-code exact-input requalification runs have identical frozen
mesh, topology, source/model/settings, seed, and rendered views; each produced
a complete, sentinel-free 1,536-face label array. The arrays differ, including
their region counts (3 versus 2), so the frozen repeatability prerequisite
fails. To preserve the one-shot truth boundary, no truth labels were opened and
the scorer was not invoked. Continue truth-free divergence investigation and
freeze an identical-output pair before scoring. Evidence:
`api/runtime/adapters/parts/evidence/TICKET04_KNOWN_TRUTH_SCORING_ELIGIBILITY_2026-09-28.md`
(SHA-256 `4c97c6231a46f3a64b9f86cb75d0fec60f3be7d356731c306c97b94ffcd09014`).

### Ticket 04 source-level diagnostic prioritization (2026-09-28)

Two independent, read-only audits found that a new prompt-coordinate/RNG-seed
probe would mostly repeat ruled-down evidence, while internal activation
localization is currently weakened by conflicting count reports across
instrumentation versions and child/parent outputs. The next focused action is
to independently double-check a few exact captured tensor finiteness counts
(device and transient CPU), preserving shape/layout/alias metadata and strict
payload-free caps, before relying on that trace. A separate single-frame
fresh/reused-predictor comparison is a follow-up to distinguish model-state,
cached-feature, and later prediction divergence. No code/settings/gates
changed and no truth or GPU was accessed. Evidence:
`api/runtime/adapters/parts/evidence/TICKET04_GEOSAM2_RNG_LIFECYCLE_SOURCE_AUDIT_2026-09-28.md`
and `api/runtime/adapters/parts/evidence/TICKET04_GEOSAM2_DIAGNOSTIC_COUNTER_AUDIT_2026-09-28.md`.

### Geometry-generation variation clarification (2026-09-28)

Per user clarification, source-image and generated-mesh similarity may be
imperfect because hidden sides/details require model assumptions, and separate
generation runs may produce different face counts. Treat each generated mesh
as its own topology-bound asset; do not compare face indices or use source
similarity as a substitute for semantic segmentation quality. The current
Ticket 04 repeatability diagnostic remains a same-saved-mesh segmentation check
under identical inputs/settings, which is a distinct gate.

### Ticket 04 dual-counter target cross-check (2026-09-28)

Two schema-v4 frame-0 prewarms on the same locked Flamingo input completed on
the RX 7900 GRE. At four fixed block-7 QKV tensors, device and independent
CPU/NumPy finite counts agreed exactly; all 13 existing Hiera boundary records
were fully finite in both runs. This supports the numeric instrumentation for
these captured tensors but does not explain full segmentation variation or
missing labels, and does not establish quality, memory, or acceptance. No
truth or segmentation was used. Exact hashes/commands and limitations:
`api/runtime/adapters/parts/evidence/TICKET04_GEOSAM2_DUAL_COUNTER_TARGET_CROSSCHECK_2026-09-28.md`.
The next bounded action is to validate a one-time finite-feature retry as a
candidate at the adapter seam, then test the unchanged registered workflow on
the frozen Ticket 04 mesh before considering any promotion. Generated assets
with different topologies remain separate assets and are not compared by face
index or source-image similarity.

### Ticket 04 lifecycle target result and car test scope (2026-09-28)

The same locked Flamingo mesh and view were checked in three fresh RX 7900 GRE
runtime processes. Each first feature extraction and prediction was
non-finite; resetting and repeating the exact view with restored random states
produced finite, byte-identical outputs across all three processes. Model and
random-state digests did not change. This narrows the current fault to a
repeatable first-use failure on this one view, but does not identify its cause
or qualify a general retry. The proposed one-time retry must be tested at the
real adapter path and through the frozen registered workflow before any
production change. No truth or quality score was used; Ticket 04 remains open.
Details: `api/runtime/adapters/parts/evidence/TICKET04_GEOSAM2_LIFECYCLE_TARGET_RESULTS_2026-09-28.md`.

The workspace has no identified car asset. A development car check must not
replace Ticket 03's frozen chair/flamingo/teapot acceptance set. Preferred
source is a user-owned or otherwise permission-cleared car mesh and observation
image. If none is supplied, a procedural car mesh with independently authored
face labels can be made as an explicitly synthetic development fixture. Record
source/recipe, rights, file digests, import settings, face correspondence, and
its own topology revision. Treat it as development evidence until applicable
ticket gates are dependency-ready and passed. The car seam-edit viewer test is
still Ticket 13 work, blocked on Ticket 12.

### Ticket 04 retry candidate and car development fixture (2026-09-28)

The opt-in finite-feature retry has fake-predictor coverage and records its
helper digest plus retry/failure counts in the run audit. It stays off by
default. The focused empty-proposal and retry-candidate suites pass 17/17; the
candidate helper is included in an inspected host-built wheel. The first
registered target run had zero invalid first passes and zero retries, and
failed raw coverage/repeatability. A second registered workflow attempt was
stopped while still processing GeoSAM2 connected components. Saved statuses
are `input_json=0`, `container=1`, `stdout_capture=130`, and `event_filter=130`;
stderr ends with `KeyboardInterrupt` in the connected-component CPU transfer.
It produced no new output artifacts, and no project worker/container process
remained after the stop. Neither run tested a successful retry. No Ticket 04
gate changes. Current adapter SHA-256:
`6d9a245728d88d355c2cc2da4d35c87e1b8dbb7075ecd992cac29562cd965621`;
helper SHA-256:
`16785f905e0e3f24074c6e451311d3f38d25b8a5e032ca110278e30e18541d1e`.
Full candidate and run evidence:
`api/runtime/adapters/parts/evidence/TICKET04_FINITE_FEATURE_RETRY_CANDIDATE_2026-09-28.md`;
`api/runtime/adapters/parts/evidence/TICKET04_FINITE_RETRY_REGISTERED_WORKFLOW_2026-09-28.md`.

A deterministic synthetic car development fixture is now available at
`api/runtime/adapters/parts/fixtures/car-development/`: 656 triangles and
topology-bound face categories for body, cabin, window, tire, rim, and
headlight. It has explicit synthetic/no-external-asset provenance and passed
its two structural tests. It is a simple overlapping-primitives model, not a
realistic car benchmark; it has not been rendered, run through segmentation,
or loaded in the Modly viewer. It can support later viewer seam-edit
development, while the frozen acceptance fixture set stays unchanged.
Generator SHA-256:
`f1512aca2c17457261fc60497d206e407c2ab6ed9c517b9054c229e0788f8c63`; manifest
SHA-256: `6af2cb88fe13ef03381173593b40c8ccb963f538e4ef414347890577ec5d7100`;
face-label SHA-256:
`88a70b0c7d638da965892f6fe38ecfa7a8b6e762b20c47162f7787462cf77d9f`; GLB
SHA-256:
`93671db566dd1dfaf45c4e1fd0ff4efccba29e1ca8bd2b09d4d6de79e57a803f`.

The first registered workflow run with the opt-in retry candidate emitted a
`done` event and saved all 48 digest-valid stage artifacts, but the wrapper
returned exit 1 and the exact cause was not captured. The candidate audit
recorded 12 feature checks, all finite, so it performed zero retries and did
not exercise recovery. Raw labels covered 768/1,536 faces; derived completion
filled the other 768, so strict raw coverage fails. Its one-region output
differs from the earlier two- and three-region outputs on the same fixture.
No truth was accessed or scored. Ticket 04 stays open; this run cannot qualify
the retry or the wrapper process. Details and hashes:
`api/runtime/adapters/parts/evidence/TICKET04_FINITE_RETRY_REGISTERED_WORKFLOW_2026-09-28.md`.

### User-directed car review and seam correction (2026-09-28)

Ticket 13 now records the user's added requirement: use a car asset for viewer
review; allow an interactive boundary point to be dragged along the surface,
previewed, confirmed/cancelled, and edited with undo/redo; preserve model
assertions and topology-bound correction history; update or flag affected
downstream mappings. Generated car meshes may differ from the source image and
from each other. User corrections may support later training, but storing a
correction alone does not mean the model learned; any actual model update must
be opted into, pinned/provenanced, and show held-out car improvement. No Ticket
13 implementation starts until Ticket 12's dependency gate is passed. The
current schema/UI gap audit is
`docs/orchestration/ticket13-seam-edit-design-gap-audit-2026-09-28.md`.

### Dependency-ready candidate refresh and finite-retry target probe (2026-09-28)

Tickets 04, 07, and 08 remain the dependency-ready frontier; no acceptance
gate changed. A new, locked Ticket 04 diagnostic ran only the fixed Flamingo
f8 view 0 on RX 7900 GRE `gfx1100`: the first feature pass was non-finite,
one identical retry produced finite cached features, and one fixed-point
prediction was finite. This is one isolated success, not a reliability or
segmentation result. The upstream optional `_C` post-processing extension was
unavailable and its prediction post-processing was skipped. The project API
wheel includes the diagnostic, helper, and lock. Its focused CPU suites pass
28/28; Python compilation, shell syntax, and lock checks pass. Ticket 04's
coverage, repeatability, quality, and acceptance gates remain open. Full
evidence: `api/runtime/adapters/parts/evidence/TICKET04_FINITE_RETRY_TARGET_PROBE_2026-09-28.md`.

The subsequent registered 12-view Flamingo workflow confirms that the retry
does not work for every view: view 0 registered 14 proposals, view 1 had
none, and the third `set_image` call remained non-finite after its one replay.
Modly failed closed before face labels or a segmentation manifest were
written. This run has no segmentation score or usable VRAM peak. The candidate
remains opt-in, and Ticket 04's requirements are unchanged. Full evidence:
`api/runtime/adapters/parts/evidence/TICKET04_FINITE_RETRY_MULTIVIEW_WORKFLOW_2026-09-28.md`.

Fresh primary-source screens also rejected COPS for Ticket 04, FMMC for
Ticket 07, and MatLat for Ticket 08 before acquisition or evaluation. Their
documented gaps include missing immutable code/checkpoint terms, CUDA-only
execution without RX 7900 GRE qualification, missing source-face/topology
binding, and/or missing required material behavior. No model weights or
fixture truth were accessed. Reports:
`api/runtime/adapters/parts/evidence/TICKET04_COPS_PRIMARY_SOURCE_SCREEN_2026-09-28.md`,
`api/runtime/adapters/material-identity/evidence/TICKET07_FMMC_CVPR2026_PRIMARY_SOURCE_SCREEN_2026-09-28.md`,
and `api/runtime/adapters/pbr/evidence/ticket08-matlat-official-source-screen-2026-09-28.md`.
Tickets 05 and 09–13 remain dependency-blocked; in particular, the requested
interactive car seam correction belongs to Ticket 13 after Ticket 12 passes.

### Ticket 04 view-2 retry follow-up (2026-09-28)

Two locked, truth-blind cache probes ran on the exact Flamingo mesh/render
bundle, GeoSAM2 checkpoint, dependency lock, and RX 7900 GRE runtime from the
registered failure. An isolated call on rendered view 2 had a non-finite first
pass and one successful identical replay. A separate ordered cache-only call
for views 0, 1, and 2 completed with finite features: view 0 needed one
successful replay; views 1 and 2 did not. The ordered probe omitted mask
proposals/predictions between views, so it does not reproduce the context in
which registered workflow view 2 failed. The feature-cache results suggest
the input image alone does not explain the failure; the effect of intervening
proposal work remains unknown. No segmentation, labels, or truth were accessed.
Ticket 04 remains open, the retry remains opt-in, and no thresholds or gates
changed. Evidence: `api/runtime/adapters/parts/evidence/TICKET04_FINITE_RETRY_VIEW2_SEQUENCE_DIAGNOSTIC_2026-09-28.md`.
The focused view-probe, retry, lifecycle, and empty-proposal suite passes
32/32; Python compilation, shell
syntax, immutable container identity, and project lock verification passed.

### Ticket 04 production-context retry follow-up (2026-09-28)

A locked RX 7900 GRE diagnostic used the actual Modly upstream seed loop, so
views 0 and 1 included automatic proposal generation and video propagation.
It stopped after view 2's image-feature setup returned, before view 2 proposal
generation, face lifting, labels, or scoring. View 0 generated/accepted 16
proposals, view 1 generated/accepted none, and all three feature trees were
finite with zero retry attempts. This run does not reproduce the earlier
registered view-2 failure, establish its variation rate, or qualify the
opt-in retry. Ticket 04 remains open and every acceptance criterion is
unchanged. Report and artifact hashes are in
`api/runtime/adapters/parts/evidence/TICKET04_FINITE_RETRY_VIEW2_SEQUENCE_DIAGNOSTIC_2026-09-28.md`.
The combined focused retry, view, and policy suite passes 31/31; Python
compilation and shell syntax checks pass. Two preserved setup-defect reports
are documented alongside the completed run.

### Ticket 07 SAMa source refresh (2026-09-28)

A bounded official-source screen found SAMa, an interactive 3D material
selection method that propagates visually similar material instances across
views. It does not classify those regions into Ticket 07's semantic material
identities or provide the required `unknown`/`ambiguous` behavior. Checkpoint
identity, acceptable rights, and RX 7900 GRE evidence are also missing. The
candidate was rejected before acquisition or evaluation; no dataset, weights,
fixture, truth, or GPU was accessed. Ticket 07 remains open with unchanged
gates. Evidence: `api/runtime/adapters/material-identity/evidence/TICKET07_SAMA_MATERIAL_AWARE_3D_SELECTION_SOURCE_REFRESH_2026-09-28.md`.

### Ticket 08 PBR3DGen source refresh (2026-09-28)

A bounded official-source screen rejected PBR3DGen before acquisition: its
checkpoint is not yet released, its described pipeline generates PBR content
and reconstructs meshes rather than estimating maps on Modly's existing mesh
and region IDs, and AMD/resource and frozen-quality evidence are unavailable.
No source code, weights, assets, fixture, truth, or GPU was accessed. Ticket
08 remains open with unchanged gates. Evidence:
`api/runtime/adapters/pbr/evidence/ticket08-pbr3dgen-official-source-screen-2026-09-28.md`.

### Ticket 04 full-tensor lifecycle rerun (2026-09-28)

The existing locked single-view lifecycle check ran again in a fresh RX 7900
GRE process. With the same view, model, settings, and restored RNG state, the
first predictor produced entirely non-finite cached features and predictions;
after resetting the predictor, features and predictions were fully finite.
A new predictor sharing the same model matched the recovered full-tensor
digests. Model parameter/buffer and RNG digests stayed identical throughout.
This confirms the first-use failure/recovery pattern for this one image in
another process; it does not explain why production-context view 2 was finite
or validate retry/warm-up for full segmentation. Ticket 04 remains open.
Evidence: `api/runtime/adapters/parts/evidence/TICKET04_GEOSAM2_LIFECYCLE_TARGET_RESULTS_2026-09-28.md`; report SHA-256 `6593de527e1855a587990de4436c9a9cd59db3d885abde1474ed07211adb72eb`.

Four additional production-context samples show the earlier view-2 behavior is
not stable: one stopped on view 1 because a retry stayed non-finite; three
reached view 2 with finite first-pass features. View 0 proposal counts ranged
from 23 to 34 accepted masks, and two runs with identical view 0/1 counts
produced identical partial view-1 outputs. Those label arrays contain sentinel
`999` for every face, so they are not usable segmentation. The updated probe
records hashes of partial files; its focused test suite passes 4/4. No truth
was accessed and no criterion changed. Full sample table, hashes, and limits:
`api/runtime/adapters/parts/evidence/TICKET04_FINITE_RETRY_VIEW2_SEQUENCE_DIAGNOSTIC_2026-09-28.md`.
The combined finite-retry, adapter-policy, lifecycle, and production-context
test suites pass 36/36; Python compilation and shell syntax checks pass.


### Ticket 04 all-call retry capture (2026-09-28)

A refreshed locked production-context probe captured every automatic image
setup and compared each input digest with the matching locked render view. On
the RX 7900 GRE, view 0 completed with 22/23 accepted/generated proposals.
View 1's first cached feature tensors and one replay were both entirely
non-finite; the adapter failed on setup call 2 with `NonFiniteFeaturesError`.
The diagnostic captured both calls and stopped before view 1 proposal generation,
view 2, face lifting, or final labels. It produced no partial mesh/label files.
This confirms an unresolved retry failure in the current ordered path; it does
not identify the cause or pass segmentation. No truth was mounted or read.
The diagnostic report SHA-256 is
`cecd3544b09de98cf8cb7b5e19a1c7501e869b550e6a0d4b7b689303e69453cc`; its lock
SHA-256 is `0f01ef7585fc53da8f4674f0e181ba5877017fa3a48fcafc85d0dd46a19d25b8`
and runner SHA-256 is
`e7910f08288e2a0b1a4d0509e88b4b33db3627298aa7e5450b593db91b54053a`. Four
focused tests pass; compilation and shell syntax checks pass. Ticket 04 remains
unaccepted, so dependent tickets do not advance.


### Ticket 04 retry-independent view capture (2026-09-28)

Independent review found the prior observer numbered rendered views using the
retry helper's underlying call count. It was correct for that specific report
(view 0 used no replay) but could drift after a successful earlier replay. The
probe now numbers outer seed-view setups independently and reports retry calls
separately; its CPU regression simulates a successful replay and confirms the
next view index is unchanged. A corrected fresh RX 7900 GRE run confirmed view 0
finite features and 24/25 accepted/generated proposals, followed by fully
non-finite view 1 features after one replay and `NonFiniteFeaturesError`. It
stopped before later proposals or segmentation output and produced no partial
labels; no truth was accessed. Report SHA-256:
`b22728e777b27d114e91b13df364d0d1cda26f69e959a02d82520658ff69c61f`. Probe
module/lock/runner hashes are recorded in the linked evidence note. All five
focused probe tests pass; Python compilation and shell syntax checks pass. This
corrects the diagnostic mapping but does not resolve Ticket 04 or change its
acceptance gates.

### Ticket 04 image-path boundary trace (2026-09-28)

A locked RX 7900 GRE lifecycle probe captured six image-model boundaries. On
the cold first image setup, non-finite values were already present in
`image_encoder` output `backbone_fpn[0]` (2,232,064/16,777,216 finite); the
position-map output was finite, while decoder `conv_s0` output was fully
non-finite. The same exact input was finite at every captured boundary after
predictor reset and through a new predictor. This is a cold-first-use
observation, not a cause or fix. No segmentation output or truth access.
Evidence: `api/runtime/adapters/parts/evidence/TICKET04_IMAGE_PATH_BOUNDARIES_F8_2026-09-28.md`.
The capture-method cleanup correction is covered by the five focused tests;
lock and shell syntax checks pass. Ticket 04 remains unaccepted, and Ticket 05
remains blocked.

Ticket 08 added and tested one source-shaped Kaolin face-gather operation
(6/6 generated CPU tests in the project AMD image). The code preserves UV
seams and separate UV/geometry indexing; it does not address rasterization or
integrate Material Anything. Ticket 08 remains open with its model, rights,
raster-parity, and quality gates unresolved. Evidence:
`api/runtime/adapters/pbr/evidence/material-anything-kaolin-gather-port-2026-09-28.md`.

### Ticket 04 five-run image-path repeat check (2026-09-28)

Five independent, current-lock RX 7900 GRE runs used the identical locked view
0 image inputs and seed. All 15 lifecycle calls had finite values at all six
captured boundaries. The earlier one-off cold-first encoder non-finite event
did not recur. First image-encoder output digests differed across all five
fresh processes; reset and new-predictor digests were each stable across the
five. This proves digest-level output variation only; hooks did not preserve
numeric summaries needed to estimate its magnitude or quality impact. No full
segmentation or truth access. Artifacts and limitations:
`api/runtime/adapters/parts/evidence/TICKET04_IMAGE_PATH_BOUNDARIES_FIVE_RUNS_2026-09-28.md`.
Ticket 04 remains unaccepted with gates unchanged.

### Ticket 04 image-encoder numeric summaries (2026-09-28)

Added bounded aggregate range/mean/spread reporting for the first image encoder
FPN tensor; no tensor values are retained. Three locked same-input RX 7900 GRE
runs had finite values in all 15 lifecycle calls. In each run the cold first
pass had a different summary from the reset/new-predictor path; the two warm
roles had identical summaries across all three runs. Cold first-pass summaries
also varied across runs. This is distribution-level diagnostic evidence only:
it does not identify the cause or prove that segmentation labels differ.
Five focused tests pass, and the updated helper, runner, lock, and shell pin
match. Full table, output hashes, and limitations:
`api/runtime/adapters/parts/evidence/TICKET04_IMAGE_PATH_NUMERIC_SUMMARIES_2026-09-28.md`.
Ticket 04 remains unaccepted; gates unchanged.

### Ticket 04 additional registered Flamingo failure (2026-09-28)

A saved RX 7900 GRE workflow attempt with the opt-in one-replay candidate
registered 14 proposals on view 0 and none on view 1, then stopped on a
non-finite image setup after its replay. The workflow reported
`GEOSAM2_INFERENCE_FAILED` and wrote no face labels. This larger-mesh run is
not the frozen known-truth scoring fixture and cannot establish quality or
repeatability. Its proposal audit, pipeline status, and hashes:
`api/runtime/adapters/parts/evidence/TICKET04_REGISTERED_RETRY_FAILURE_E7A3_2026-09-28.md`.
Ticket 04 remains unaccepted; no gate changed.

### Ticket 08 Material Anything provenance and weight-identity hold (2026-09-28)

The pinned Material Anything README says a significant portion of its code is
based on Text2Tex; Text2Tex's official repository identifies CC BY-NC-SA 3.0,
and Material Anything issue #18 asking about commercial scope/provenance
remains open. This is an unresolved rights question, not a legal conclusion.
Estimator/refiner file identities and local bytes are incomplete, and the
inherited Stable Diffusion 2.1 files/terms remain to be resolved. Weight
acquisition stays on hold; no source was run and no fixture or truth was
accessed. The source-shaped gather does not resolve UV-raster parity. Evidence:
`api/runtime/adapters/pbr/evidence/material-anything-text2tex-provenance-rights-screen-2026-09-28.md`.
Ticket 08 remains open; gates unchanged.

### Ticket 04 third frozen-fixture current-adapter sample (2026-09-28)

The RX 7900 GRE container completed and emitted a `done` event on the same
1,536-face fixture bytes, but the adapter SHA changed from the earlier pair,
so its output cannot be compared as a same-code repeat. Raw output was 768
label-29 faces and 768 sentinel-999 faces; the separate completion artifact
filled all faces to label 29. The host wrapper returned 1 because its output
filter rejected HIP attention-kernel descriptor debug lines. A narrowly
patterned, fail-closed filter update now accepts those exact numeric descriptor
formats; replay of the captured full stdout returns 0, and ten focused filter
tests pass. No fresh workflow was run after the parser-only change. The run's
raw coverage and quality remain unaccepted, and truth was not accessed. Full
record: `api/runtime/adapters/parts/evidence/TICKET04_FROZEN_FIXTURE_RUN3_2026-09-28.md`.
Ticket 04 remains open.

### Ticket 04 frozen-fixture run 4 and one-time score (2026-09-28)

Fresh RX 7900 GRE registered run `2ed3ab4d-7862-40a1-8793-df9e25504091`
completed all workflow steps with the event-filter fix in place. It used the
same 1,536-face mesh, topology, adapter digest, model/dependency locks, seed,
and render manifest as run 3. Raw label IDs differed (27 vs 29), but both raw
outputs labeled 768 faces and left 768 sentinel `999`; the pinned completion
step filled the remainder from that single label. The resulting canonical
label arrays are byte-identical, with one region. This satisfies the frozen
canonical-output repeatability prerequisite; it does not mean the upstream
label arrays or completion inputs match.

The unchanged known-truth scorer was invoked once after verifying those exact
canonical arrays. Macro-IoU is **0.50**, below the required **0.90**; coverage
is 1.0 and overlaps are zero. The quality gate fails, so no Ticket 04
acceptance claim is made. The raw 50% coverage is retained as a separate
limitation and is not hidden by completion. The first report-serialization
attempt failed after scoring because it expected a missing manifest `run_id`
field; scoring was not repeated. Run identities, hashes, single-call result,
and scope: `api/runtime/adapters/parts/evidence/TICKET04_FROZEN_FIXTURE_SCORE_RUN3_RUN4_2026-09-28.md`.

A separate truth-free audit traced why high 2D proposal counts do not become a
multi-region 3D result: 41–43 proposals were accepted/registered, but
per-view arrays show only the later six views produced any face labels, and
those contain one assigned class plus 50% sentinels. The multi-view projection
and voting stage can discard/merge distinctions; the available audit lacks
per-proposal survival and lift outcomes, so the exact cause remains unknown.
Completion can fill missing faces from the sole surviving class but cannot
invent another segment. No thresholds were changed and no truth was read.
Detailed artifact/source identities:
`api/runtime/adapters/parts/evidence/TICKET04_RUN3_RUN4_PROPOSAL_TO_LABEL_BOUNDARY_AUDIT_2026-09-28.md`.

### Ticket 07 new glass-only source screen (2026-09-28)

Two additional official material-segmentation leads were screened without
downloads or evaluation. GEM and AAAI26-MSNet produce binary glass masks,
which do not meet Ticket 07's five-material identity and unknown/ambiguous
requirements; AAAI26-MSNet also states academic research terms and documents a
CUDA/24 GB GPU route. Neither is eligible for a target probe. No code, model,
fixture, truth, or GPU was accessed. Ticket 07 remains open with gates
unchanged. Evidence and exact source references:
`api/runtime/adapters/material-identity/evidence/TICKET07_GLASS_ONLY_EXPERT_SOURCE_SCREEN_2026-09-28.md`.

### Ticket 04 SUV development sample and triangle reduction (2026-09-28)

The exact user-supplied 1536 × 1024 SUV PNG was saved under
`development-samples/suv-2026-09-28/` with SHA-256
`f5092fde7f8a18a2f0660fea6ac157c3826a7fdc485874e74ea91694aa8381af`.
Modly's pinned `reference-geometry` process extension completed on the RX 7900
GRE and registered a valid original Structured Asset: 1,108,620 triangles,
554,053 vertices, PyTorch ROCm with CPU offload, seed 1337, balanced detail.
The shaded preview shows a recognizable SUV with wheels, windows, bumper,
and roof equipment. The generated GLB remains geometry-only.

At the user's request, the active GeoSAM2 run on that original dense mesh was
stopped. It had produced twelve renders and its face correspondence but no
accepted segmentation result; the container and child processes exited.
Its artifacts target the original topology and cannot be reused for a reduced
mesh. No Ticket 04 acceptance claim is made.

A separate development copy was reduced with Blender 4.0.2 quadric edge
collapse and smooth normals with 45-degree sharp-edge preservation. Front
views compared 50,000, 75,000, 100,000, and 150,000-triangle candidates;
side and rear views compared the selected 75,000-triangle mesh with the
original, with a 100,000-triangle side/rear check. It has 40,184
vertices and is 92.91% smaller in bytes than the original. A new validated
Structured Asset records its independent topology revision, source-image and
original-geometry lineage, method, and tool digests. No original face labels,
mappings, or corrections were carried forward. Full paths, hashes, previews,
and stopped-run evidence are in `development-samples/suv-2026-09-28/README.md`.
Ticket 04 remains open; the next segmentation attempt, when authorized to run,
must use the reduced mesh with fresh renders and face correspondence and still
meet the unchanged frozen acceptance criteria independently.

### Ticket 07 MatSee and Visual2Echo source screen (2026-09-28)

The authors' MatSee and Visual2Echo sources were screened against the frozen
material-identity contract. MatSee has no documented pinned classifier or
checkpoint terms. Visual2Echo includes a 23-class material branch, but its
documented output and evaluation are for audio-depth prediction, and the
material checkpoint has no immutable published identity or terms in the
inspected source. Neither source establishes the required five-class selective
labels, topology-bound region output, frozen fixture quality, or RX 7900 GRE
resource evidence. No code, weights, fixture, truth, inference, or GPU was
accessed for this screen. Ticket 07 remains open; full evidence is in
`api/runtime/adapters/material-identity/evidence/TICKET07_MATSEE_VISUAL2ECHO_SOURCE_SCREEN_2026-09-28.md`.

### Ticket 08 live candidate qualification review (2026-09-28)

The current primary-source review found no candidate eligible to advance to
Ticket 08 acceptance evaluation. Material Anything still lacks resolved
Text2Tex-derived use/provenance terms, complete estimator/refiner/base asset
identities, and full Kaolin UV raster parity on the target path. SuperMat
remains held on base-model terms, full consumed-artifact identities,
dependencies, and topology-bound fusion. Existing fixed-geometry inverse
rendering candidates fail the frozen per-channel quality gates; other screened
routes do not establish the complete AMD, topology, and PBR contract. No
weights, fixture truth, or GPU were used in this review. The unchanged gate
and sources are detailed in
`api/runtime/adapters/pbr/evidence/ticket08-live-primary-source-qualification-2026-09-28.md`.

### Ticket 04 proposal-to-lift diagnostic integration (2026-09-28)

The GeoSAM2 adapter now has an opt-in, locked, bounded trace of accepted
proposals, predictor registrations, and face-lift inputs. A propagated object
ID is joined across rendered views as one global run ID. The trace stores
counts, shapes, keyed aliases, and aggregate summaries rather than masks,
coordinates, face IDs, or raw label vectors. It was validated with 19 focused
tests, Python compilation, a synthetic cross-view join, and exact helper-lock
verification. No GPU run or frozen truth was used. Full evidence and digests
are in `api/runtime/adapters/parts/evidence/TICKET04_PROPOSAL_LIFT_TRACE_INTEGRATION_2026-09-28.md`.
The frozen 0.50 macro-IoU still fails the 0.90 requirement; Ticket 04 and its
dependent Ticket 05 remain open. A future authorized same-input run must prove
that the trace is complete and locate where distinct proposals are lost.

### Ticket 04 trace correction and workflow wiring (2026-09-28)

Independent review found that the first trace could claim completion with no
lift calls and could renumber surviving proposal summaries after an omitted
mask. Schema v2 now records expected, called, successful, and failed lift
passes for each view and preserves original proposal ordinals, including
explicit unavailable slots. The adapter supplies the pinned flow's expected
lift counts. The focused suite passed 21/21 and Python compilation passed.
The workflow launcher now lets the adapter install that trace directly,
avoiding its previous second wrapper; it still mounts Modly's renderer script.
`bash -n scripts/modly-amd-runtime.sh` and Python compilation of the retained
standalone research bootstrap passed. No frozen gate changed. Source evidence:
`api/runtime/adapters/parts/evidence/TICKET04_PROPOSAL_LIFT_TRACE_INTEGRATION_2026-09-28.md`.

### Ticket 07 FMMC release correction (2026-09-28)

The author project now links a public MIT code repository and trained-model
log download links, superseding an earlier no-public-code finding. The
published DINOv2 evaluation route uses an eight-class head that omits glass
and paint/plaster/enamel and has no frozen abstention behavior. Checkpoint
identity/terms, topology binding, AMD execution, and target quality remain
unproved. No assets, fixture, truth, or GPU were accessed in the source
review. Ticket 07 remains open. Evidence:
`api/runtime/adapters/material-identity/evidence/TICKET07_FMMC_RELEASE_RECHECK_2026-09-28.md`.

### Ticket 08 Material Anything atlas contract extension (2026-09-28)

The source-visible CPU contract now captures the downstream image-derived
coverage mask after XYZ-to-RGB conversion and five dilation passes. This
exposes a covered-white collision and newly filled originally uncovered
pixels that face-ID coverage alone misses. The focused suite passed 10/10.
Kaolin raster and OpenCV border parity, model rights/identities, AMD execution,
and PBR quality remain open. Evidence:
`api/runtime/adapters/pbr/evidence/material-anything-uv-reference-contract-2026-09-28.md`.

### Shared GPU desktop-impact stop (2026-09-28)

The user reported desktop rendering impact during the reduced-SUV GeoSAM2
run and requested that all task processes stop. The run was interrupted and
its partial artifacts are not an accepted segmentation result. It also
reported a PyTorch ROCm out-of-memory request of 15.05 GiB on the 16 GiB
RX 7900 GRE. Host inspection found desktop applications and Xwayland using
that same GPU; project Podman and KFD checks found no remaining task process
after the stop. A project pause marker now blocks GPU launcher commands and
the reference geometry/part-segmentation process extensions before model
load. CPU-only guard checks passed. GPU work must remain paused while the
shared-device impact and bounded-resource policy are resolved. Full local
evidence and limits: `docs/orchestration/2026-09-28-shared-gpu-desktop-impact.md`.

### Shared-GPU budget and GeoSAM2 CPU offload candidate (2026-09-28)

After the user's game-impact report, GPU execution remains paused. The
geometry and GeoSAM2 adapters now apply a live-free-memory PyTorch allocator
limit before model load, hold back 4 GiB of observed free VRAM, and fail
closed when the limit cannot be applied or headroom is below 2 GiB. GeoSAM2
can opt into the pinned predictor's video/state CPU storage flags and records
whether the state honored both. The workflow launcher forwards that opt-in.
CPU-only budget, offload, and focused adapter tests passed 23/23; Python
compilation and launcher syntax checks passed. These checks do not show that
desktop graphics or Golf With Your Friends can coexist with a real run, nor
do they establish Ticket 04 quality. The pause marker remains in place and
Ticket 04's frozen 0.50 macro-IoU still fails its 0.90 gate. Details:
`docs/orchestration/2026-09-28-shared-gpu-desktop-impact.md` and
`api/runtime/amd/RUNTIME.md`.

### Ticket 04 run-output containment (2026-09-28)

A CPU-only audit of the registered part-segmentation process found that its
output root used ordinary path joins and final sidecar publication used a
replacing rename. Both P3-SAM and GeoSAM2 now create new run folders
through directory handles that reject symlink parents, reject run-ID reuse,
map arbitrary asset IDs to safe filename components, and publish sidecars
without replacing an existing final file. Six focused path tests passed;
the combined path/budget/offload set passed 16/16, and the Ticket 03 plus
GeoSAM2 focused regression set passed 18/18 after resolving an import-path
test harness conflict. Python compilation passed. No GPU work or Steam file
access occurred. This is preventive containment, not a cause finding for the
game report or evidence that Ticket 04 quality passes. The GPU pause and the
frozen 0.90 macro-IoU gate remain unchanged. Incident detail:
`docs/orchestration/2026-09-28-shared-gpu-desktop-impact.md`.

### Ticket 04 failed-SUV allocation trace (2026-09-28)

A CPU-only review of the retained 75,000-face SUV run found the last completed
work at view 2: 59/70/27 accepted proposals on views 0/1/2, followed by no
view-3 proposal or lift. The container stderr has no Python traceback for the
15.05 GiB OOM request. The pinned seed loop next attempts view-3 automatic
proposals, but no retained evidence names the failing internal operator.
Future inference failures now add at most 24 call frames with file, line,
and function to the durable proposal audit, without local variables or masks.
The focused audit suite passed 8/8 CPU-only checks and Python compilation
passed. No GPU process was started. Ticket 04 and the shared-GPU pause remain
open; details are in the incident record.

### AMD pause entry-point closure (2026-09-28)

Direct geometry and part process extensions now check the source project
pause marker as well as the input workspace marker before adapter import.
The project launcher fails closed for any command outside its non-GPU
build/status set while paused. CPU-only marker and GeoSAM2 failure-audit
checks passed 11/11, both processors and the adapter compiled, launcher
syntax passed, and an unlisted launcher command exited 78. No Podman or GPU
work was started. This closes the identified external-workspace pause bypass;
the marker still cannot guarantee desktop coexistence. Ticket 04 quality and
the RX 7900 GRE acceptance gates remain open.

### Ticket 07 material-identity output isolation (2026-09-28)

The material-identity process previously published its Structured Asset to
one shared filename derived from `asset_id` with a replacing rename. It now
creates a unique `StructuredAssets/material-identity-runs/<run_id>/` folder
before model inference and publishes `structured-asset.json` only if that
name is new. Directory handles reject symlinked output parents, and existing
digest-addressed evidence is reusable only when its bytes match. Five focused
CPU-only output tests passed; the Ticket 07 API contract suite passed 14 with
one optional real-model integration skip, and Python compilation passed.
The reported game impact remains unexplained. No GPU work or game-file access
occurred, and Ticket 07's quality/abstention/coverage/hardware gates remain
open. The audited handoff files were left unchanged.

### Ticket 04 diagnostics and Ticket 07/08 protocol hardening (2026-10-01)

Ticket 04's opt-in diagnostic telemetry now records bounded generated-mask
summaries and RNG-state fingerprints with explicit capture status, so an
unavailable digest cannot appear to be a valid match. The versioned telemetry
and proposal/lift trace locks were updated. On the repeat v7 run, geometry
renders and input payloads matched, views 1–10 produced identical label
arrays, and view 11 differed on 768 of 1,536 faces; completion filled the
repeat's remaining sentinel labels and collapsed the final result to one
region. The live v7 records do not contain per-view RNG fingerprints, so
repeatable RNG state is not established. Ticket 04 remains open: its
repeatability precondition and frozen quality gate have not passed. Focused
Ticket 04/07 tests passed 46/46 on the published-source clone with ROCm
PyTorch available; the project Python 3.12 test environment lacks PyTorch.

Ticket 07's project-owned classifier now rejects conflicting cross-view
labels for one object/region, supported-versus-abstention conflicts, and
duplicate sample IDs. Its classifier and development evaluator tests passed
14/14, and node integration tests passed 12/12. These data-integrity checks
do not pass Ticket 07's model quality or AMD acceptance gates.

Ticket 08 now has a frozen candidate runner limited to training members and a
separate one-shot scorer that verifies candidate identities before opening
target arrays. Protocol tests passed 3/3. The frozen fixture has no validated
topology-bound per-face material-region map; no map was derived from target
labels, and no frozen target arrays were accessed. Therefore the real Ticket
08 candidate run and score remain blocked. The three-ticket changes were
checked in the publish clone with `git diff --check`; no GPU work was used.

### Ticket 04 v8 monitored GPU diagnostic attempts (2026-10-01)

After the bounded v8 diagnostics and focused tests passed, two same-input
attempts were made through Modly's registered GeoSAM2 workflow using separate
new workspaces and a two-second board-wide VRAM monitor with a 4 GiB stop
reserve. The first attempt stopped safely when PyTorch rejected a 9.50 GiB
allocation with 8.57 GiB allowed. A second attempt enabled the existing CPU
offload option and stopped safely when PyTorch rejected a 7.88 GiB allocation
with 8.57 GiB allowed. Monitor peaks were 7,767,351,296 and 7,417,511,936
bytes of total board use; minimum sampled free memory was 9,395,740,672 bytes
on the first attempt and remained above 4 GiB on the second. Both returned to
baseline and no KFD process remained. Neither reached face-label generation,
so neither yielded segmentation or RNG-state comparison evidence. No limit was
raised to force a result. These attempts do not meet Ticket 04 acceptance.
Evidence and isolated partial artifacts are under
`.modly-amd-runtime/ticket04-v8-rng-diag-20261001-run{3,4}/` and remain
excluded from publication.


### Ticket 08 source-bound region map and frozen v2 score (2026-10-01)

The missing per-face candidate map was built from a real Modly Ticket 06
`segment-material-regions` stage over the four allowed training views. Its
source segmenter is still provisional deterministic RGB appearance clustering,
and its confidence is uncalibrated. The source GLB and frozen fixture GLB have
identical SHA-256 bytes and six identical indexed faces. Their recorded topology
revisions differ, so the source map was explicitly rebound only after the
frozen GLB accessors recomputed the frozen revision; source face IDs were kept
unchanged and region IDs were recomputed for that revision. The full source,
stage, map, and rebind hashes are in
`api/runtime/adapters/pbr/evidence/ticket08-region-inverse-v2-source-bound-score-2026-10-01.md`.

The frozen training-only runner completed on CPU, opened no target/held-out
arrays, and reported 0 accelerator devices with peak host RSS 209,580 KiB. The
one-shot scorer verified the saved run and map before opening only the frozen
target allowlist. It rejected the candidate: base-color MAE 0.12485 / SSIM
0.55152, roughness MAE 0.38590, metallic MAE 0.26881, and conductor/dielectric
bias 0.41947 all fail their unchanged gates; novel-light MAE 0.04840 passes.
Coverage 1.0 remains informational. This does not pass Ticket 08 or the RX
7900 GRE gate. The frozen lock's forward-renderer SHA had a one-character
transcription error; it was corrected to the measured source hash and the
focused frozen protocol tests passed 3/3.
