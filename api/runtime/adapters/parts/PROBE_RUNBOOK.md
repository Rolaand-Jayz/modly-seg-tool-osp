# Ticket 04 target probe runbook

## First-convolution digest trace (locked, GPU-held)

`GEOSAM2_FIRST_CONV_LOCK.v1.json` and the opt-in `encoder-first-conv` mode add a
versioned, digest-only capture at the pinned Hiera patch embedding. It records
the image tensor entering `image_encoder.trunk.patch_embed`, the exact tensor
entering `image_encoder.trunk.patch_embed.proj` (`Conv2d`), that convolution's
output, and per-call weight/bias digests. It also records read-only autocast,
deterministic-algorithm, and TF32 flags at the convolution input boundary.
Transient tensor copies are capped at 64 MiB each; tensor payloads are not
written. This trace does not alter precision or determinism settings and is
diagnostic evidence only.

The project's GPU pause is active, so this command is currently expected to
exit with status 78 before creating a container. Do not remove or bypass the
pause marker to run it. Once the hold has been cleared through the project's
normal process, run the host wrapper from the project root with a new output
directory:

```sh
scripts/modly-amd-runtime.sh trace-geosam2-lifecycle "$PWD" \
  .modly-amd-runtime/results/ticket04-first-conv-next encoder-first-conv
```

The host wrapper checks the pause marker first, then verifies the added lock
and runner identity. The probe also revalidates the existing source, model,
dependency, lifecycle, image-boundary, and Hiera-stage locks before loading the
pinned Flamingo view-zero input. Read the resulting
`lifecycle-diagnostic.json` field `image_encoder_first_conv` alongside the
existing lifecycle and encoder-stage reports. Compare per-role hashes and
flags; a difference localizes a changed boundary but does not by itself prove
the source of the variation or pass Ticket 04.

The probe records measured acceptance; source inspection and unit tests do not
substitute for it. Run it only after the shared RX 7900 GRE slot is released.
It performs no downloads. It requires a project-managed ROCm/MIGraphX image and
hash-verified source/checkpoints already staged in the project-owned cache.

## Candidate identities

P3-SAM source is Tencent Hunyuan3D-Part commit
`e96be065375438962375b55326416291342958a7`. P3-SAM weights are Hugging Face
revision `ee23ed4fb349b9144a061297acff7122927faf5d`, file
`p3sam/p3sam.safetensors`, SHA-256
`eb76550cfbe06f154c6e9b17167ccfc28222bb4a216ec7b12ac2bf7d762de38c`.
Sonata weights are revision `0e6f990a2f7bfdffc9d2f58139cfa2b883d8e8cb`,
file `sonata.pth`, SHA-256
`c5ced5acdae30d1c469713398073a866e25e6e414e23feed5dc025373657ac50`.
The combined files total 884,976,331 bytes. The adapter checks file length and
both hashes before model load; the report also includes the canonical composite
weights-manifest digest.

The alternate is PartField source commit
`373025dbd283bb44cc4a6dc78c99994dbc91de32` with `model_objaverse.ckpt` at
Hugging Face revision `f8cda8fd7dcef0596654015a482cc89407977a29`, 1,243,631,761
bytes, SHA-256
`463efc8a3afd3913142aa025e0125c00f16ef452b8de6a132ebe32bbe7877ee4`. This
weight is pickle serialized; inspect its origin and use terms before any
execution. Its upstream NVIDIA license limits third-party use to noncommercial
research or education, and its published dependency list uses a CUDA-specific
`torch-scatter` wheel. It is a comparison candidate only until target evidence
and the license scope are reviewed.

## Bounded P3-SAM run

The command uses the existing project image
`localhost/modly-amd-migraphx:ticket02`; it does not touch a global Python or
ROCm installation. From the repository root, after setting the three model
paths to the already hash-verified project cache files, run the command below
in the approved host context that permits device passthrough:

```bash
ROOT_DIR="$(pwd)"
PODMAN_ROOT="$ROOT_DIR/.modly-amd-runtime/storage"
PODMAN_RUNROOT="$ROOT_DIR/.modly-amd-runtime/run"
MODEL_CACHE="$ROOT_DIR/.modly-amd-runtime/models"
RESULTS="$ROOT_DIR/.modly-amd-runtime/results"
mkdir -p "$RESULTS"
podman --root "$PODMAN_ROOT" --runroot "$PODMAN_RUNROOT" run --rm \
  --network=none \
  --device /dev/kfd --device /dev/dri --group-add video --ipc=host \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  --volume "$ROOT_DIR/api:/modly/api:ro" \
  --volume "$ROOT_DIR/.modly-amd-runtime/api-test-venv:/opt/modly-test-venv:ro" \
  --volume "$MODEL_CACHE:/models:ro" \
  --volume "$RESULTS:/results:rw" \
  --env PYTHONPATH=/modly/api:/opt/modly-test-venv/lib/python3.12/site-packages:/models/hunyuan-mini-turbo/python-packages:/models/triposr/python-packages:/opt/rocm/lib \
  --env MODLY_API_DIR=/modly/api \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env MODLY_P3SAM_SOURCE=/models/hunyuan3d-part-e96be065375438962375b55326416291342958a7 \
  --env MODLY_P3SAM_WEIGHTS=/models/p3sam/ee23ed4fb349b9144a061297acff7122927faf5d/p3sam/p3sam.safetensors \
  --env MODLY_SONATA_WEIGHTS=/models/sonata/0e6f990a2f7bfdffc9d2f58139cfa2b883d8e8cb/sonata.pth \
  localhost/modly-amd-migraphx:ticket02 \
  python -m runtime.adapters.parts.probe \
    --workspace /results \
    --output /results/ticket04-p3sam-probe.json
```

The source/checkpoint paths above are the expected project-cache layout. Do not
create those files from this command. Before running, confirm the cache really
contains the pinned clean checkout and both exact files; the adapter rejects
missing, dirty, or mismatched inputs.

The low-memory probe is fixed at 10,000 points, 32 prompts, batch size 4, seed
42, and threshold 0.95. It runs one cold inference to qualify ten dense modules
(`mlp`, stage-1/2 segmentation, and IoU MLPs) and one warm repeat through the
selected callables. Every dense module is sent through
`AMDInferenceRuntime.run_region()` and the actual selected callable is invoked
through `AMDInferenceRuntime.execution_callable()`. The report includes source
and both weight identities, ROCm/PyTorch/device facts, observed native module
and loaded shared-object inventory, CUDA/NVIDIA artifact findings, fixture GLB
hash/topology revision, per-module backend and runtime reports, cold/warm
latency, allocated and reserved VRAM, canonical mask digest, face coverage,
overlap, macro IoU, and gate results. The fixture GLB and Structured Asset
import sidecar remain under `StructuredAssets/` in the results mount.

### Frozen upstream-density chunked comparison (2026-09-25)

After the low-memory preset failed its fixed quality gate and the official
100,000-point/400-prompt/batch-32 preset OOMed, one bounded comparison was
predeclared with the same upstream point and prompt counts and only prompt
batch size reduced to 4. This changes execution chunking to reduce activation
memory; it does not change the model, weights, config, threshold, seed, or
quality gate. It is a single target run and will not be tuned against fixture
truth. The probe mode is `upstream_chunked` in `runtime.adapters.parts.probe`.
During target preflight, MIGraphX terminated on a zero-element dynamic tensor
lowering (`tosa.reshape` from `tensor<0xf32>`) before a candidate output was
produced. For the data-producing run, explicitly set
`MODLY_P3SAM_BACKEND=pytorch_rocm`; this is the supported fallback justified
by the recorded compiler incompatibility, with no model or quality setting
change. Do not retry MIGraphX for this candidate without a source/runtime fix.

The in-tree adapter installs a narrowly scoped `spconv.pytorch` compatibility
namespace (`SparseConvTensor`, `replace_feature`, `SubMConv3d`, and Sonata's
module predicate) plus `torch_scatter.segment_csr` before importing vendored
Sonata. Its CPU reference and output wiring tests are recorded in
`evidence/ticket04-sonata-ops-cpu-tests.md`. This does not prove ROCm execution,
performance, or parity. The adapter also applies the hash-pinned
`SONATA_CONFIG_OVERRIDE.json` (`enable_flash=false`) through Sonata's public
`custom_config` argument, selecting its existing pure-PyTorch attention path;
upstream source/config/checkpoint bytes remain untouched.

Pinned Sonata's only observed `addict.Dict` use is Point mapping construction;
the adapter provides a local compatibility subset for mapping construction,
attribute/item access and assignment, attribute deletion, and recursive
mapping conversion. It is reported under provider
`modly.adapter.sonata_dict_compat`; it is not the upstream `addict` package and
does not claim its full API. Method-name collisions resolve to the inherited
mapping method, matching Python attribute lookup behavior. Any newly observed
upstream use outside the recorded subset blocks import until reviewed.

The selected Sonata files contain one `timm` import (`timm.layers.DropPath`),
instantiated in the encoder block and applied by `PointSequential` to
`Point.feat` tensors. The local adapter exposes only that operator. Eval mode
and zero drop probability return the input unchanged; training uses one
Bernoulli mask per leading tensor row and scales kept rows by inverse keep
probability. This matches the sibling per-row `DropPath` implementation in
the pinned source; flattened points therefore receive per-point masks during
training. Provider is `modly.adapter.sonata_timm_compat`, not the upstream
`timm` package. Other `timm` APIs are unsupported and remain unavailable.

The pinned transform module imports SciPy only for
`ElasticDistortion.elastic_distortion()`: three `scipy.ndimage.filters.convolve`
calls per smoothing pass and `scipy.interpolate.RegularGridInterpolator`.
P3-SAM's `sonata.transform.default()` config contains only `CenterShift`,
`GridSample`, `NormalizeColor`, `ToTensor`, and `Collect`; it does not include
`ElasticDistortion`. A hash-locked in-memory source overlay relocates the four
SciPy imports from module scope into that method and raises
`SCIPY_REQUIRED_FOR_ELASTIC_DISTORTION` if it is explicitly invoked without
SciPy. The numerical statements are unchanged and upstream bytes stay intact.
Source SHA-256 is
`9e04bbadec8f7519463eb640ac316c55f9c20725d325e8a7b1046bbf246a779c`; overlay
manifest SHA-256 is
`bfa300fa8e1458c345dc4e1d45d4d041fd31fa4b9318d30ee7c01a26c5b3af90`. Runtime
reports also include the hash of the resulting effective source. SciPy is
therefore optional for the selected default path but remains required for the
separate elastic-distortion augmentation.

PCA in P3-SAM's `mesh_sam` is used only to write debug outputs inside
`if save_mid_res`; the adapter passes `save_mid_res=False` explicitly. A
hash-locked in-memory auto-mask overlay moves only the scikit-learn PCA import
into that branch and raises `SKLEARN_REQUIRED_FOR_P3SAM_DEBUG_PCA` if debug
output is explicitly requested without scikit-learn. Its upstream source
SHA-256 is `8fea0b770880462c97e5bedab017b6b36f447d0c88cc6adaaa55cfa5f362dea8`;
manifest SHA-256 is
`627bd6c7983de1281d4bdb5a40fd75f92b91b283c18cc5dd5c159ba489623aaa`. The PCA
calculation and output behavior are unchanged. The same overlay routes FPS
and face adjacency to hash-identified adapter NumPy implementations.

FPS sampling is invoked unconditionally by `mesh_sam`, which requests 400
prompts from 100,000 points. Its two-argument call is source-matched to
`fpsample==1.0.2` / commit `4124a21dc664c3ee745e3083833d310da814453b` by
`runtime.adapters.parts.p3sam_fpsample`; exact output parity and global RNG
draw parity against the compiled release extension are recorded in
`evidence/ticket04-fpsample-numba-cpu-screen.md`. The helper supports the
observed single-start `None` or Python-int API only. The adapter calls P3-SAM's
`set_seed(seed)` before each inference pass, fixing the upstream global NumPy
start draw. Upstream Numba face adjacency is likewise replaced by a tested
source-order-preserving NumPy operation; neither distribution is required by
the overlaid selected path. Target model inference and ROCm qualification
remain pending. Chamfer is used by the separate `auto_mask_no_postprocess.py`
path, not the selected automatic runner's direct imports or calls.

The source README documents `P3-SAM/utils/chamfer3D` as a CUDAExtension built
from `chamfer3D.cu`; it must not be built with CUDA in the AMD image. Pinned
`P3-SAM/model.py` and the automatic inference module do not directly reference
Chamfer, but transitive use has not yet been ruled out by target execution. If
the accepted inference path imports or requires the CUDA operator, P3-SAM fails
the native dependency gate until the actual dependency is replaced by an AMD
implementation. The probe's inventory is retained even when import/inference
fails.

The upstream requirements still name `spconv-cu124` and a CUDA-specific
`torch_scatter` wheel. Those packages are not installed. The adapter substitutes
its tested in-tree tensor subset for only the selected Sonata call sites; probe
reports must distinguish these modules from installed distributions and leave
target ROCm qualification pending until an actual hardware run. The older
`ticket04-p3sam-amd-probe.json` snapshot predates these adapter-local shims and
must not be used as the current runtime inventory. Its upstream pins remain
valid source facts. Do not install either CUDA package to force candidate
execution through.

## Decision after P3-SAM

Do not try upstream default settings unless the low-memory P3-SAM run passes
the fixed 0.90 macro-IoU, exact coverage, no-overlap, deterministic-repeat,
no-CUDA, runtime-routing, and 16 GiB reserved-memory gates. Save the report and
exit status. Before loading PartField, terminate the process, verify memory
returns to the pre-run baseline, confirm its revision/hash/license, and build
an AMD-safe runtime/dependency route. Record its CUDA-specific package
requirements as blockers rather than installing NVIDIA binaries. Compare the
same fixture, metric, label-map digest, VRAM limits, and runtime/native
dependency evidence. The selected segment-parts capability contract remains
unchanged.
# GeoSAM2 all-rendered-view automatic proposal candidate (2026-09-25)

The fixed `opposite_views` candidate used seed views 0 and 6 and left half of
the fixture faces at GeoSAM2's unassigned sentinel. Before any further run, a
separate proposal-sampling candidate is frozen as `all_rendered_views`: use
one start frame `[0]` and seed each of the 12 rendered views exactly once via
`start_to_seed_views={0: [0, 1, ..., 11]}`. Keep source, checkpoint, mesh,
render bundle, seed 42, automatic mask-generator thresholds, postprocessing,
and all acceptance gates unchanged. This policy uses only the candidate's
available rendered views and does not consult fixture truth. Run two separate
single-inference processes (`--run-count 1`, distinct output directories) to
release process-local inference state between repeats; freeze both truth-free
outputs before checking coverage. If either pass leaves any face unassigned,
do not invoke the truth scorer. If both passes have full coverage and identical
labels, invoke the existing scorer once on the frozen output; all remaining
macro-IoU, repeatability, topology, and 16 GiB gates still apply.

The frozen settings and runner schedule are covered by
`api/tests/test_ticket04_geosam2_probe_schedule.py`. This is an additional
candidate screen and does not replace or reinterpret the rejected
`opposite_views` result.

## Production renderer repeatability correction (2026-09-25)

Integration review found that the pinned `geosam2_render.py` calls
`random.seed()` without a value while sampling the 12 camera poses. The
candidate's two inference repeats reused one already-rendered bundle, so they
did not verify cross-process render repeatability. The project wrapper
`geosam2_render_entry.py` now scopes a deterministic override for that exact
no-argument seed call to Python seed 42, restores the standard function after
the upstream render returns, and records the seed in `render_manifest.json`.
The upstream source and model identities, render count/resolution/samples,
all-view inference schedule, quality metric, thresholds, and acceptance gates
are unchanged. `geosam2.py` rejects render manifests that omit or mismatch this
policy. Before claiming production repeatability or Ticket 04 acceptance,
render the same canonical mesh in two separate processes and compare every
artifact digest, then run the unchanged frozen quality scorer on the
truth-blind inference outputs.

## GeoSAM2 ROCm allocator recovery screen (2026-09-25)

The first complete Modly process-extension inference using the live renderer
failed during the third automatic view while requesting a 5 GiB allocation;
PyTorch reported 9.20 GiB reserved but unallocated and explicitly suggested
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. A single allocator-only
recovery screen is predeclared using that exact setting. It leaves source,
checkpoint, deterministic seed-42 render bundle, view schedule, inference
parameters, quality metric, and every acceptance threshold unchanged. It runs
once through the full process extension. Record allocator telemetry, output
completeness, deterministic output, and frozen quality. Do not sweep other
allocator or inference settings from this screen.

### Repeat comparison for topology-bound output IDs

Review of two complete separate-process GeoSAM2 outputs under the deterministic
renderer showed the same exact face partition with upstream label numbers
shifted by +1 between processes. These are model-local object labels; they are
not Modly region identities. The acceptance comparison therefore remains exact
repeatability of the face-to-part assignment: normalize each complete partition
to contiguous integer labels ordered by its sorted canonical face membership,
then require byte-identical normalized arrays. Persist the untouched upstream
label array and its per-region ID mapping as separate evidence. The public
Structured Asset region IDs remain the existing digest of topology revision
and exact face membership. This representation correction preserves the
same disjoint face partition and identity evidence; it changes no quality,
coverage, repeatability, latency, memory, or unsupported-geometry threshold.
