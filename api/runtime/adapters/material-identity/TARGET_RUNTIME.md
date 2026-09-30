# Material identity target runtime and entrypoint record

## Modly entrypoint

- Process extension: `src/areas/workflows/nodes/reference-material-identity/manifest.json`
- JSON-lines executable: `src/areas/workflows/nodes/reference-material-identity/processor.py`
- Classifier implementation: `api/runtime/adapters/material-identity/classifier.py`
- The node runs as a Modly Python process extension and consumes the incoming
  Structured Asset, source-observation artifacts, camera-aligned face maps, and
  an explicitly selected candidate with its pinned model artifact configuration.
- DMS46 requests without `materialInference`, the exact upstream source
  commit, a workspace-local weight file/digest, or a workspace-local taxonomy
  file/digest fail closed. The extension rejects the old `materialPredictions`
  input route and does not fabricate output when classifier weights are absent.
  The workflow's `candidate_id` selector and `materialInference` config must
  agree when both provide a candidate identity; candidate conflicts fail closed.

## Candidate runtime requirements

The project AMD runtime evidence records the target image
`localhost/modly-amd-migraphx:ticket02`, derived from
`rocm/pytorch:rocm7.14_ubuntu24.04_py3.12_pytorch_release_2.11.0`, with Python
3.12.3, PyTorch `2.11.0+rocm7.14.0`, ROCm/HIP `7.14.60850`, MIGraphX
`2.16.0.dev+20250912-17-575-g4bcfe75b2`, and Torch-MIGraphX `1.2`. That is the
only candidate target tuple declared here; Ticket 02 runtime acceptance does
not qualify DMS46 itself.

Required Python interfaces for this adapter are `torch` with TorchScript,
`torch_migraphx` (preferred compile backend), Modly's
`services.amd_runtime.AMDInferenceRuntime`, NumPy, and Pillow. The adapter uses
PIL RGB decode/resize and NumPy for dense map transfer; it does not import
OpenCV, torchvision, PyTorch3D, CUDA extensions, HIP custom kernels, or Vulkan.
On AMD the `AMDInferenceRuntime` first evaluates the model through eager
PyTorch ROCm and then attempts the declared `torch.compile(backend="migraphx")`
candidate under its numerical and latency gates. If MIGraphX fails those
runtime gates, the whole DMS module falls back explicitly to PyTorch ROCm. CPU
is visible development fallback only and does not pass Ticket 07's AMD gate.

The local Python 3.12 API test environment does not install torch,
Torch-MIGraphX, MIGraphX, or Pillow; API acceptance tests therefore exercise
the extension protocol, topology/identity aggregation, and fail-closed model
pin checks only. The dedicated AMD container has the neural runtime but has not
yet loaded DMS46. No additional native DMS dependency was observed in the
official inference code used for the adapter transform. The new SigLIP2 route
has run through the process extension inside the pinned AMD image on CPU with
network and devices disabled. DMS46 process inference and SigLIP2 ROCm/MIGraphX
execution remain unproven.

## Immutable source and checkpoint status

- Repository: `https://github.com/apple-aiml-research/ml-dms-dataset`
- Source commit: `a379a63e9435e32134a465eb31ecb0aefebed985`
- Pinned source entrypoints: `inference.py`, `taxonomy.json`, and `LICENSE.txt`
  at that commit.
- Official weight archive URL:
  `https://docs-assets.developer.apple.com/ml-research/datasets/dms/dms46_v1.zip`
- Publisher-advertised archive size: 170 MB (decimal size from README); the
  local exact byte count is now recorded below. Apple publishes no checksum in
  the pinned README.
- Pinned taxonomy file is staged at
  `api/runtime/adapters/material-identity/cache/taxonomy.json`; byte count
  9,157; SHA256
  `5fb9f5b6d81b800468db78532a9284f060ddc2e2a44b8214658be797029439ea`.
- Downloaded official archive is staged at
  `.modly-amd-runtime/models/dms46/dms46_v1.zip`; 173,688,250 bytes;
  SHA256 `3aa3645927148e9ed64571bcbd6997efcc07378eb14690db328b5530de8460c1`.
  Publisher README advertises 170 MB and provides no checksum.
- The archive contained exactly one safe member, `DMS46_v1.pt` (187,516,330
  uncompressed bytes, CRC32 `b4e8b923`); the ZIP CRC check passed. Only this
  exact member was extracted, with no path traversal. It is staged at
  `.modly-amd-runtime/models/dms46/DMS46_v1.pt`; SHA256
  `4261c0d88922c48116c4f5c4d5a04f6be25077d5b6840b2ff7f0aa068b5d9421`.
  Both archive and checkpoint hashes were computed locally. The checkpoint is
  not loaded or otherwise executed.
- The pinned README identifies the dataset as CC-BY-NC 4.0 and points the
  pretrained model to Apple's separate `LICENSE.txt`; original RGB inputs
  retain their own source licenses. The Apple terms grant a personal,
  non-exclusive license subject to conditions. `LICENSE.txt` also refers to
  `ACKNOWLEDGEMENTS`, but that path and the plausible `.txt`/lowercase variants
  were not present at the pinned source revision. Resolve that license record
  and Modly deployment/use review before any model inference.
- The pinned Apple `environment.yaml` describes the research environment as
  Python 3.8, PyTorch 1.9.1, torchvision 0.10.1, and CUDA toolkit 10.2. The
  sample inference also imports OpenCV, NumPy, Pillow, and TorchVision. The
  Modly adapter reimplements the documented image transform with Pillow,
  NumPy, and Torch, but running its TorchScript checkpoint under the target
  PyTorch ROCm 2.11 and MIGraphX remains unproven.

## Bounded target probe plan

The exact run protocol and outstanding gates are in
[`PROBE_PLAN.md`](PROBE_PLAN.md). It is CPU-authored and has not executed the
model. It fixes the input and memory bounds, target image, source/weight
identities, backend comparison, and required telemetry. Actual process input
must come from a real topology-bound asset with calibrated observation views;
do not invent view digests or face maps. Ticket 03/04 GPU availability and
license review are still prerequisites.

The project-authorized local transfer completed on 2026-09-24. ZIP integrity
and the sole expected member were verified before extraction; exact local
SHA-256 values are now recorded. The model has not been loaded, imported, or
probed. Source hashes in `ASSET_LOCK.json` identify GitHub blob objects and
must not be confused with SHA-256 values for the downloaded model bytes.

## Replaceable SigLIP2 region-crop route

The process extension also exposes the explicitly selected
`candidate_id=google.siglip2.base-patch16-224` route. It uses the same Modly
JSON-lines stage and AMDInferenceRuntime seam; the DMS46 route remains
independently selectable. SigLIP2 requires an exact immutable repository
revision, local `model.safetensors` digest and all config/processor/tokenizer
asset identities listed in `SIGLIP2_ASSET_LOCK.json`. Runtime assets are read
from the configured `MODLY_SIGLIP2_MODEL_ROOT` directory (mounted read-only by
the launcher) or a workspace-local model directory. It uses local-only
Transformers loading and fails closed if any expected byte count or digest
differs.

Each SigLIP2 view carries its source observation digest, segmenter ID, exact
mask digest, finite `world_to_clip` matrix and matching projection digest,
Ticket 06 view identity digest, and a face-ID map with its own computed digest.
The adapter verifies source image bytes and dimensions, then masks pixels
outside each current topology-bound material region, scores that crop against
the pinned prompt revision `builtin:siglip2-material-prompts-v2`, and records raw
logits per crop/view. Its five scored labels exactly match the frozen identity
classes: `rubber_latex`, `glass`, `clear_plastic`, `paint_plaster_enamel`, and
`metal`. Separate `metal_bare_subtype` and `metal_painted_subtype` prompts are
auxiliary evidence and do not split the required generic `Metal` class. Crop, prompt,
processor/model asset, observation, mask, projection, face-map, and view input
identities are included in provenance and the digest-addressed stage artifact.
No part labels or glTF material/PBR slots are read.

The frozen Ticket 07 rubric contains no calibration set or decision policy for
SigLIP2 similarity logits. Therefore any region with crop scores remains
`ambiguous`; a region with no visible crop evidence is `unknown`. Raw logits are
not converted to probabilities or normalized into a forced identity. This is
an inference and evidence path, not a classifier quality result or candidate
acceptance. A future calibration policy must be fixed on the calibration split
before it can promote a single label, and must pass the existing held-out gates.
The synthetic CPU preflight in `ALTERNATIVE_MODEL_SCREEN.md` proved local
loading and output shapes. The region-crop process suite passed 10/10 in the
pinned image after the prompt taxonomy correction, with `--network=none`, two
CPUs, no device mounts, and read-only source/model/wheel mounts; the real
process test loaded the model and scored masked crops. This remains synthetic
CPU integration only. Later AMD target qualification and held-out material
quality evaluation remain outstanding.

The staged repo is named SigLIP 2, but its pinned model config declares
`model_type=siglip` and `processor_class=SiglipProcessor`; `AutoModel` resolves
to `SiglipModel` in Transformers 4.50.0. The exact source files used for this
contract check are `models/siglip/modeling_siglip.py` SHA256
`2245f8ada489e7bee3218318e107949c74750a2882b2ebd7b449d6699088d109` and
`models/siglip/processing_siglip.py` SHA256
`0b3a6e5517945bed4dc7d6a5d8e7b6c100618b953b981a372cb7813c2112ef42`; the
Transformers wheel digest is in `WHEEL_LOCK.json`. `SiglipModel.forward`
accepts an optional `attention_mask=None`. The pinned processor returns exactly
`pixel_values` and `input_ids` for this fixed square crop, and may omit the
text attention mask. The adapter passes `encoded.get("attention_mask")`. Its
real synthetic process test asserts those actual tensor keys and shapes before
checking crop evidence and attached assertions.
