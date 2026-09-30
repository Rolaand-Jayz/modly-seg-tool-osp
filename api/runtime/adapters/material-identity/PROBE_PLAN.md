# DMS46 bounded RX 7900 GRE probe plan

Status: local model use approved; CPU contract preparation complete; RX 7900 GRE
probe not executed. The DMS46 evaluator and batch collector have synthetic
contract coverage. The owner approved a development-only supported-region
coverage floor of 0.87, derived from the unchanged held-out coverage and
abstention conjunction. Fixture scoring still requires an actual Modly AMD run
and a passing development screen. This plan does not change held-out quality gates in
`SELECTION_AND_GATES.md`.

## Ticket07 fixture scoring batch contract

The label-blind input manifest omits split fields. Evaluator split assignment
must be reconstructed from the pinned renderer's deterministic case/object
IDs, with exact input support checked against both split plans. Run one Modly
material-identity workflow stage per rendered object asset; do not synthesize
or pass predictions to the scorer. Collect exactly 35 development stage
artifacts or 110 held-out stage artifacts. Verify each stage artifact's exact
saved-byte digest, candidate/model/policy identity, geometry digest, planned
region ID, and topology revision. Write each combined raw batch as canonical
JSON with an adjacent fsynced SHA-256 sidecar before producing its development
screen report. Held-out evaluation must re-read and verify both raw-batch
sidecars and the durable development report before opening truth.

The development stop/go uses supported-only coverage >= 0.87, in addition to
macro-F1 >= 0.85, minimum supported-class recall >= 0.80, and unknown and
ambiguous abstention >= 0.90. This rule was approved by the owner because the
held-out all-region 0.80 floor cannot be applied to the development sample's
100:40 supported-to-OOD composition alongside the same OOD abstention floors.
The held-out conjunction and all thresholds remain unchanged; held-out
coverage is still computed over all 440 regions and requires at least 352
accepted single-label outputs.

## Immutable inputs

| Input | Identity |
|---|---|
| Upstream repository | `https://github.com/apple-aiml-research/ml-dms-dataset` |
| Upstream source revision | `a379a63e9435e32134a465eb31ecb0aefebed985` |
| Model archive | `.modly-amd-runtime/models/dms46/dms46_v1.zip`; 173,688,250 bytes; SHA256 `3aa3645927148e9ed64571bcbd6997efcc07378eb14690db328b5530de8460c1` |
| TorchScript checkpoint | `.modly-amd-runtime/models/dms46/DMS46_v1.pt`; 187,516,330 bytes; SHA256 `4261c0d88922c48116c4f5c4d5a04f6be25077d5b6840b2ff7f0aa068b5d9421` |
| Taxonomy | `api/runtime/adapters/material-identity/cache/taxonomy.json`; SHA256 `5fb9f5b6d81b800468db78532a9284f060ddc2e2a44b8214658be797029439ea` |
| Classifier source | SHA256 `90662ff2821605b70f48315fa2cc350697ac4d9289584feb54a17c9a866538c6` |
| Process extension source | SHA256 `7da3e608ff9c190f8a8c5ccc33f2b52831e099b291437769727dfb8ee50665c1` |
| Manifest | SHA256 `4009ca256e18c6d479a1161bd8760f757ef0f00070ff0e350976aa23d505bc0d` |

The Apple README advertises 170 MB and provides no published archive/model
checksum. Local hashes therefore identify the acquired bytes; they are not
publisher signatures. The outer ZIP passed CRC validation, contained exactly
one safe member, and only `DMS46_v1.pt` was extracted. The extracted archive
was listed as data only; no pickle was deserialized and no TorchScript code was
loaded. Treat it as executable model content and load only in the isolated
project adapter after the source/license and runtime controls are confirmed.

## Preconditions

1. The owner approved local evaluation of the pretrained checkpoint and
   clarified that the project does not use the DMS training dataset. The
   dataset CC-BY-NC-4.0 terms are not treated as the license for this model
   inference. Keep the weights local and do not redistribute/package them as
   part of this task. Rights to each original input image remain separate.
   The candidate-specific two-phase evaluation protocol is frozen in
   `evidence/DMS46_EVALUATION_PREREGISTRATION_2026-09-26.md`.
2. Wait until Ticket 03/04 releases the RX 7900 GRE. Do not overlap heavyweight
   model activity with other GPU owners.
3. Use a real Structured Asset with current topology and source observations.
   It must have actual calibrated camera/face correspondence produced by the
   upstream view stage. No synthetic digest or fabricated face map may be used
   for this run.
4. Verify the staged archive/checkpoint/taxonomy hashes in the target
   workspace immediately before launch; disable network access for inference.
5. Run through Modly's `reference-material-identity` JSON-lines process
   extension, not a separate model launcher. The API test venv does not have
   the target inference dependencies and is not the probe environment.

## Exact adapter configuration values

The process input's `materialInference` object must contain these values:

```json
{
  "upstream_revision": "a379a63e9435e32134a465eb31ecb0aefebed985",
  "adapter_revision": "sha256:90662ff2821605b70f48315fa2cc350697ac4d9289584feb54a17c9a866538c6",
  "weights_id": "apple.dms46.v1",
  "weights_path": ".modly-amd-runtime/models/dms46/DMS46_v1.pt",
  "weights_digest": "sha256:4261c0d88922c48116c4f5c4d5a04f6be25077d5b6840b2ff7f0aa068b5d9421",
  "taxonomy_path": "api/runtime/adapters/material-identity/cache/taxonomy.json",
  "taxonomy_digest": "sha256:5fb9f5b6d81b800468db78532a9284f060ddc2e2a44b8214658be797029439ea"
}
```

The actual `views` entries are generated from the selected asset's observation
and calibrated face maps at launch. Each view must carry the real observation
digest, width, height, and face-ID map. Do not save an empty or placeholder
`views` list as a runnable configuration.

## Bounded execution sequence

1. Record the Modly runtime report and assert target is RX 7900 GRE `gfx1100`,
   not an NVIDIA/CUDA backend. Target image is
   `localhost/modly-amd-migraphx:ticket02`: Python 3.12.3, PyTorch
   `2.11.0+rocm7.14.0`, HIP `7.14.60850`, MIGraphX
   `2.16.0.dev+20250912-17-575-g4bcfe75b2`, Torch-MIGraphX `1.2`.
2. Launch one real view first, batch size one. Official transform: RGB image,
   aspect-preserving resize so the longest dimension is scaled to 512 (including
   upscaling smaller inputs), ceil output dimensions, LANCZOS, tensor
   ImageNet mean/std scaled by 255. Confirm observed input and output shapes
   match the adapter contract. If the output shape is not `[1,1,H,W]` or
   `[1,46,H,W]`, fail closed and record it; do not adapt silently.
3. Run the eager PyTorch ROCm reference and the runtime's MIGraphX candidate
   on identical bytes. Record per-pixel output parity (`atol=1e-4`,
   `rtol=1e-3`), compile outcome, cold and warm latency, peak allocated plus
   reserved VRAM, peak host RSS, selected backend, and runtime versions. Keep
   MIGraphX only if parity passes and warm latency is no worse than eager ROCm;
   otherwise report explicit PyTorch ROCm fallback. Record any CPU fallback as
   a failed target attempt, not a GPU success.
4. Respect the implementation's pre-inference bounds: at most 64 distinct
   views, 64 MiB per source image, source dimensions no greater than 2048 per
   axis, and no more than 2,000,000 aggregate source pixels or resized pixels
   in one process run. Inference is sequential. Do not tune these caps during
   this probe.
5. Hard memory gate: isolated DMS46 stage peak allocated plus reserved VRAM
   must be at most 14 GiB of the 16 GiB card. Stop and report if exceeded.
   Report host RAM, startup and warm latency, and the actual Modly process
   result/stage artifact digest.
6. A one-view runtime probe only establishes target execution mechanics. It
   does not pass the fixed held-out quality/coverage gates. Quality acceptance
   still requires the frozen split-by-object fixture and thresholds in
   `SELECTION_AND_GATES.md`, plus any required unknown/ambiguous cohorts.

## Known integration risks

The original Apple research environment declares Python 3.8, PyTorch 1.9.1,
TorchVision 0.10.1, and CUDA Toolkit 10.2. The adapter instead relies on
PyTorch TorchScript, NumPy, Pillow, and Modly AMD Runtime in the project ROCm
image. TorchScript compatibility with PyTorch 2.11 ROCm has not been measured.
The DMS sample script checks `torch.cuda.is_available()` and optionally calls
`.cuda()`; the adapter has a distinct explicit AMD-runtime seam. The sample
also imports torchvision, OpenCV, and PIL, while the adapter reimplements the
reference preprocessing with Pillow and NumPy. The exact checkpoint license
review and actual target model load remain gates. This note makes no quality,
latency, memory, or backend claim.
