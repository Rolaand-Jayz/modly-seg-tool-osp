# 07 — Material identity classification

### MatSee and Visual2Echo official-source screen (2026-09-28)

MatSee's published README does not establish a pinned classifier checkpoint or
the fixed selective-output and AMD contracts. Visual2Echo includes a frozen
23-class MINC material branch, but its documented task and metrics are audio
depth prediction; the material checkpoint's immutable identity/terms and
Ticket 07's five-label, abstention, topology, AMD, and quality gates are not
established. Neither qualifies for acquisition or fixture evaluation. No
assets, fixture, truth, inference, or GPU were accessed, and no gate changed.
Evidence: [`TICKET07_MATSEE_VISUAL2ECHO_SOURCE_SCREEN_2026-09-28.md`](../../../api/runtime/adapters/material-identity/evidence/TICKET07_MATSEE_VISUAL2ECHO_SOURCE_SCREEN_2026-09-28.md).

### FMMC CVPR 2026 official-source refresh (2026-09-28)

FMMC is a relevant masked-region material-classification research lead, but
it is not eligible for acquisition or fixture evaluation. The official paper
says its source and dataset will be released; the inspected sources provide
no immutable FMMC inference-code/checkpoint identities or FMMC checkpoint
terms. Its described test inference also calls GPT-4V without a fixed hosted
model revision or offline path. The five frozen Modly classes, unknown and
ambiguous abstention, bare-versus-painted-metal abstention, topology-bound
Modly IDs, <=14 GiB AMD execution, and frozen Modly quality are not established.
Stop before acquisition or evaluation. No artifacts, fixtures, or truth were
accessed and no gates changed. Evidence:
[`TICKET07_FMMC_CVPR2026_PRIMARY_SOURCE_SCREEN_2026-09-28.md`](../../../api/runtime/adapters/material-identity/evidence/TICKET07_FMMC_CVPR2026_PRIMARY_SOURCE_SCREEN_2026-09-28.md).

### MatSeg zero-shot material-state source screen (2026-09-28)

MatSeg is a class-agnostic, query-point similarity segmentation method; its
published interface does not provide the five frozen material identities,
unknown/ambiguous or bare-versus-painted abstention, or a topology-bound
Modly-region contract. The inspected code is CC0, but checkpoint links are
mutable and no checkpoint-specific terms or digest are established. Its
documented inference path pins CUDA PyTorch and calls `.cuda()`, with no
ROCm/RX 7900 GRE evidence. Rejected before acquisition or evaluation. No
packages, weights, fixture, truth, inference, or GPU were accessed; no gate
changed. Evidence:
`api/runtime/adapters/material-identity/evidence/TICKET07_MATSEG_ZERO_SHOT_SOURCE_SCREEN_2026-09-28.md`
(SHA-256 `7dbabd074e59fdc3047f669d7c8ed78531d36ca1578ed5ba5fddea23ecfc547b`).

**What to build:** Attach open-vocabulary material identity assertions to material regions through a replaceable `Classify Material Identity` adapter, while keeping semantic material names independent of render-property/PBR estimates.

**Blocked by:** 01 — Structured Asset headless round-trip; 02 — AMD Runtime proof through Modly; 06 — Material-region segmentation independent of parts.

**Status:** in-progress and acceptance-blocked — replaceable process routes and the frozen object-disjoint fixture are implemented. SigLIP2, DINOv2, feature fusion, cross-view variability, SmolVLM2, Qwen3, and the owner-approved local DMS46 candidate have failed their unchanged development screens. DMS46 was evaluated through the corrected, frozen dense-map adapter using its staged checkpoint; the training dataset was not used, and heldout truth was not opened. The DMS46 result is recorded in [`TICKET07_DMS46_DEVELOPMENT_SCREEN_2026-09-26.md`](../../../api/runtime/adapters/material-identity/evidence/TICKET07_DMS46_DEVELOPMENT_SCREEN_2026-09-26.md). No candidate has passed the frozen development gates; heldout acceptance remains open.

### Additional source lead: SSUF (2026-09-28)

The new Sparse Surface Understanding Framework paper is an outdoor four-label surface classifier, not a dense named-material classifier over Modly's five-class material-region contract. The inspected source does not provide a released inference implementation/checkpoint identity, topology/region binding, required unknown/ambiguous or bare/painted-metal behavior, or RX 7900 GRE evidence. It is rejected before acquisition or evaluation; no fixture or truth data was accessed and no gate changed. Details and official links: [`TICKET07_SSUF_OFFICIAL_SOURCE_SCREEN_2026-09-28.md`](../../../api/runtime/adapters/material-identity/evidence/TICKET07_SSUF_OFFICIAL_SOURCE_SCREEN_2026-09-28.md).

- [ ] Before model acceptance, material-classification fixtures, accepted label set/normalization policy, coverage target, and correctness threshold are declared; the classifier then meets those thresholds for supported candidates such as rubber, glass, plastic, painted metal, bare metal, or equivalent open-vocabulary descriptions.
- [x] The classifier can emit unknown/ambiguous and multiple competing candidates rather than forcing a single answer. Verified by deterministic dense-label aggregation fixtures; these are not DMS accuracy evidence.
- [x] Optional normalization preserves the original label while mapping known aliases/categories to a controlled vocabulary when configured.
- [x] Material identity assertions remain separate from metallic/roughness/base-color values and can be corrected/replaced without regenerating PBR maps. Replacement tests preserve the prior PBR assertion and its provenance.
- [x] Assertions record evidence kind, confidence state/source, model/weights provenance, observations/features used, and topology-bound material-region target. The real synthetic SigLIP2 process test now populates this evidence on masked region crops; it makes no material-quality claim.
- [x] The adapter is not coupled to the selected part segmenter or geometry generator; it consumes Structured Asset material regions and observations without reading parts or glTF material/PBR slots.

## Evidence and remaining acceptance

### Molmo2 use-scope recheck (2026-09-27)

Official Ai2 sources do not establish that the released Molmo2-4B checkpoint is permitted for a general public-facing asset-creation product. Its model card says Apache-2.0 while describing research/education intent and third-party training data subject to academic and non-commercial research use; Ai2's Responsible Use Guidelines define Research Use as sole-purpose scientific research and development. The public sources do not clearly say whether the data restriction attaches to downstream checkpoint use, so this is uncertain rather than a finding of express prohibition. Modly's intended use is not evidenced as solely research/education; rights clearance remains blocked. This terms review does not establish material accuracy, abstention quality, AMD execution, or resource eligibility. Sources: [Molmo2 model card](https://huggingface.co/allenai/Molmo2-4B), [Ai2 Responsible Use Guidelines](https://allenai.org/responsible-use), [Ai2 Molmo2 announcement](https://allenai.org/blog/molmo2), and [Molmo2 code repository](https://github.com/allenai/molmo2). No model files were acquired, no fixture truth was accessed, and no criteria changed.

- RF-MatID official-source screen: [`evidence/RF_MATID_OFFICIAL_SOURCE_SCREEN_2026-09-25.md`](../../../api/runtime/adapters/material-identity/evidence/RF_MATID_OFFICIAL_SOURCE_SCREEN_2026-09-25.md). Candidate is ineligible because it consumes RF measurements rather than imagery and has no identified immutable inference checkpoint or checkpoint terms.
- DMS46 pinned acknowledgement and evaluation records: [`evidence/DMS46_PINNED_ACKNOWLEDGMENT_RECHECK_2026-09-26.md`](../../../api/runtime/adapters/material-identity/evidence/DMS46_PINNED_ACKNOWLEDGMENT_RECHECK_2026-09-26.md), [`evidence/DMS46_EVALUATION_PREREGISTRATION_2026-09-26.md`](../../../api/runtime/adapters/material-identity/evidence/DMS46_EVALUATION_PREREGISTRATION_2026-09-26.md), and [`TICKET07_DMS46_DEVELOPMENT_SCREEN_2026-09-26.md`](../../../api/runtime/adapters/material-identity/evidence/TICKET07_DMS46_DEVELOPMENT_SCREEN_2026-09-26.md). The DMS46 evaluator was implemented and frozen, then the candidate failed the development screen; heldout truth remains unopened.

- Frozen evaluation gates and candidate decision: [`SELECTION_AND_GATES.md`](../../../api/runtime/adapters/material-identity/SELECTION_AND_GATES.md).
- Runtime dependencies, process-extension entrypoint, immutable source commit, staged model/checkpoint hashes, and bounded target runbook: [`TARGET_RUNTIME.md`](../../../api/runtime/adapters/material-identity/TARGET_RUNTIME.md), [`ASSET_LOCK.json`](../../../api/runtime/adapters/material-identity/ASSET_LOCK.json), [`SIGLIP2_ASSET_LOCK.json`](../../../api/runtime/adapters/material-identity/SIGLIP2_ASSET_LOCK.json), [`PROBE_PLAN.md`](../../../api/runtime/adapters/material-identity/PROBE_PLAN.md).
- API contract command: `PYTHONPATH=api .modly-amd-runtime/api-test-venv/bin/python -m unittest discover -s api/tests -p 'test_ticket07_material_identity.py' -v` — parent most recently reported 9 passed and 1 model-root integration skip in the local environment; it is not the actual SigLIP2 process integration evidence.
- Pinned project-image process command: `XDG_RUNTIME_DIR="$PWD/.modly-amd-runtime/run" podman --root "$PWD/.modly-amd-runtime/storage" --runroot "$PWD/.modly-amd-runtime/run" run --rm --userns=host --network=none --cpus=2 --env PYTHONPATH=/project/api:/preflight:/project/.modly-amd-runtime/api-test-venv/lib/python3.12/site-packages --env PYTHONNOUSERSITE=1 --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env HF_HOME=/tmp/siglip2-hf-empty --env MODLY_SIGLIP2_MODEL_ROOT=/models --env CUDA_VISIBLE_DEVICES= --env HIP_VISIBLE_DEVICES= --env ROCR_VISIBLE_DEVICES= --volume "$PWD:/project:ro" --volume "$PWD/.modly-amd-runtime/models/material-identity/siglip2:/models:ro" --volume "$PWD/.modly-amd-runtime/package-cache/siglip2/site-packages:/preflight:ro" --workdir /project localhost/modly-amd-migraphx:ticket02 python -m unittest discover -s /project/api/tests -p 'test_ticket07_material_identity.py' -v` — parent reran the corrected pinned-image suite at 10/10 in 18.687 seconds. `/preflight` precedes the API venv in `PYTHONPATH` so the hash-locked Transformers 4.50.0 and huggingface-hub 0.28.1 overlay wins over the base image's incompatible hub 2.0.0.
- That process test uses only one generated 4×2 solid-color image and two masked crops to verify the actual model/processor stage contract. It asserts the five frozen primary class prompts, two auxiliary metal-subtype prompts, topology-bound crop evidence, raw similarity logits, ambiguous/uncalibrated state, CPU-only telemetry, prompt/crop digests, and PBR preservation. It is process integration only; it does not score the rendered holdout or establish any quality gate.
- Generated held-out evaluation fixture and source audit: [`fixtures/RENDERED_FIXTURE.md`](../../../api/runtime/adapters/material-identity/fixtures/RENDERED_FIXTURE.md), [`fixtures/render_fixture.py`](../../../api/runtime/adapters/material-identity/fixtures/render_fixture.py), and [`FIXTURE_SOURCE_AUDIT.md`](../../../api/runtime/adapters/material-identity/FIXTURE_SOURCE_AUDIT.md). The ignored bundle `.modly-amd-runtime/material-identity-fixture-v1/` has 145 object instances and 580 regions: 440 heldout (80/class ×5 classes, 20 unknown, 20 ambiguous) and 140 development (20/class ×5 classes, 20 unknown, 20 ambiguous). Unknown and ambiguous cohorts each have five object instances in both splits; all views of an object stay within one split. Selective-coverage denominator is all 440 heldout regions, including unknown/ambiguous; >=80% means >=352 accepted single-label outputs. Truth labels and recipes are separate from model inputs. The pinned manifest SHA256 is `c7c5d9c2ab31d3d956411c019e2ac19a72f991da5829ed18ceacf36e47094466`; input digest `453ac070aadd84eaf45eb381fb2910e906026b47cc929a5e6fdc4d6dd5f29694`; truth digest `8ca5699c1f3f7d96c2d28b024df6a67db0d9a6d725fbbf2e2f6f604bede6e6e8`. The focused fixture command `PYTHONPATH=api .modly-amd-runtime/api-test-venv/bin/python -m unittest discover -s api/tests -p 'test_ticket07_rendered_fixture.py' -v` passes 6/6 in 50.469 seconds, including exact-mask/face-map consistency and two full deterministic generations. Parent independently verified all 1,887 indexed artifact files, sizes, digests, and the manifest sidecar. This synthetic ellipsoid-render fixture does not establish real-world generalization; the separate CPU-only SigLIP2 full-fixture score is recorded below.
- Acceptance remains open: the CPU held-out accuracy/coverage and development-only threshold/calibration results below fail the frozen quality gates; no RX 7900 GRE MIGraphX/ROCm correctness, latency, VRAM, or target-resource evidence exists. Apple DMS46 local evaluation is owner-approved and preregistered. The dataset is not used. Target runtime and quality remain unmeasured; a DMS46-compatible evaluator must be frozen before fixture scoring or truth access. SigLIP2 is permissively licensed and has completed only CPU process integration. No gate changed.

## Frozen rendered evaluation result (CPU quality evidence only)

### Ordered four-view feature candidate (2026-10-01)

A second project-owned multi-view candidate added the ordered per-view profile
of six fixed highlight/contrast cues to the same 30-cue mean/variation
signature. A preregistered 48-setting CPU screen on the development-only
split performed worse than the unordered candidate: best macro-F1 `0.65286`,
minimum supported-class recall `0.40`, supported coverage `0.92`, unknown
abstention recall `0.40`, and ambiguous abstention recall `0.80`; zero settings
met all gates. Clear plastic and glass confusion increased, and three of five
unknown regions were named as painted surface. No held-out data/GPU was used
and no model was promoted. Evidence:
[`PROJECT_OWNED_ORDERED_MULTIVIEW_RBF_PREREGISTRATION_2026-10-01.md`](../../../api/runtime/adapters/material-identity/evidence/PROJECT_OWNED_ORDERED_MULTIVIEW_RBF_PREREGISTRATION_2026-10-01.md)
and [`PROJECT_OWNED_ORDERED_MULTIVIEW_RBF_DEV_2026-10-01.md`](../../../api/runtime/adapters/material-identity/evidence/PROJECT_OWNED_ORDERED_MULTIVIEW_RBF_DEV_2026-10-01.md).
Ticket 07 remains acceptance-blocked.

### Project-owned four-view RBF development candidate (2026-10-01)

The preregistered mean-plus-population-standard-deviation four-view RBF
candidate was evaluated only on the existing 140-row development split. It
searched a fixed 48-setting grid across four gamma values, three ridge values,
and four unknown-head scales using five object-disjoint folds. The best
candidate measured macro-F1 `0.87727`, minimum supported-class recall `0.60`
(clear plastic), supported coverage `0.96`, unknown abstention recall `0.80`,
and ambiguous abstention recall `1.00`; zero settings met every unchanged
gate. Clear plastic/glass confusion and one unknown-to-painted-surface error
remain. No weights were promoted; heldout rows/truth were not opened and no
GPU run was made. Details and fixed identities are in
[`PROJECT_OWNED_MULTIVIEW_RBF_PREREGISTRATION_2026-10-01.md`](../../../api/runtime/adapters/material-identity/evidence/PROJECT_OWNED_MULTIVIEW_RBF_PREREGISTRATION_2026-10-01.md)
and [`PROJECT_OWNED_MULTIVIEW_RBF_DEV_2026-10-01.md`](../../../api/runtime/adapters/material-identity/evidence/PROJECT_OWNED_MULTIVIEW_RBF_DEV_2026-10-01.md).
Ticket 07 acceptance remains blocked.

The evaluator is implemented at `api/runtime/adapters/material-identity/evaluator.py` and is separate from the process classifier. It loads the classifier's prompt/model literals from source, verifies the staged SigLIP2 assets, checks all 1,887 fixture-manifest file sizes and SHA-256 values, reconstructs topology-derived region crops with the classifier's neutral-gray masked pixels, and scores all 580 crops in one CPU process with one processor/model load. Raw truth-free logits are persisted before `truth.json` is opened. Thresholds are selected from development rows only; a policy bound to the fixture, truth/input/manifest digests, model/weight lock, prompt digest, and evaluator source digest is persisted before held-out metrics are computed. Calibration enumerated 19,881 threshold pairs and found zero pairs meeting every frozen development gate.

The frozen run used the local `localhost/modly-amd-migraphx:ticket02` image with `--network=none`, `--cpus=2`, `--memory=8g`, no device mounts, offline Hugging Face flags, and read-only source/fixture/model/package mounts; only the result mount was writable. Exact command:

```sh
XDG_RUNTIME_DIR="$PWD/.modly-amd-runtime/run" podman --root "$PWD/.modly-amd-runtime/storage" --runroot "$PWD/.modly-amd-runtime/run" run --network=none --rm --userns=host --cpus=2 --memory=8g --volume "$PWD:/project:ro" --volume "$PWD/.modly-amd-runtime/material-identity-fixture-v1:/fixture:ro" --volume "$PWD/.modly-amd-runtime/models/material-identity/siglip2:/models:ro" --volume "$PWD/.modly-amd-runtime/package-cache/siglip2/site-packages:/preflight:ro" --volume "$PWD/.modly-amd-runtime/ticket07-evaluation:/output:rw" --env PYTHONPATH=/preflight:/project/api --env HIP_VISIBLE_DEVICES=-1 --env ROCR_VISIBLE_DEVICES=-1 --env CUDA_VISIBLE_DEVICES=-1 --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --entrypoint python localhost/modly-amd-migraphx:ticket02 /project/api/runtime/adapters/material-identity/evaluator.py --fixture-dir /fixture --model-dir /models --output-dir /output --batch-size 4
```

Exact outcome: 580 crops; CPU only (`torch 2.11.0+rocm7.14.0`, `hip_version 7.14.60850`, `cuda_available=false`); processor/model each loaded once; model-load 119.95 ms; inference 350,695 ms; peak host RSS 2,037,297,152 bytes. Heldout has 440 regions (80 per supported class and 20 in each OOD cohort). Macro-F1 is 0.224914 (gate >=0.85); minimum per-class recall is 0.0 (gate >=0.80); all-region coverage is 314/440 = 0.713636 (gate >=0.80, requiring 352); unknown abstention recall is 0.10 (gate >=0.90); ambiguous abstention recall is 0.45 (gate >=0.90). The coverage/OOD conjunction therefore fails, as do all other classifier-quality gates except fixture support. Clear plastic, metal, and rubber/latex each have zero heldout true positives. These are CPU synthetic-fixture quality measurements only, not AMD acceptance, real-world quality, or a reason to change thresholds/gates.

Frozen evidence files are in `api/runtime/adapters/material-identity/evidence/siglip2-rendered-cpu-v1/`: raw logits SHA-256 `54776b8efc0498f44c3e2145de6d120773a5248d796989286ef9b0007a1df85e`; development-only policy SHA-256 `1f98fb9d46a440b01ab3479ca3c4bda92827fc624858cf6059829ba247b0cd0c`; heldout report SHA-256 `f86e20a6c6512e024d7ecefc80e200a9116576e05e7324866587fd7238744474`. Deterministic evaluator unit tests pass 4/4 with `PYTHONPATH=api .modly-amd-runtime/api-test-venv/bin/python -m unittest api.tests.test_ticket07_evaluator -v`. No GPU run was made. Ticket 07 remains in progress and acceptance blocked; do not accept SigLIP2 based on this run.

## Development-only prompt study result

To check whether the pinned SigLIP2 candidate could meet the fixed gates without weakening them, `api/runtime/adapters/material-identity/prompt_study.py` scored only the 140 development crops. It tested frozen v2 and three newly pinned close-up prompt formulations (material, surface, texture), plus arithmetic-mean and elementwise-max aggregation across the three new forms. The fixture truth bytes were verified against the pinned digest; only development split records were joined to development crops. No heldout logits were opened or scored, no heldout truth labels were consulted for selection, and the prior heldout report was not opened or modified.

All six aggregation candidates independently searched 19,881 development threshold pairs and had zero feasible pairs. Macro-F1 ranged from 0.2265 to 0.3084; minimum per-class recall was 0.0 for every candidate. The best macro-F1 was 0.3084 (three-prompt mean), with only 0.75 all-region development coverage, 0.15 unknown abstention, and 0.45 ambiguous abstention. The surface prompt reached 0.90 coverage but had macro-F1 0.3020, minimum recall 0.0, unknown abstention 0.0, and ambiguous abstention 0.05. Thus no prompt/aggregation policy can be frozen for heldout evaluation under the development gates. Following the frozen selection rule, the candidate is rejected at the development boundary and the study stops without heldout scoring.

The study ran in the pinned target image with networking disabled, no device mounts, CPU only, two CPUs and 8 GiB memory; source/fixture/model/package mounts were read-only, with only the owned study output writable. PyTorch `2.11.0+rocm7.14.0`; CUDA unavailable; model and processor each loaded once; inference 206,339 ms. Evidence is in `api/runtime/adapters/material-identity/evidence/siglip2-development-prompt-study-v1/`: development report SHA-256 `fd029802eef1d82607ff19dc38a027daeb1f639731b8e89eb6a2c080f42122d6`, development-only raw logits SHA-256 `0b4d11cfd32bd81e469ce116ed87f7bf74b368a5b7e245ba07ffc4cc882ebcaa`, exact prompt/aggregation definitions SHA-256 `a800c573ab31f6e10c2419ffd571b7c139d436c6d104480be4cb6d7e28b8adb8`. Deterministic study/evaluator tests pass 7/7. This is development-only CPU quality evidence and does not change any acceptance gate or constitute AMD acceptance.


## DMS46 license follow-up (2026-09-25)

This dated note records the initial DMS46 source review and is superseded by the 2026-09-26 exact-pinned-source recheck and owner scope clarification. That recheck found `ACKNOWLEDGMENTS.txt` at the pinned source revision. No checkpoint was deserialized and no held-out data was accessed during those reviews. The owner has approved local pretrained-model evaluation and clarified that this project does not use the DMS training dataset. See `api/runtime/adapters/material-identity/evidence/DMS46_PINNED_ACKNOWLEDGMENT_RECHECK_2026-09-26.md`, the evaluation preregistration `api/runtime/adapters/material-identity/evidence/DMS46_EVALUATION_PREREGISTRATION_2026-09-26.md`, and the original review `api/runtime/adapters/material-identity/evidence/DMS46_LICENSE_REVIEW_2026-09-25.md`. The current remaining blockers are technical runtime and quality evidence plus a DMS46-compatible evaluator.

## DINOv2 alternate candidate result (2026-09-25)

The official Meta standard DINOv2 ViT-B/14 checkpoint is an Apache-2.0 candidate, distinct from the unused Hugging Face safetensors snapshot whose pinned card says CC-BY-NC-4.0. The CPU evaluator is isolated in `api/runtime/adapters/material-identity/dinov2_cpu.py` and `dinov2_evaluator.py`; it verifies the official source/checkpoint identities, uses `weights_only=True`, embeds all 580 topology-bound views without truth, saves them durably, then applies five-fold object-disjoint OOF cosine prototypes to the 140 development views. Frozen development gates failed: macro-F1 `0.6652343` (<0.85), minimum class recall `0.25` (<0.80), all-region coverage `0.8357143` (>=0.80), unknown abstention recall `0.0` (<0.90), ambiguous abstention recall `0.5` (<0.90), and `0/19,881` threshold pairs feasible. It stopped before opening `truth.json`; no heldout logits, metrics, policy, or result were generated. No quality gates changed. CPU-only results do not establish RX 7900 GRE backend, latency, parity, or VRAM acceptance. Exact command and full license/runtime provenance: `api/runtime/adapters/material-identity/evidence/DINOV2_LICENSE_PROVENANCE_REVIEW_2026-09-25.md`.

Durable CPU evidence is under ignored `.modly-amd-runtime/ticket07-dinov2-evaluation/`: `dinov2-embeddings.json` (580 truth-free embeddings, SHA-256 `97039cbb28301ffb3d5c9582d4fc380c98572be35e70c56d00732781c5f7a086`), `dinov2-development-oof-logits.json` (140 OOF rows, SHA-256 `f89e268ab1ff71778dddfad54eff0a84d9d8d9de0199e274db695c78c38e9d1c`), and `dinov2-development-evaluation.json` (development gate result, SHA-256 `744fd6bb479d471011043deb18b900fb21ee19b99898b27deecf1e73c87bb3b0`). The embedding pass took `209,467.74 ms` on CPU with batch size four. Focused evaluator/source-bound tests pass 9/9; this is development-only model screening evidence, not Ticket07 acceptance.

A single fixed lightweight dual-ridge classifier head was separately screened on those durable features, using `lambda=1.0`, one object-mean training vector, object-disjoint folds, and no hyperparameter sweep. Its development gates also failed: macro-F1 `0.6668177`, minimum recall `0.20`, all-region coverage `0.7214286`, unknown abstention `0.20`, ambiguous abstention `0.80`, and `0/19,881` feasible threshold pairs. It stopped without opening heldout truth. OOF logits SHA-256 `afef82d434d39a1759aefbbdda14af08ae16b20ec194a615188056762a78ac57`; report SHA-256 `290bbe720e947120e8926912a3a27ce3e393a161cbc25ce01f6658fda64baac5`. Details and the exact command are in `api/runtime/adapters/material-identity/evidence/DINOV2_RIDGE_DEV_SCREEN_2026-09-25.md`. This candidate does not independently validate beyond the frozen OOF development process and does not change the gates.


## Additional candidate screens (2026-09-25)

The official Cornell MINC model archive remains uncleared: its page licenses annotations but not the 714 MB trained weights, and publishes no immutable archive identity/hash. Its 23-class patch/dense material task and documented categories do not establish the frozen five-class/OOD gates. Evidence: `api/runtime/adapters/material-identity/evidence/MINC_MODEL_SCREEN_2026-09-25.md`.

RMSNet/KITTI-Materials is also source-only: repository code is MIT, dataset is CC BY 4.0, but RMSNet and MiT-B2 checkpoint terms are unspecified; Google Drive checkpoint files have no pinned revision/hash. Its 20-category dense road-scene model has no demonstrated frozen-label/OOD fit or AMD execution. A topology projection is only possible with actual calibrated source views and maps; no fixture crop/3D route is demonstrated. Evidence: `api/runtime/adapters/material-identity/evidence/RMSNET_MODEL_SCREEN_2026-09-25.md`. No additional deployment candidate is cleared. Ticket07 remains acceptance blocked; all gates are unchanged.

## SigLIP2 image embedding plus fixed ridge screen (2026-09-25)

The already-cleared pinned SigLIP2 image tower was screened separately from the failed prompt classifier. Transformers 4.50.0 `SiglipModel.get_image_features(pixel_values=...)` produced L2-normalized 768-dimensional projected pooled image vectors for all 580 fixture crops. The complete vector file was durably written before development labels were derived. One fixed dual-ridge head (`lambda=1.0`) was evaluated with five-fold object-disjoint OOF on 140 development crops; no parameter sweep or prompt input was used. The frozen gates failed: macro-F1 `0.6797408`, minimum recall `0.45`, all-region coverage `0.7285714`, unknown abstention `0.40`, ambiguous abstention `0.40`, and `0/19,881` feasible threshold pairs (`102/140` accepted). The evaluator stopped without opening fixture truth; no heldout result or policy was produced. All thresholds remain unchanged. Exact command, environment, and output hashes: `api/runtime/adapters/material-identity/evidence/SIGLIP2_IMAGE_EMBEDDING_RIDGE_DEV_SCREEN_2026-09-25.md`. Focused embedding/ridge regression tests pass 2/2. Ticket07 remains acceptance blocked; this is not AMD target qualification.

## Current open vision-language model screen (2026-09-25)

Qwen3-VL-2B-Instruct is a plausible source-screened lead only, not cleared for acquisition/evaluation/deployment. Official Hugging Face metadata marks Apache-2.0 and reports a 4.26 GB checkpoint with SHA-256 `7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0`, but only abbreviated commit `78448d7` and no complete immutable model/tokenizer/processor manifest were established. Qwen3-VL requires Transformers >=4.57; AMD documents Radeon support for Qwen3-VL-8B on a different Radeon platform, not this checkpoint on the RX 7900 GRE. No measured target memory/runtime evidence exists. Its generative interface does not provide calibrated material logits or native unknown/ambiguous states; a fixed answer grammar plus teacher-forced token scores could be considered only after pinning and validating that scorer, then run through the existing dev-only gates. No download/execution occurred and no fixture truth was accessed. Evidence: `api/runtime/adapters/material-identity/evidence/QWEN3_VL_SMALL_OPEN_MODEL_SCREEN_2026-09-25.md`. No new deployment candidate is cleared; Ticket07 remains blocked and all gates are unchanged.

### Qwen3-VL pinned snapshot reassessment

The official Hub now exposes the complete snapshot revision `89644892e4d85e24eaac8bacfd4f463576704203`, a 12-file model/tokenizer/processor manifest, Apache-2.0 metadata, and the pinned principal checkpoint SHA-256 `7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0`. This supersedes the earlier missing-snapshot blocker. Transformers' official Qwen3-VL `forward` interface exposes image-conditioned causal token logits, which can support a predeclared constrained-answer score interface, but no source promises deterministic per-label scores. Qwen is conditionally eligible for a synthetic-only CPU scorer preflight. Require fixed answer-code tokenization and parser, pinned model/runtime, repeated single-item `eval()` score checks, finite complete outputs, and deterministic tolerance before accessing development labels. If that preflight passes, proceed to the unchanged development-only Ticket07 procedure; stop before heldout on any failed gate. No assets were downloaded/executed, no fixture or heldout data accessed, no AMD target gate passed, and no acceptance threshold changed. Full evidence and source links: `api/runtime/adapters/material-identity/evidence/QWEN3_VL_PINNED_SNAPSHOT_REASSESSMENT_2026-09-25.md`.

A further source-only audit confirms ROCm Linux lists RX 7900 GRE (`gfx1100`) support, establishing a plausible generic PyTorch ROCm route. This does not establish model-specific parity, memory, latency, or quality acceptance. The immutable SigLIP2 and DINOv2 candidates both failed the frozen development quality gates; Qwen3-VL remains incompletely snapshot-pinned. No current candidate clears both the unchanged quality gates and target execution evidence. Details: `api/runtime/adapters/material-identity/evidence/TICKET07_IMMUTABLE_MODEL_AND_RX7900GRE_ROUTE_SCREEN_2026-09-25.md`.

### Fixed physics-appearance feature candidate (2026-09-25)

A truth-isolated classifier using one fixed 30-dimensional image-appearance descriptor and a fixed dual-ridge head was rejected at development gates. It featurized all 580 crops before deriving development labels, then used five object-disjoint folds over the 140 development views; the held-out truth was not opened. Metrics: macro-F1 0.5896883, minimum class recall 0.30, all-region coverage 0.5142857, unknown abstention 0.60, ambiguous abstention 0.90, and 0/19,881 feasible threshold pairs. Focused tests passed 3/3 under system Python 3.14.7 because the project API venv lacks Pillow; these are candidate-development checks, not product/runtime acceptance. Candidate and test sources plus durable hashes/limitations are in `api/runtime/adapters/material-identity/physics_feature_candidate.py`, `api/tests/test_ticket07_physics_feature_candidate.py`, and `api/runtime/adapters/material-identity/evidence/PHYSICS_APPEARANCE_RIDGE_DEV_SCREEN_2026-09-25.md`. No gate changed and no heldout evaluation occurred.

### Topology-masked physics appearance candidate v2 (2026-09-25)

A v2 fixed-descriptor candidate excludes pixels outside each topology-derived region mask. It featurized and durably stored all 580 inputs before applying only the 140 development labels; five object-disjoint folds and fixed dual ridge `lambda=1.0` were used. Development gates still failed: macro-F1 0.6261040, minimum recall 0.40, all-region coverage 0.6214286, unknown abstention 0.45, ambiguous abstention 0.45, and 0/19,881 feasible threshold pairs. It stopped without opening heldout truth. Two focused tests passed in system Python 3.14.7/Pillow 12.3.0; the project API venv lacks optional Pillow, so its test skips there. Evidence, hashes, and exact command: `api/runtime/adapters/material-identity/evidence/PHYSICS_REGION_RIDGE_DEV_SCREEN_2026-09-25.md`. No threshold or acceptance gate changed.
### FMMC official model gate screen (2026-09-25)

FMMC uses task-adjacent DINOv2 patch features/mask pooling and a CLIP-derived
language prior, but its public artifacts do not identify immutable model
weights, checksums, or weight-use terms. The public sources also do not
establish the frozen five-class taxonomy or calibrated unknown/ambiguous
outputs. No assets were downloaded or run. It is not eligible for staging or
scoring. Evidence: `api/runtime/adapters/material-identity/evidence/FMMC_OFFICIAL_CANDIDATE_SCREEN_2026-09-25.md`.

### MatSim one-shot descriptor screen (2026-09-25)

MatSim can consume an RGB region plus ROI mask and return a 512-D descriptor,
so prototype voting could in principle add development-calibrated unknown and
ambiguous outcomes. It is not eligible: official sources do not pin its code
revision, checkpoint rights are unclear (the author-linked Zenodo archive has
only MD5), five-class coverage including clear plastic and rubber is
unproven, and the documented route is CUDA 11.3 with no AMD qualification.
No artifacts were downloaded and no fixture truth was opened. See the
official-source findings in `api/runtime/adapters/material-identity/evidence/MATSIM_OFFICIAL_CANDIDATE_SCREEN_2026-09-25.md`.

### MaRI retrieval model screen (2026-09-25)

MaRI retrieves a masked region against a material embedding library, so a
future adapter could construct `unknown`/`ambiguous` states with development
calibration. It is not eligible: weights are linked through Google Drive with
no immutable revision/hash or distinct rights, `main` is mutable, published
sources do not establish clear-plastic and rubber coverage or open-set
calibration, and no AMD route is qualified. No download or fixture access.
Evidence: `api/runtime/adapters/material-identity/evidence/MARI_OFFICIAL_CANDIDATE_SCREEN_2026-09-25.md`.

## Additional material-identity frontier screen (2026-09-25)

### New candidate frontier (2026-09-25)

A disjoint source-only screen considered Fine-Grained Spatially Varying Material Selection, ObjectFolder Material Classification, and Materialistic. The first and third perform exemplar-conditioned pixel/material selection rather than named identity classification; ObjectFolder is object-level and its taxonomy does not match the frozen region labels. None documents the required calibrated unknown/ambiguous behavior, immutable rights-cleared weights, or RX 7900 GRE acceptance. No code, weights, package, fixture, or truth data was accessed. Ticket 07 remains open and all thresholds are unchanged. Reopening criteria and official links are in `api/runtime/adapters/material-identity/evidence/ticket07-new-candidate-frontier-2026-09-25.md`.

### Four-view pooling development screen (2026-09-25)

A fixed four-view mean over existing label-free region features with dual ridge (`lambda=1`) and five object-disjoint development folds increased macro-F1 to 0.8596, but minimum recall was 0.60, coverage 0.7429, unknown abstention 0.60, ambiguous abstention 0.60, and no threshold pair met every development gate (0/1,296). The candidate stopped before heldout scoring or AMD probing. No sweep or gate change occurred. The input feature digest, method, exact metrics, and evidence limitations are recorded in `api/runtime/adapters/material-identity/evidence/ticket07-multiview-pooling-dev-screen-2026-09-25.md`.

A source-only screen of MaterialSeg3D and MatSeg found no eligible Ticket 07 classifier. MaterialSeg3D is a category-limited multi-view material-region workflow with CUDA/NVIDIA dependencies and Google Drive weights lacking immutable hashes and checkpoint-specific terms; its documented output does not establish the required identity taxonomy or calibrated unknown/ambiguous classification. MatSeg addresses material-state segmentation rather than material identity and lacks checkpoint SHA-256, rights, and the frozen classifier contract. No assets were downloaded, no code was executed, and no fixture/heldout data or gates were changed. See `api/runtime/adapters/material-identity/evidence/MATERIAL_IDENTITY_FRONTIER_SCREEN_2026-09-25.md`. Existing SigLIP2 and DINOv2 development failures remain; the DMS46 missing-acknowledgement claim is corrected by `DMS46_PINNED_ACKNOWLEDGMENT_RECHECK_2026-09-26.md`, while intended-use review remains open. Ticket 07 is still open.

### SiPhy official candidate screen (2026-09-25)

SiPhy is an object-level material-reasoning research lead, not a qualified region classifier. Its official workflow downloads an unpinned Hugging Face checkpoint and base models, requires an API key for its main material-proposal path, and does not establish full checkpoint/use terms, a topology-bound region-crop contract with the frozen class/abstention behavior, or AMD inference evidence. No assets or fixture data were accessed. Full primary-source findings: `api/runtime/adapters/material-identity/evidence/SIPHY_OFFICIAL_CANDIDATE_SCREEN_2026-09-25.md`. Ticket 07 remains open; gates unchanged.
A subsequent acquisition fixed the earlier missing-snapshot identity blocker: Qwen3-VL-2B-Instruct is pinned at full revision `89644892e4d85e24eaac8bacfd4f463576704203`, and all 12 staged files match the project lock. The project-owned Transformers 4.57.1 / tokenizer overlay is hash-locked separately for CPython 3.14. A full synthetic CPU model preflight, rerun with `PYTHONNOUSERSITE=1`, passed: a pinned neutral 224×224 image produced five finite raw next-token logits that were exactly repeatable. The A–E answer token IDs are 32–36 and map in the evaluator's supported class order; prompt SHA-256 is `96869f69dd8db709fb72da2ff79d4bba381ba44f2df15fd1161b43e98fb87835`; result JSON SHA-256 is `acfc57427de698f7ee0c33ea4c74ac695da0efb4d5d8ca4f9711abd42f779d9b`. No fixture/development/heldout data was read. The system torchvision import needed a process-local schema-only shim for a missing `torchvision::nms` declaration; this is excluded from product inference and cannot count as AMD target evidence. See `api/runtime/adapters/material-identity/evidence/QWEN3_VL_SYNTHETIC_MODEL_CPU_PREFLIGHT_2026-09-25.md`. The candidate may proceed to the unchanged development-only quality screen. Earlier processor-only exploration with the user site enabled is explicitly excluded from acceptance evidence. All Ticket 07 quality, abstention, coverage, hardware, and deployment gates remain unchanged.

Qwen3-VL was then scored on all 580 frozen topology-bound views using one batch-1 CPU forward per crop. The complete label-free logits and manifest were durably hashed before deriving the 140 development targets; the module has no held-out truth reader. Frozen dev calibration found no feasible threshold pair and failed every unchanged gate: macro-F1 0.2982 (>=0.85), minimum recall 0.05 (>=0.80), all-region coverage 0.7357 (>=0.80), unknown abstention 0.10 (>=0.90), ambiguous abstention 0.50 (>=0.90). It stopped before held-out truth. Raw logits SHA-256 `13a7dd98a9155e201b7c2fc66b9e26dee3d9540d2553682f97af78f187fe50b2`; manifest `85e7d12eef0b448e11ded4963289d7274c1797ee27f51a0cdd61e1358258befb`; development report `ab45f108003f847b4cdab05507675c2b96c25b64730cee930b8c51ee9e2731ac`. CPU scoring took 918,461 ms (15m18s) for 580 crops at 0.6315 crops/s. Detailed record: `api/runtime/adapters/material-identity/evidence/QWEN3_VL_DEVELOPMENT_EVALUATION_2026-09-25.md`. Qwen is rejected for this frozen candidate contract; no policy promotion or held-out evaluation followed. Three protocol tests passed and its runner compiles. Ticket 07 remains open.

A further official-source screen found one more apparent material-specific model, HRIM2021 Material-Based Semantic Segmentation. Its authors state pretrained checkpoints cannot be shared due to confidentiality; the published taxonomy also omits required material classes and does not provide calibrated unknown/ambiguous behavior. It is ineligible before artifact access. Report: `api/runtime/adapters/material-identity/evidence/HRIM2021_MATERIAL_SEGMENTER_SCREEN_2026-09-25.md`. No weights or fixture data were accessed.

A separate fixed object-disjoint ridge head was evaluated over the persisted Qwen raw logits: normalized five-value features, one mean vector per supported training object, five object folds, fixed lambda 1.0, and no search. It improved dev macro-F1 to 0.4379 but failed unchanged gates: minimum recall 0.05, all-region coverage 0.9929, unknown abstention 0.00, ambiguous abstention 0.05; all 20 unknown and 19/20 ambiguous crops were forced to labels. This is a second candidate screen with elevated selection-bias risk because it was pursued after the direct model failed. It stopped before held-out truth. OOF/report SHA-256 `57d5fea522282bd63ea838f2ad077ee02f08c647d6226a0ebd610424c4ea9bbe` / `b6d2b5d962fd0ba4af5f13433d618c32bffca7a518bf1e79e72c49366b31a0f3`. Evidence: `api/runtime/adapters/material-identity/evidence/QWEN3_FIXED_RIDGE_DEVELOPMENT_2026-09-25.md`. No gate changed.

### Additional supervised MINC/SigLIP2 source screen (2026-09-25)

The new `prithivMLmods/Minc-Materials-23` snapshot is a supervised 23-class SigLIP2 material classifier, pinned at `4b7c27f906aea2e751e74d1ee879fc870e709d07`, with publisher model-weight SHA-256 `9ed120a4c80210413d39e3cb38a4cbe12af017394691be9bee4df9b5e73381b5`. Although its model card declares Apache-2.0 and its MINC-2500 dataset card declares CC-BY-4.0, the upstream Cornell MINC page notes that source photographs keep their own licenses, so dataset-image provenance remains to be reviewed. More directly, its declared taxonomy lacks the required rubber/latex class, its single-label head has no native unknown/ambiguous output, and it does not distinguish painted from bare metal. No weights or fixture data were accessed; no AMD behavior is claimed. It is not eligible to advance against the unchanged Ticket 07 gates. Full source screen: `api/runtime/adapters/material-identity/evidence/MINC_SIGLIP2_MODEL_SCREEN_2026-09-25.md`.

### SmolVLM2 candidate source screen (2026-09-25)

The official-source screen identified `HuggingFaceTB/SmolVLM2-2.2B-Instruct` at immutable revision `482adb537c021c86670beed01cd58990d01e72e4` as a possible development-only CPU probe. The model card/tree identify Apache-2.0 terms, a Transformers image-text path without model-specific remote Python, and two upstream-hashed safetensors shards. This does not establish Ticket 07 material-class performance, open-set calibration, or AMD acceptance. A complete local all-file asset lock and synthetic-only runtime preflight are required before any fixture evaluation. No fixture or truth data has been accessed. Source report: `api/runtime/adapters/material-identity/evidence/SMOLVLM2_OFFICIAL_CANDIDATE_SCREEN_2026-09-25.md`.

The Qwen synthetic preflight verifier was hardened to enforce the frozen asset-lock digest and verify all 12 local snapshot file hashes. `verify_snapshot()` passes on the staged revision and the full synthetic CPU preflight repeats the same finite, exactly equal five-score vector (8 PyTorch CPU threads, generated neutral-gray input, Transformers 4.57.1, `PYTHONNOUSERSITE=1`). The nine Qwen scoring contract tests pass. This additional integrity evidence does not change Qwen's rejection: its unchanged development gates failed, and no held-out truth was read.

The SmolVLM2 snapshot is now staged and locked locally at immutable revision `482adb537c021c86670beed01cd58990d01e72e4`. All 16 files (8,992,155,310 bytes) pass local size/SHA-256 checks, including both publisher LFS shard digests. With `trust_remote_code=False`, the pinned Transformers 4.57.1 processor/model loaded from the project overlay, and a generated neutral 32x32 image plus generic prompt produced finite, bitwise-identical full-vocabulary FP32 logits in two CPU forwards (79.80 s and 71.35 s). This verifies synthetic local loading and determinism only; the generic preflight does not exercise the material-label scoring contract. Evidence: `api/runtime/adapters/material-identity/SMOLVLM2_ASSET_LOCK.json` and `api/runtime/adapters/material-identity/evidence/SMOLVLM2_CPU_PREFLIGHT_2026-09-25.md`. No material fixture, dev targets, or heldout data were read. A bounded AMD synthetic preflight is the next step; all Ticket 07 quality and acceptance gates remain open.

The same snapshot then passed a synthetic RX 7900 GRE run through the pinned project container with `torch 2.11.0+rocm7.14.0` / HIP `7.14.60850`. FP16 PyTorch ROCm model load/transfer took 29.13 s; two generated-image forwards took 11.61 s and 0.91 s, producing finite bitwise-equal full-vocabulary last-token logits. PyTorch ROCm peak allocated/reserved memory was 5,982,042,624 / 6,306,136,064 bytes. This is a runtime/resource preflight only. The subsequent fixed label scorer and synthetic prompt passed focused tests, but its RX 7900 GRE development screen failed every quality gate; see `api/runtime/adapters/material-identity/evidence/SMOLVLM2_DEVELOPMENT_EVALUATION_2026-09-25.md`. Ticket 07 remains blocked on quality and unchanged acceptance gates.

### Truth-path integrity correction (2026-09-25)

Review found that the historical shared `load_fixture_inputs()` integrity loop hashed the indexed `truth.json` as opaque bytes before inference. It did not parse that JSON or use heldout labels; Qwen development labels were derived only after its complete raw score artifact was durably saved. Earlier wording that said truth bytes were never opened/read is therefore too broad and is superseded by this correction. The loader now validates the pinned truth digest and byte-count metadata from `fixture-manifest.json` and skips the truth path before filesystem stat/hash/open. `api.tests.test_ticket07_evaluator_truth_boundary` proves input loading succeeds with no truth file present. Do not describe prior generic-loader runs as having had no truth-byte access; do retain the distinct claim that no heldout truth rows were parsed or used for candidate selection. The historical hash-only integrity read did not change candidate scores or gates.


### SmolVLM2 frozen development evaluation (2026-09-25)

SmolVLM2-2.2B-Instruct at pinned revision `482adb537c021c86670beed01cd58990d01e72e4` scored all 580 frozen topology-bound views once each on the RX 7900 GRE using PyTorch ROCm FP16. The 580-row raw logits and manifest were fsync-committed before deriving the 140 development recipe labels and running the unchanged evaluator calibrator and `_dev_gate_passes()`. It failed with 0/6,014 feasible threshold pairs: macro-F1 0.1585382, minimum class recall 0.00, coverage 0.75, unknown abstention recall 0.00, ambiguous abstention recall 0.40. Selected development thresholds were 8.078125000000002 and 0.03906250000000001. The runner stopped before any heldout operation; no heldout truth was read, no policy/heldout report was created, and no gates changed.

The run used the shared corrected `load_fixture_inputs()` and a read-only staged fixture containing 1,886 non-truth assets with `truth.json` absent; the raw file has no truth labels or split fields. Runtime: PyTorch `2.11.0+rocm7.14.0`, HIP `7.14.60850`, Python 3.12.3, Transformers 4.57.1; scoring 677,474.90 ms (0.8561 views/s); PyTorch allocator peak allocated/reserved memory 5,982,346,752/6,809,452,544 bytes. The project container ran with networking disabled; model/source/runtime/fixture mounts were read-only. Focused candidate/dev-protocol tests pass 9/9. Exact command, four artifact hashes, metrics, and boundaries: `api/runtime/adapters/material-identity/evidence/SMOLVLM2_DEVELOPMENT_EVALUATION_2026-09-25.md`. This candidate fails development quality; do not proceed to heldout.

### BFMS Mask2Former source-only screen (2026-09-25)

`jinfengxie/BFMS_1014` is a 42-class dense Mask2Former taxonomy match, with a public MIT model declaration and immutable HF revision `32cd86eb4837b870a9a94bd408084da65ab4ac00`; its model file's publisher SHA-256 is recorded in the source report. It is not eligible to advance: the paper/model sources do not establish permissive training-image provenance, published BFMS per-class accuracy is only 64.19% for clear plastic and 32.93% for rubber/latex, its façade domain differs from the frozen asset crops, and no calibrated abstention or RX 7900 GRE evidence exists. No artifacts, code, fixture, development data, or truth were accessed. No acceptance gate changed. See `api/runtime/adapters/material-identity/evidence/BFMS_MASK2FORMER_OFFICIAL_SOURCE_SCREEN_2026-09-25.md`.

Screen detail: the HF checkpoint snapshot pins model config/tensors, but the linked demo is mutable and this screen did not freeze/hash the Transformers implementation that would execute it. Treat the immutable executable-code requirement as an additional blocker; see the evidence report.

### DINOv2 fixed RBF kernel-ridge development screen (2026-09-25)

A distinct fixed RBF kernel-ridge head was cross-fit over normalized four-view means of supported development objects, with the median-distance scale recomputed from training objects inside each fold. All 35 development objects remained whole across the five frozen folds; only the 140 development feature rows entered fitting, OOF scoring, and calibration. It improved over the prior fixed linear ridge screen but failed all gates except near-threshold supported coverage: macro-F1 0.7418491, minimum recall 0.50, coverage 0.7928571, unknown abstention 0.15, ambiguous abstention 0.80, and 0/19,881 feasible threshold pairs. The runner stopped with heldout truth unopened; no policy or AMD run followed. Report and exact artifact/model digests: `api/runtime/adapters/material-identity/evidence/DINOV2_RBF_KERNEL_RIDGE_DEV_SCREEN_2026-09-25.md`. No acceptance gate changed.

### DINOv2 + SigLIP2 normalized feature fusion (2026-09-25)

Two predeclared rules combined the immutable, truth-free DINOv2 and SigLIP2 image-feature artifacts: equal L2-normalized concatenation with fixed linear ridge, and the same concatenation with fixed RBF kernel ridge. Both used five whole-object development folds and only the 140 development rows for training, OOF scoring, and calibration. Neither met the frozen gates. Linear: macro-F1 0.6775857, minimum recall 0.30, coverage 0.7785714, unknown abstention 0.35, ambiguous abstention 0.55, 0/19,881 feasible threshold pairs. RBF: macro-F1 0.7470846, minimum recall 0.40, coverage 0.7285714, unknown abstention 0.20, ambiguous abstention 0.85, 0/19,881 feasible pairs. Both stopped before heldout truth; no policy was locked and no AMD run followed. Full source/model/feature/output hashes and command: `api/runtime/adapters/material-identity/evidence/DINOV2_SIGLIP2_FUSION_DEV_SCREEN_2026-09-25.md`. No gate changed.

### MateViT/MateRobot source-only screen (2026-09-25)

A fresh source screen found no Ticket 07 candidate that clears the frozen pre-evaluation gates. MateViT is a dense material segmentation lead, but its available weights are Google Drive binaries without immutable file digests or explicit weight terms; its setup/demo are CUDA-bound; and sources provide no calibrated unknown/ambiguous or bare-vs-painted metal subtype abstention. No model files or code were downloaded/run, and no fixture or truth data were accessed. Details and primary-source links: `api/runtime/adapters/material-identity/evidence/MATEROBOT_OFFICIAL_SOURCE_SCREEN_2026-09-25.md`.

### Hierarchical Material Recognition from Local Appearance (2026-09-25)

This is a scientifically relevant new classifier lead, not a candidate ready for evaluation. Its Matador taxonomy contains 57 material categories with hierarchical parent classes, and the model is designed to predict at coarser hierarchy levels when fine classes are uncertain; the paper also reports an out-of-distribution benchmark. However, the paper's reported Matador-C1 evaluation deliberately omits glass, paint, thermoplastic, thermoset, and elastomer classes for insufficient local texture and consolidates several metal types into a generic metal class. The published OOD score is top-1 accuracy, not an explicit calibrated unknown/ambiguous abstention output, and the hierarchy's coarse-parent predictions do not establish Ticket 07's separate unknown and ambiguous signals. The reviewed official author/project/paper pages expose the paper and dataset, but no official executable implementation or model checkpoint with immutable source/weight digests or license/terms was located. The reported model is 28M parameters with a ResNet50 backbone, trained on NVIDIA A6000 Ada; that suggests a modest model-size footprint but does not establish an RX 7900 GRE execution route, <=14 GiB peak memory, or runtime correctness. No artifacts/code/dataset/fixture/truth were accessed or run, and no gate changed. Sources: [author's research page](https://www.cs.columbia.edu/~beveridge/), [ICCV paper](https://openaccess.thecvf.com/content/ICCV2025/html/Beveridge_Hierarchical_Material_Recognition_from_Local_Appearance_ICCV_2025_paper.html), [official supplement](https://openaccess.thecvf.com/content/ICCV2025/supplemental/Beveridge_Hierarchical_Material_Recognition_ICCV_2025_supplemental.pdf), [Columbia CAVE project page](https://cave.cs.columbia.edu/projects/categories/project?cid=Representation+and+Recognition&pid=Hierarchical+Material+Recognition+from+Local+Appearance), [Matador dataset page](https://cave.cs.columbia.edu/repository/Matador). No candidate is selected; Ticket 07 remains acceptance-blocked.

### Cross-view variability development screen (2026-09-25)

The preregistered candidate concatenated mean and population standard deviation over the four views for each of 30 fixed topology-masked appearance cues, then applied the existing dual-ridge classifier/folds/calibrator once. It failed: macro-F1 0.7183261, minimum recall 0.20, coverage 0.60, unknown abstention 0.60, ambiguous abstention 0.40, and 0/1,296 feasible threshold pairs. Only development labels were accessed; heldout truth was never opened, and there was no AMD run. Source, preregistration, feature rows, OOF output, and report are recorded in `api/runtime/adapters/material-identity/evidence/TICKET07_VIEW_VARIABILITY_DEV_SCREEN_2026-09-25.md` and its adjacent hashed artifacts. Ticket 07 remains blocked; gates unchanged.


A new official-source screen examined the RGB-only `akde/mwc-rgbt-waste-sorting` tracklet classifier. It is not eligible for the frozen material-region screen: required labels, unknown/ambiguous calibration, topology-bound region output, checkpoint identity/rights, and RX 7900 GRE evidence are missing. No assets, code, fixture, or truth were accessed. Evidence: `api/runtime/adapters/material-identity/evidence/AKDE_MWC_RGBT_WASTE_SORTING_SOURCE_SCREEN_2026-09-25.md`.

### FMMC official-source screen (2026-09-26)

FMMC (CVPR 2026) is a relevant recent material-classification project, but it does not clear Ticket 07's pre-acquisition gates. Its official project page links a source repository whose README identifies trained logs/checkpoints only through Google Drive without immutable file digests or checkpoint-specific terms; MIT is declared for repository source, but checkpoint/training-data rights were not established. The reported 10/21-class benchmark taxonomies do not establish the distinct required `Rubber/latex`, `Plastic, clear`, and `Paint/plaster/enamel` outputs, unknown/ambiguous abstention, or bare-versus-painted-metal abstention. Mask pooling makes regional classification structurally plausible, but no topology-bound Modly contract, RX 7900 GRE route, <=14 GiB measurement, or frozen-gate metrics are documented. No code, weights, fixture inputs, development labels, or held-out truth were accessed. Full source-screen findings: `api/runtime/adapters/material-identity/evidence/FMMC_OFFICIAL_SOURCE_SCREEN_2026-09-26.md`. Ticket 07 remains acceptance-blocked; gates unchanged.

A source-only refresh screened the recent Apple-DMS SegFormer-B5 v2 work. The official 57-class dense taxonomy includes the required material labels plus `I cannot tell`, `Not on list`, and `Multiple materials`, so it is a useful research lead. However, its published v2-run2 evaluation lists Plastic clear 0.3886, Rubber/latex 0.4111, and Metal 0.5541 class accuracy/recall proxies, below Ticket 07's 0.80 floor (these are not scores on the frozen Modly fixture). Metal remains generic, and the source does not establish the required abstention behavior. Reviewed pages also leave code/checkpoint/dataset rights unresolved and do not expose a complete immutable checkpoint-file digest; ROCm/MIGraphX and <=14 GiB execution are unmeasured. No code, weights, dataset samples, fixture inputs, development labels, or held-out truth were accessed. No thresholds changed. Evidence: `api/runtime/adapters/material-identity/evidence/APPLE_DMS_SEGFORMER_B5_V2_OFFICIAL_SOURCE_SCREEN_2026-09-26.md`. Ticket 07 remains acceptance-blocked.

### DMS46 evaluator integration and owner-approved development screen (2026-09-26)

The evaluator now reconstructs deterministic development/held-out membership from renderer IDs, verifies exact persisted per-asset stage bytes, creates durable split batches with SHA-256 sidecars, and requires valid development commitments before held-out truth access. The owner approved a development-only supported-region coverage screen of >=0.87, derived as (352 required held-out accepts - 2 permitted unknown - 2 permitted ambiguous) / 400 supported. Held-out gates remain unchanged. Focused evaluator and Ticket07 process tests ran 17 tests: 16 passed, one optional-dependency test skipped. Current evaluator source SHA-256 is `9ed7ee2d2aa5ba53d84b0c79e33bcd38f9b5dc384a9c8d3f2e521d5dc506a1a1`; focused evaluator test SHA-256 is `1a1760510e960aad633acb5a92def4f67becd630a3a4cab85ca5eba48c5e6b70`.

A read-only run-path audit found a remaining topology bridge blocker. Fixture `topology_revision` hashes only `faces.tobytes()`, while `StructuredAssets.inspect_geometry()` hashes mesh/primitive identity, vertex/face counts, positions and decoded triangle indices. Thus a correctly imported GLB has a different canonical Modly topology revision, even when face order is preserved, and the evaluator then required direct equality. Do not force-copy the renderer digest into the Structured Asset. **Historical status, superseded:** a verified development-only face-correspondence bridge was subsequently implemented and exercised by the 35-case DMS46 run recorded in `TICKET07_DMS46_DEVELOPMENT_SCREEN_2026-09-26.md`; this old bridge issue is no longer the current blocker. The current blocker is DMS46's failed development quality gates. The earlier note's statement not to start inference was accurate before the correspondence bridge and development run existed.

### DMS46 corrected development result (2026-09-26)

The corrected mapping was applied to the already committed development stage archive without re-running model inference. The raw batch was committed and read back with its SHA-256 sidecar before the separate development-only scorer ran. Its `development_gate_pass` is false: macro-F1 0.0, minimum supported-class recall 0.0, supported-region coverage 0.40, unknown abstention recall 0.40, and ambiguous abstention recall 1.0. The candidate is rejected under the unchanged development criteria; heldout evaluation was not run and heldout truth remained unopened. Full aggregate metrics, per-class results, gate outcomes, artifact digests, and source identities are recorded in [`TICKET07_DMS46_DEVELOPMENT_SCREEN_2026-09-26.md`](../../../api/runtime/adapters/material-identity/evidence/TICKET07_DMS46_DEVELOPMENT_SCREEN_2026-09-26.md). No acceptance gate changed.

### Truth-free DMS46 preprocessing parity follow-up (2026-09-26)

A source-to-adapter audit found no confirmed image decode/color, resize, normalization, dense output, or face-map preprocessor mismatch. A CPU-only synthetic RGB PNG produced exact Pillow versus OpenCV+BGR-to-RGB byte equality for the tested path. The audit did identify stale documentation saying “shortest side” reaches 512; the pinned Apple helper actually scales the longest side to 512, rounds resized dimensions upward, and uses LANCZOS. The two descriptions were corrected in `api/runtime/adapters/material-identity/PROBE_PLAN.md` and `SELECTION_AND_GATES.md`. This does not resolve DMS46's severe development quality failure; no ROCm/GPU parity, full image-format parity, or rendered face-map reprojection was tested. Heldout remains unopened and gates are unchanged. Evidence: `api/runtime/adapters/material-identity/evidence/DMS46_ADAPTER_PREPROCESSING_PARITY_AUDIT_2026-09-26.md`.

### LFM2.5-VL-3B source refresh (2026-09-26)

A primary-source-only refresh found Liquid AI LFM2.5-VL-3B as a possible local single-crop classifier route, pinned to initial Hub revision `3463bfe2e3e8ea4c5d617d231447043a4194789f` with the publisher-reported checkpoint hash recorded in `api/runtime/adapters/material-identity/evidence/TICKET07_LFM25_VL3B_SOURCE_REFRESH_2026-09-26.md`. This is a candidate lead only. Its LFM Open License has a commercial-use revenue condition whose applicability is unresolved; RX 7900 GRE ROCm/MIGraphX behavior, <=14 GiB VRAM, deterministic parsing/calibration, and all frozen development gates remain unproven. No weights, code, packages, fixtures, or truth were accessed. It must clear license/runtime/protocol prerequisites before any development evaluation. The existing DMS46 failure and all ticket gates remain unchanged.

### Process import-order defect fixed (2026-09-26)

The registered process added the API source root before loading Pydantic-backed schemas, allowing the empty `api/typing_extensions.py` marker to shadow the installed dependency and fail before the missing-weight-pin or workspace-containment checks ran. The processor now preloads installed `typing_extensions` before exposing the API root. The named focused test passed under the project Python 3.12 test venv and host Python 3.14.7. Processor SHA-256: `b38c650702f48f3b6c77862cf1573e1fcb15e43828ec172103a6a9021703fa15`. This fixes process import order only; no quality or heldout gate changed, no heldout data was accessed, and Ticket 07 acceptance status is unchanged.

### VLMaterial official-source screen (2026-09-27)

The distinct VLMaterial project is a procedural material-program generation/rendering workflow, not a topology-bound material-identity classifier. Its published documentation does not provide Ticket 07's five-class/unknown/ambiguous outputs, calibration, or bare-versus-painted-metal abstention. The inference checkpoint ZIP has no immutable file identity in the inspected page; MIT is stated for code and pretrained weights while the separate material dataset is CC BY-NC 4.0. Its documented LLaVA-NeXT 8B / 48 GB VRAM recommendation, 8×H100 scripts, and CUDA 11.8 setup provide no viable RX 7900 GRE <=14 GiB path. Rejected before acquisition or inference; no fixture/truth access and no gate change. Evidence: `api/runtime/adapters/material-identity/evidence/TICKET07_VLMATERIAL_OFFICIAL_SOURCE_SCREEN_2026-09-27.md`.

### Hierarchical material recognition source screen (2026-09-27)

Columbia CAVE's 57-class local-appearance hierarchy is a relevant image-classification lead but does not establish the frozen Ticket 07 taxonomy or calibrated `unknown`/`ambiguous` handling. Its Matador-C1 report omits glass, plastics, and paint and combines metal categories; the inspected official sources expose no immutable inference checkpoint, checkpoint terms, caller-topology region interface, or AMD/RX 7900 GRE resource evidence. Rejected before acquisition or evaluation; no gates changed. Evidence: `api/runtime/adapters/material-identity/evidence/TICKET07_HIERARCHICAL_MATERIAL_RECOGNITION_SOURCE_SCREEN_2026-09-27.md`.

### YOLOE official-source screen (2026-09-27)

Ultralytics YOLOE is a relevant open-vocabulary instance-segmentation lead, but does not qualify for acquisition or evaluation. Official docs describe general object detection/segmentation, report 22–40 LVIS minival mAP for prompted checkpoints, and explicitly say zero-shot accuracy is below class-trained models; these are not material-identity results. They do not establish the five frozen material classes, calibrated unknown/ambiguous or bare-versus-painted-metal abstention, a topology-bound mesh interface, immutable checkpoint digests/terms, or AMD/RX 7900 GRE <=14 GiB behavior. Deployment notes document NVIDIA 4–8 GB VRAM. Ultralytics identifies AGPL-3.0 and an Enterprise license for its code/models, so applicable terms would also need resolution. Rejected before acquisition or evaluation; no checkpoint/code/package/fixture/label/truth access and no gate changes. Evidence: [`TICKET07_YOLOE_OFFICIAL_SOURCE_SCREEN_2026-09-27.md`](../../../api/runtime/adapters/material-identity/evidence/TICKET07_YOLOE_OFFICIAL_SOURCE_SCREEN_2026-09-27.md).
# Additional community-checkpoint screen (2026-09-27)

The source-only MINC-SigLIP2 screen found a pinned community model card and a published digest, but its reported macro-F1 (0.7700) and glass/metal/plastic recalls (0.6753/0.6626/0.5304) fall below the unchanged Ticket 07 gates. Its labels also omit required distinctions and it has no unknown/ambiguous abstention behavior. No weight, code, fixture, or truth was accessed. Reject before acquisition; details and primary sources: `api/runtime/adapters/material-identity/evidence/TICKET07_MINC_SIGLIP2_COMMUNITY_CHECKPOINT_SCREEN_2026-09-27.md`. Ticket 07 remains acceptance-blocked.

### CLAMP visuo-haptic material recognition screen (2026-09-27)

Cornell CLAMP is a material-recognition research lead, but its published method relies on haptic observations; the official sources do not establish an image-only material-region classification route. They also do not establish Ticket 07's unknown/ambiguous behavior, full immutable inference/weight identities and terms, or RX 7900 GRE resource qualification. Rejected before acquisition. No assets or fixture data were accessed; no gates changed. Details: `api/runtime/adapters/material-identity/evidence/TICKET07_CLAMP_VISUOHAPTIC_SOURCE_SCREEN_2026-09-27.md`.

OpenMR's open-set method uses robot tactile/e-skin observations, not image crops, and its eight tactile textures do not establish the frozen visual taxonomy or abstention behavior. The inspected publisher sources also do not identify an immutable deployable image checkpoint, complete terms, frozen-fixture metrics, or AMD target evidence. Rejected before acquisition; no code or data assets were accessed and no gates changed. Report: `api/runtime/adapters/material-identity/evidence/TICKET07_OPENMR_TACTILE_SOURCE_SCREEN_2026-09-27.md`.

Microsoft RegionCLIP can score text concepts against image regions, but its official qualification and results concern open-vocabulary object detection. They do not establish material-specific quality, Ticket 07's `unknown`/`ambiguous` outputs, immutable weight identity and full terms, or AMD resource qualification. Rejected before acquisition; no code or data or fixture assets were accessed and no gate changed. Report: `api/runtime/adapters/material-identity/evidence/TICKET07_REGIONCLIP_OFFICIAL_SOURCE_SCREEN_2026-09-27.md`.

MaskTerial has a versioned checkpoint archive, but its task is microscopy classification of exfoliated two-dimensional materials, with uncertainty around thickness and optical contrast. It does not establish Modly's surface-material taxonomy or `unknown`/`ambiguous` contract, and its official installation is CUDA-based with no AMD evidence. Rejected before acquisition; no weights or fixtures were accessed and no gates changed. Report: `api/runtime/adapters/material-identity/evidence/TICKET07_MASKTERIAL_OFFICIAL_SOURCE_SCREEN_2026-09-27.md`.

### SigLIP2/MINC source and target-readiness audit (2026-09-27)

The pinned MINC fine-tune is immutable but does not cover Rubber/latex, does not separate bare from painted metal, and uses forced single-label classification without documented calibrated unknown/ambiguous outputs or dense topology-bound material regions. Its own reported macro-F1 is 0.7700 with glass/metal/plastic recalls below Ticket 07's unchanged source-screen floors. The inspected publisher sources also do not establish <=14 GiB RX 7900 GRE execution; dataset source-photo rights remain unresolved. The generic SigLIP2 ridge path already failed development gates and its earlier screen was CPU-only. Neither route is recommended for a target-GPU slot. No weights, dataset, fixture/truth, or GPU were accessed and no gates changed. Evidence: `api/runtime/adapters/material-identity/evidence/TICKET07_SIGLIP2_MINC_RUNTIME_PROVENANCE_AUDIT_2026-09-27.md`.

### SMARC minimal-cue material classification screen (2026-09-27)

SMARC reports 85.10% classification accuracy on Touch-and-Go using surface classes such as grass, concrete, wood, and rock. The inspected official sources do not establish Modly's five material labels, unknown/ambiguous abstention, topology-bound material-region evidence, immutable inference/checkpoint identity and terms, or AMD <=14 GiB target support. It was rejected before acquisition or evaluation. No model, data, fixtures, truth, or GPU were accessed and no gates changed. Primary-source findings: `api/runtime/adapters/material-identity/evidence/TICKET07_SMARC_MINIMAL_CUE_OFFICIAL_SOURCE_SCREEN_2026-09-27.md`.

### OvarNet region-attribute source screen (2026-09-27)

OvarNet is a relevant open-vocabulary object-attribute research lead with box-crop attribute predictions, including a generic `metal` attribute example. The inspected sources do not establish Modly's five material classes or bare/painted-metal distinction, calibrated `unknown`/`ambiguous` behavior, dense topology-bound material-region output, an immutable inference checkpoint/source identity, complete licenses, or AMD/RX 7900 GRE memory evidence. A bounding-box crop around a mapped region would include out-of-mask pixels and is not an established compatible input. Rejected before acquisition or evaluation; no code, weights, dataset, fixture, truth, or GPU were accessed and no gates changed. Report: `api/runtime/adapters/material-identity/evidence/TICKET07_OVARNET_OPEN_VOCAB_REGION_ATTRIBUTE_SOURCE_SCREEN_2026-09-27.md`.


### Material Palette official-source screen (2026-09-27)

Material Palette is ineligible before acquisition/evaluation: its documented task generates and decomposes image textures rather than classifying Modly's fixed material identities, it lacks calibrated unknown/ambiguous outputs and topology-bound region evidence, and its deployed model/code/weight/terms and RX 7900 GRE resource gates are not established. No code, weights, examples, fixture, truth, or GPU were accessed. No gates changed. Full primary-source findings: [`TICKET07_MATERIAL_PALETTE_OFFICIAL_SOURCE_SCREEN_2026-09-27.md`](../../../api/runtime/adapters/material-identity/evidence/TICKET07_MATERIAL_PALETTE_OFFICIAL_SOURCE_SCREEN_2026-09-27.md).


### MatForge official-source screen (2026-09-28)

MatForge is an adjacent lead for dense PBR maps and broad material groups, but its public sources do not establish the required five-class topology-bound identity classifier with selective abstention, exact deployable weight identities and complete terms, RX 7900 GRE support within the memory ceiling, or frozen Modly identity/abstention results. Rejected before acquisition/evaluation; no model, code, data, fixtures, truth, or GPU was accessed and no gates changed. Report: [`TICKET07_MATFORGE_OFFICIAL_SOURCE_SCREEN_2026-09-28.md`](../../../api/runtime/adapters/material-identity/evidence/TICKET07_MATFORGE_OFFICIAL_SOURCE_SCREEN_2026-09-28.md).


### MatForge official-source screen (2026-09-28)

MatForge is an adjacent lead for dense PBR maps and broad material groups, but its public sources do not establish the required five-class topology-bound identity classifier with selective abstention, exact deployable weight identities and complete terms, RX 7900 GRE support within the memory ceiling, or frozen Modly identity/abstention results. Rejected before acquisition/evaluation; no model, code, data, fixtures, truth, or GPU was accessed and no gates changed. Report: [`TICKET07_MATFORGE_OFFICIAL_SOURCE_SCREEN_2026-09-28.md`](../../../api/runtime/adapters/material-identity/evidence/TICKET07_MATFORGE_OFFICIAL_SOURCE_SCREEN_2026-09-28.md).


### VLMAT camera-radar official-source screen (2026-09-28)

VLMAT reports uncertainty-aware material identification, but relies on camera-radar fusion and does not establish Modly's fixed image-region labels, calibrated abstention, topology binding, or an eligible AMD implementation. Rejected before acquisition/evaluation; no code, model, fixtures, truth, or GPU were accessed and no gates changed. Report: [`TICKET07_VLMATERIAL_CAMERA_RADAR_SOURCE_SCREEN_2026-09-28.md`](../../../api/runtime/adapters/material-identity/evidence/TICKET07_VLMATERIAL_CAMERA_RADAR_SOURCE_SCREEN_2026-09-28.md).


### MaterialSeg3D/MIO++ official-source screen (2026-09-28)

MaterialSeg3D is a relevant dense, multi-view material-to-UV lead, but official evidence does not establish the required clear-plastic label or calibrated unknown/ambiguous and metal-subtype abstention, Modly's topology/provenance contract, immutable released model weights and terms, RX 7900 GRE <=14 GiB execution, or frozen Modly fixture results. Rejected before acquisition/evaluation; no assets, fixtures, truth, inference, or GPU accessed; gates unchanged. Report: [`TICKET07_MATERIALSEG3D_FIXED_DENSE_3D_SOURCE_SCREEN_2026-09-28.md`](../../../api/runtime/adapters/material-identity/evidence/TICKET07_MATERIALSEG3D_FIXED_DENSE_3D_SOURCE_SCREEN_2026-09-28.md).

### 3DCoMPaT++ point-cloud material segmentation official-source screen (2026-09-28)

3DCoMPaT++ is a relevant RGB point-cloud/per-point material segmentation route, but its published winner predicts coarse fixed classes, documents no required unknown/ambiguous or metal-subtype abstention, and has no released immutable winner checkpoint (official README marks it “will be uploaded soon” / `TODO`). Its documented coarse taxonomy does not establish clear plastic or paint/plaster/enamel labels; Modly topology-bound provenance and RX 7900 GRE <=14 GiB execution are also unproven. Rejected before acquisition/evaluation. No assets, fixtures, truth, inference, or GPU accessed and no gate/pin changed. Report: [`TICKET07_3DCOMPATPP_POINTCLOUD_SOURCE_SCREEN_2026-09-28.md`](../../../api/runtime/adapters/material-identity/evidence/TICKET07_3DCOMPATPP_POINTCLOUD_SOURCE_SCREEN_2026-09-28.md).

### API contract suite rerun (2026-09-28)

The literal documented invocation using `PYTHONPATH=api` currently fails during import because the empty source marker `api/typing_extensions.py` shadows the installed `typing_extensions` package, and the installed `pydantic-core` expects `typing_extensions.Sentinel`. Preloading the installed package before adding `api` to `sys.path` avoids the shadowing, matching the explicit import-order protection already used by the registered process. Exact successful command:

```sh
.modly-amd-runtime/api-test-venv/bin/python -c 'import typing_extensions, sys, unittest; sys.path.insert(0, "api"); suite = unittest.defaultTestLoader.discover("api/tests", pattern="test_ticket07_material_identity.py"); result = unittest.TextTestRunner(verbosity=2).run(suite); raise SystemExit(not result.wasSuccessful())'
```

Outcome: 15 tests run, 14 passed and one model-root integration test skipped because `MODLY_SIGLIP2_MODEL_ROOT` is not set in the CPU test environment. This verifies the API contract module only; it does not replace the separately recorded pinned-image SigLIP2 process test or establish quality/AMD acceptance. The gate-source digest discrepancy and failed candidate quality gates remain unchanged; no fixture truth or model inference was accessed in this rerun.

### CPU contract-suite check (2026-09-28)

The project API venv Ticket 07 discovery ran 71 tests with 6 optional-Pillow skips, but it is not green: the owner-approved selection-gate source pin test fails because `SELECTION_AND_GATES.md` currently hashes to `sha256:bbdb9e384fcc96f7abbbb33d92d09f37f092f18b4fe32f8a5744aa1eb85c5ca0`, while the development preregistration and runner pin `sha256:6998d86185bf1d91d3af0fe2e3643f232cb0d6765c554109a962f07b8e14caea`. The same discovery also cannot import the Qwen3 scoring module because Torch is absent from this CPU venv. No gate pin was changed during this check. The hash discrepancy must be reconciled against the preregistered frozen source before relying on further DMS46 development evaluations; do not silently update the source or lock. The Torch-only module needs the project AMD image or an equivalent project-local Torch environment.

The Torch-only Qwen3 scorer contract test was then run inside the project GeoSAM2 image without GPU devices, preloading the image's installed `typing_extensions`; it passed 9/9. This resolves the CPU-venv import limitation for that module only. The remaining full-suite failure is the stale gate-source digest pin; no hash update or gate change has been made pending audit.

### Fixed-gate source digest discrepancy audit (2026-09-28)

The current `SELECTION_AND_GATES.md` hashes to `sha256:bbdb9e384fcc96f7abbbb33d92d09f37f092f18b4fe32f8a5744aa1eb85c5ca0`, while the frozen DMS46 preregistration and runner pin `sha256:6998d86185bf1d91d3af0fe2e3643f232cb0d6765c554109a962f07b8e14caea`. The preregistered runner checks exact file bytes; its gate-pin contract test correctly fails on this mismatch. Current held-out numeric thresholds, cohorts, denominator, resource and backend requirements, and the approved 0.87 development-only supported-region floor appear consistent with the preregistration. However, no byte-for-byte baseline or usable Git history exists to prove that the complete drift is only additive candidate/evidence text and a DMS46 preprocessing wording correction. No hash or gate changed. Before launching further preregistered scoring, restore authoritative frozen bytes if recoverable; otherwise use a reviewed versioned preregistration that restates the unchanged limits and pins the current source. Do not silently repin.

### Frozen-source recovery recheck (2026-09-28)

I rechecked the current workspace before deciding whether DMS46 development scoring can safely resume. `sha256sum api/runtime/adapters/material-identity/SELECTION_AND_GATES.md` still returns `bbdb9e384fcc96f7abbbb33d92d09f37f092f18b4fe32f8a5744aa1eb85c5ca0`; the preregistration still names `6998d86185bf1d91d3af0fe2e3643f232cb0d6765c554109a962f07b8e14caea`. The expected digest appears in the preregistration, DMS46 audit/development evidence, and `development_batch_runner.py`, but those are references, not copies of the frozen file bytes. A search for selection/gate backups, Ticket07 backup files, and the expected digest under the project and `/tmp` found no recoverable baseline. Git recovery is unavailable in this checkout: `git rev-parse` and `git cat-file` report that this directory is not a Git repository. The existing DMS46 audit records the old digest and the one documented wording correction, but does not contain the original full file, so it cannot prove that a hand-reconstructed copy would match the preregistered bytes.

Therefore no further DMS46 preregistered scoring can safely resume from the current frozen record. Preserve the current source and all pins; do not regenerate a file to chase the old digest, silently change the pin, or consult held-out truth. Safe next steps are limited to locating an authoritative copy outside this checkout or preparing a versioned preregistration for independent review that explicitly retains every existing threshold and cohort rule. This recheck used read-only workspace searches and did not access fixture/truth data, download model assets, or run inference.

### New-source discovery refresh (2026-09-28)

A bounded primary-source search found no eligible new classifier. The newly surfaced FMMC, Apple-DMS, and MaterialSeg3D++ leads were already screened; DenseVLM's official repository describes general dense prediction without establishing Modly's material labels, abstention behavior, or other frozen gates. This is a source-discovery screen only, not proof no candidate exists. No code, weights, datasets, fixtures, or truth were accessed; no gates changed. Full findings and official links: [`TICKET07_NEW_SOURCE_DISCOVERY_REFRESH_2026-09-28.md`](../../../api/runtime/adapters/material-identity/evidence/TICKET07_NEW_SOURCE_DISCOVERY_REFRESH_2026-09-28.md).

### MINC GoogLeNet Caffe implementation source screen (2026-09-28)

The MINC GoogLeNet Caffe implementation is a distinct dense-classification lead, but its public repository requires a user-supplied `.caffemodel` and does not identify immutable checkpoint bytes or checkpoint-specific terms. Its documented Caffe/CUDA and Intel demo routes do not establish AMD support, and the sources do not establish the frozen label/abstention contract or Modly fixture quality. Rejected before acquisition or fixture evaluation; no weights, data, fixtures, truth, or GPU were accessed and no gates changed. Report: [`TICKET07_MINC_CAFFE_THESIS_IMPLEMENTATION_SOURCE_SCREEN_2026-09-28.md`](../../../api/runtime/adapters/material-identity/evidence/TICKET07_MINC_CAFFE_THESIS_IMPLEMENTATION_SOURCE_SCREEN_2026-09-28.md).

### MatPredict official-source screen (2026-09-28)

MatPredict is a synthetic-data lead only, not an eligible identity classifier
yet. Its current dataset card/README describe per-pixel material labels, but
the linked paper describes its released models as base-color/roughness
regressors and says meshes have one uniform material. No trained segmentation
checkpoint, immutable checkpoint identity/terms, five-label selective
abstention behavior, Modly topology binding, or RX 7900 GRE evidence is
established. The Hugging Face page labels the dataset MIT, while the README
still describes an anonymous dataset link; upstream Replica/ReplicaCAD and
MatSynth terms/provenance also need reconciliation. Rejected before
acquisition or evaluation; no data, code archive, checkpoint, fixture, truth,
or GPU was accessed and no gate changed. Evidence:
[`TICKET07_MATPREDICT_OFFICIAL_SOURCE_SCREEN_2026-09-28.md`](../../../api/runtime/adapters/material-identity/evidence/TICKET07_MATPREDICT_OFFICIAL_SOURCE_SCREEN_2026-09-28.md).

Material Magic Wand is a distinct 3D method that groups already segmented mesh parts by material similarity. Its published group IDs and selected part indices are not Ticket 07's fixed material names or its `unknown`/`ambiguous` selective outputs; the public interface also does not establish Modly's topology-revision binding, checkpoint terms/digest, AMD runtime, or required fixture quality. Rejected before model, code, dataset, fixture, or truth acquisition. No gates changed. Report: [`TICKET07_MATERIAL_MAGIC_WAND_SOURCE_SCREEN_2026-09-28.md`](../../../api/runtime/adapters/material-identity/evidence/TICKET07_MATERIAL_MAGIC_WAND_SOURCE_SCREEN_2026-09-28.md).

### Material Magic Wand official-source screen (2026-09-28)

Material Magic Wand is a distinct 3D method that groups already segmented mesh parts by material similarity. Its published group IDs and selected part indices are not Ticket 07's fixed material names or its `unknown`/`ambiguous` selective outputs; the public interface also does not establish Modly's topology-revision binding, checkpoint terms/digest, AMD runtime, or required fixture quality. Rejected before model, code, dataset, fixture, or truth acquisition. No gates changed. Report: [`TICKET07_MATERIAL_MAGIC_WAND_SOURCE_SCREEN_2026-09-28.md`](../../../api/runtime/adapters/material-identity/evidence/TICKET07_MATERIAL_MAGIC_WAND_SOURCE_SCREEN_2026-09-28.md).

### Glass-only expert source screen (2026-09-28)

GEM and AAAI26-MSNet are glass-surface maskers, not five-class material identity classifiers with the frozen selective-abstention contract. GEM's published glass IoU values are below 0.90 (not directly comparable to Modly macro-F1), and its source/checkpoint bytes are not frozen in the repository; MSNet states academic-research-only use, documents CUDA/24 GB hardware, and supplies its trained weights via Google Drive. Both were rejected before acquisition or fixture evaluation. No code, weights, datasets, fixture inputs, development labels, held-out truth, or GPU were accessed, and no gate changed. Evidence: [`TICKET07_GLASS_ONLY_EXPERT_SOURCE_SCREEN_2026-09-28.md`](../../../api/runtime/adapters/material-identity/evidence/TICKET07_GLASS_ONLY_EXPERT_SOURCE_SCREEN_2026-09-28.md).
### FMMC and ObjectFolder classifier recheck (2026-09-28)

An additional primary-source screen found no eligible new candidate. The FMMC
CVPR 2026 paper reports masked-region material classification but says code
and data will be released; no pinned inference checkpoint or checkpoint terms
were identified. Its reported classes and evaluation do not establish Modly's
fixed taxonomy, `unknown`/`ambiguous` and bare-versus-painted-metal
abstention, topology-bound outputs, or RX 7900 GRE memory/runtime gate. The
official [CVPR paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Lin_Harnessing_the_Power_of_Foundation_Models_for_Accurate_Material_Classification_CVPR_2026_paper.pdf)
and [ObjectFolder classifier repository](https://github.com/objectfolder/material-classification)
were also checked. ObjectFolder is an object-level, seven-class classifier
without the required selective output contract or pinned checkpoint terms.
No code, weights, data, fixtures, truth, or GPU were accessed. Ticket 07 stays
open; acceptance gates are unchanged.

### FMMC public release recheck (2026-09-28)

The earlier statement that FMMC had no public code/checkpoint links is stale:
the authors now link an MIT code repository and trained-log download links.
The released DINOv2 evaluation script uses an eight-class head that omits the
required glass and paint/plaster/enamel labels and emits argmax without the
frozen abstention behavior. Checkpoint byte identity, checkpoint terms,
topology-bound output, and RX 7900 GRE evidence are also unproved. The
released route is not eligible for frozen Ticket 07 evaluation. An upstream
21-class configuration with matching pinned checkpoint and terms is the next
specific lead. No assets, fixture, truth, or GPU were accessed. Full evidence:
[`TICKET07_FMMC_RELEASE_RECHECK_2026-09-28.md`](../../../api/runtime/adapters/material-identity/evidence/TICKET07_FMMC_RELEASE_RECHECK_2026-09-28.md).
