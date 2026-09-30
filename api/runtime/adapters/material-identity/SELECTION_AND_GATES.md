# Material identity classifier selection and gates

Status: implementation is in place; model qualification and Ticket 07 acceptance
are pending. Gates below were frozen before comparing classifiers and may not be
lowered to make a candidate pass.

## Frozen fixture and acceptance gates

The fixed held-out fixture set must contain at least 20 independently rendered
regions per material class, spread over at least five object instances per
class. Required source labels are `Rubber/latex`, `Glass`, `Plastic, clear`,
`Paint/plaster/enamel`, and `Metal`. Metal has separate truth cohorts for bare
metal and painted metal; the classifier must abstain on that distinction when
the evidence only supports the upstream generic `Metal` category. Include at
least 20 unknown/out-of-taxonomy regions and 20 deliberately ambiguous
multi-material or low-resolution regions. Split fixture objects, not image
views, between calibration/development and held-out evaluation.

The frozen rendered fixture contains 440 held-out regions: 400 supported, 20
unknown, and 20 ambiguous. Coverage is the number of regions receiving a
single supported label divided by **all 440 held-out regions**, including
unknown and ambiguous cohorts. On held-out regions, normalized supported labels
must achieve macro-F1 >= 0.85 and each of the five source classes must have
recall >= 0.80. All-region selective coverage must be >= 0.80 (at least 352
accepted regions). At least 90% of declared ambiguous and out-of-taxonomy
examples must be emitted as `ambiguous` or `unknown`, respectively; a forced
class on these examples is a failure. The coverage and abstention criteria are
conjunctive: at least 352/440 single-label outputs while at least 90% of each
20-region OOD cohort abstains. Those abstention gates permit at most two
accepted unknown and two accepted ambiguous regions, so at least 348/400
supported regions (87%) must be accepted to reach 352 overall. Supported-only
coverage is diagnostic and cannot satisfy the gate. Report confusion matrices,
per-class support/precision/recall/F1, all-region and supported-only coverage,
abstention recall, and all thresholds. Candidate ranking over multiple views
uses only per-pixel class identity votes; vote fractions are derived and remain
uncalibrated confidence, not model probabilities.

### Development-only candidate screen

With 100 supported and 40 OOD views in the development split, applying the
held-out 0.80 all-region coverage floor together with 0.90 abstention for each
20-view OOD cohort is impossible: even accepting every supported view and the
maximum two allowed OOD views per cohort reaches only 104/140 (0.743). The
owner approved a development-only supported-region coverage floor of 0.87,
derived from the held-out conjunction above. Development still requires
macro-F1 >= 0.85, minimum supported-class recall >= 0.80, and >= 0.90 unknown
and ambiguous abstention. This rule applies only to candidate stop/go on the
development split; held-out all-region coverage remains >= 0.80 over all 440
regions, and every held-out criterion above is unchanged. Report development
all-region coverage as a diagnostic; do not use it as a separate gate.

The 16 GB target gate requires measured peak VRAM <= 14 GiB for the isolated
classifier stage, no CUDA or NVIDIA-only dependency, and a Modly AMD Runtime
report for the actual backend. On RX 7900 GRE, Torch-MIGraphX is preferred only
when the pixel outputs match PyTorch ROCm at atol=1e-4 / rtol=1e-3 and warm
latency is no worse than ROCm. Otherwise record and use explicit PyTorch ROCm
fallback. CPU is allowed for development tests only and cannot satisfy the AMD
gate. Adapter integration must remain in Modly's JSON-lines process-extension
workflow seam.

## Candidate screen

| Candidate | Task and evidence | Gate result |
|---|---|---|
| Apple DMS46 | Official Apple DMS release advertises a TorchScript model that predicts 46 dense material classes; its published taxonomy includes glass, clear plastic, rubber/latex, metal, and paint/plaster/enamel. The pinned sample inference code scales the longest image side to 512 (including upscaling smaller inputs), rounds output dimensions upward, uses LANCZOS, applies ImageNet normalization, and emits dense label maps. | Selected for a bounded AMD evaluation. The owner approved local use of the pretrained checkpoint; this project does not use the DMS training dataset and does not redistribute the checkpoint. Its implementation and taxonomy source are pinned to [`a379a63e9435e32134a465eb31ecb0aefebed985`](https://github.com/apple-aiml-research/ml-dms-dataset/tree/a379a63e9435e32134a465eb31ecb0aefebed985). The official archive is staged and local archive/checkpoint SHA256 values are recorded in `ASSET_LOCK.json`; Apple publishes no checksum. Not accepted: target ROCm/MIGraphX correctness, latency, VRAM, and fixed-set accuracy remain unmeasured. A DMS46-compatible evaluator must be implemented and frozen before fixture scoring or truth access. The coarse `Metal` category cannot independently distinguish bare from painted metal; Ticket acceptance requires abstention when no stronger evidence is available. |
| OpenAI CLIP | Official model card describes zero-shot matching and states that deployment needs context-specific study; even constrained image search requires thorough in-domain testing with a fixed taxonomy. It is image-level scoring, not dense material segmentation. | Not selected as the reference material-region classifier. It would need an additional dense mask/reduction path and still needs the same fixture evaluation; no comparative score is claimed. |
| MatSpectNet | WACV 2025 RGB-to-hyperspectral material segmentation evaluated on LMD and OpenSurfaces. | Ineligible before acquisition: official README links an unpinned Google Drive model checkpoint with no stated checkpoint terms or root-level code license; published sources do not establish Ticket07 unknown/ambiguous or metal-subtype abstention; repository requirements include `faiss-gpu`, pin Torch 1.12.1, and target eight NVIDIA RTX 3090 GPUs without an evidenced AMD path. Source screen: [`MATSPECTNET_OFFICIAL_SOURCE_SCREEN_2026-09-26.md`](evidence/MATSPECTNET_OFFICIAL_SOURCE_SCREEN_2026-09-26.md). |
| Apple-DMS SegFormer-B5 v2-run2 | Recent dense 57-class Apple-DMS classifier; authors publish code, dataset mirror, checkpoint, and per-class scores. | Relevant but not eligible for fixture evaluation: source card reports below-0.80 class accuracy/recall proxies for Plastic clear (0.3886), Rubber/latex (0.4111), and Metal (0.5541); these are publisher-set metrics, not Modly fixture scores. It uses generic Metal, does not establish unknown/ambiguous or bare-vs-painted abstention, and reviewed source/model/dataset rights plus immutable file digest remain unresolved. No weights/code/data/fixture/truth accessed; AMD/VRAM unmeasured. Source screen: [`APPLE_DMS_SEGFORMER_B5_V2_OFFICIAL_SOURCE_SCREEN_2026-09-26.md`](evidence/APPLE_DMS_SEGFORMER_B5_V2_OFFICIAL_SOURCE_SCREEN_2026-09-26.md). |

Sources: [Apple DMS model/data description](https://github.com/apple-aiml-research/ml-dms-dataset/tree/a379a63e9435e32134a465eb31ecb0aefebed985),
[Apple DMS reference inference script](https://github.com/apple-aiml-research/ml-dms-dataset/blob/a379a63e9435e32134a465eb31ecb0aefebed985/inference.py),
[Apple DMS taxonomy](https://github.com/apple-aiml-research/ml-dms-dataset/blob/a379a63e9435e32134a465eb31ecb0aefebed985/taxonomy.json),
[OpenAI CLIP model card](https://github.com/openai/CLIP/blob/main/model-card.md).

The DMS training dataset is not used by this project, and its CC-BY-NC 4.0
terms are not treated as the license for inference with the pretrained model.
The pinned README identifies a separate Apple `LICENSE.txt` for the checkpoint;
the owner approved local evaluation without checkpoint redistribution in this
task. Original RGB image rights remain separate. The exact pinned source has
the referenced `ACKNOWLEDGMENTS.txt` file (American spelling, `.txt` suffix).
The official archive is staged under
`.modly-amd-runtime/models/dms46/`. It contains only `DMS46_v1.pt`, whose exact
bytes and archive/checkpoint SHA256 values are recorded in `ASSET_LOCK.json`.
Apple's pinned README advertises 170 MB but exposes no publisher checksum.

## Runtime behavior

`reference-material-identity` loads only the official workspace-local DMS46
TorchScript file whose SHA256 matches the declared digest and a taxonomy file
whose SHA256 also matches. It requires the exact pinned upstream commit
`a379a63e9435e32134a465eb31ecb0aefebed985`, processes one view at a time through `AMDInferenceRuntime`, and
preserves the source observation identity and calibrated face map. The adapter
does not accept precomputed caller-supplied class maps through the process
interface. Its internal projection/fusion routine is unit tested independently
from model execution. The stage artifact contains the actual inference inputs,
derived dense class maps, per-region candidates, topology revision, and digests.

Input bounds: 64 distinct views; 2,000,000 pixels per view; 2,000,000 pixels
per classifier resize canvas; 64 MiB per observation image; 512 MiB for weights;
2 MiB for taxonomy JSON. Dimensions or payloads above those bounds fail before
prediction results are attached.

## Current evidence and blocker

The CPU contract tests exercise topology mapping, candidate ranking, unknown and
ambiguous behavior, alias normalization, source identity digests, preservation
of PBR assertions, and fail-closed behavior when model identity is not pinned.
They do not run DMS46 and do not establish material-label accuracy. The
checkpoint is staged and byte-hashed but has not been loaded or probed. No
RX 7900 GRE model probe has been run. The bounded target runbook is
`PROBE_PLAN.md`; it requires a real topology-bound observation and a reserved
RX 7900 GRE slot. Fixture scoring additionally requires the frozen DMS46
evaluator recorded in `evidence/DMS46_EVALUATION_PREREGISTRATION_2026-09-26.md`.
The taxonomy is staged from the
immutable source commit at `cache/taxonomy.json` (9,157 bytes, SHA256
`5fb9f5b6d81b800468db78532a9284f060ddc2e2a44b8214658be797029439ea`).
