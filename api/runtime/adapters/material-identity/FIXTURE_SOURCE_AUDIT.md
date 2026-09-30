# Ticket 07 fixed-set source audit (CPU only)

Status: source/label audit complete; fixed evaluation fixture is **not ready**. No RGB images or pixel maps were fetched or extracted for this audit. No classifier weights were loaded and no GPU was used.

## Immutable source and local archive evidence

The annotation sources below are pinned to the official `apple-aiml-research/ml-dms-dataset` repository revision `a379a63e9435e32134a465eb31ecb0aefebed985`. The official README describes 3 million polygons across 44 thousand RGB images, says the RGB images must be obtained separately from Open Images, licenses the DMS dataset CC BY-NC 4.0, and explicitly requires separate licensing for original RGB imagery. The official README's data-license statement is not permission to redistribute source photos. [Pinned DMS README](https://github.com/apple-aiml-research/ml-dms-dataset/blob/a379a63e9435e32134a465eb31ecb0aefebed985/README.md)

| Artifact | Exact bytes | SHA-256 | Handling |
|---|---:|---|---|
| `.modly-amd-runtime/models/dms46/dataset/dms_v1_labels.zip` (fused labels) | 660,365,654 | `78e22ffa002e168c096a007ad3e50aa4fb5f1d2a04182c31f1a73733d1867290` | ZIP integrity and safe member paths checked; only metadata/taxonomy/license extracted to ignored runtime cache |
| `.modly-amd-runtime/models/dms46/dataset/dms_v1_polygons.zip` (opinions and corrections) | 2,523,082,947 | `de86053fa13aa5344957770a91f4598529a9f556782490b07e6a64b389210a01` | ZIP integrity and safe member paths checked; streamed/read annotation members for CPU-side metadata and class analysis |
| `DMS_v1/info.json.gz` | 22,719,415 | `32e21b05ef381d5fd85b20ce384a88bb10416b230e67c74ceedc86a6bfbbce6b` | Extracted under ignored runtime cache; 44,557 entries |
| `DMS_v1/taxonomy.json` | 9,157 | `5fb9f5b6d81b800468db78532a9284f060ddc2e2a44b8214658be797029439ea` | Extracted under ignored runtime cache |
| `DMS_v1/LICENSE` | 268 | `c45a9bc83ca08a89e034412e21db26a03438aad804d6949bb59bb739d1ae2180` | Extracted under ignored runtime cache; states CC BY-NC 4.0 |

The official Open Images metadata repository documents the image metadata fields (including original landing URL, declared license, author, title, and source URL). It also cautions that the dataset's image-license records are not a warranty and that each image's license must be verified individually. [Open Images metadata documentation](https://github.com/openimages/dataset/blob/master/READMEV1.md). The CC BY 2.0 deed allows redistribution and adaptation, including commercial use, subject to appropriate credit, license link, and change indication; it warns that other rights may apply. [CC BY 2.0 deed](https://creativecommons.org/licenses/by/2.0/).

## Polygon label ID resolution and counts

The raw polygon `shapes[*][0]` IDs and correction `shapes[*][0]` IDs resolve by **direct index into `taxonomy.json`'s `names` array**, not by `semantic_labels`. This was cross-checked against correction photo 22494: its polygon label ID 52 is `Wood`; its `info.json.gz` histogram/fraction has nonzero taxonomy index 52 and its `materials` list includes `Wood`. `semantic_labels` is a separate legacy semantic-ID mapping and must not be used to relabel raw polygon shape IDs. The two special classes needed for rejection evaluation are taxonomy index 22 (`I cannot tell`) and index 55 (`Multiple materials`).

The polygon opinion archive parses as 100,014 photo/annotator records, 8,924,398 polygon shapes, and 56 distinct observed raw label IDs. It contains 109,377 polygons with label 22 over 13,913 photos and 96,285 polygons with label 55 over 15,629 photos. These are **polygon support counts**, not held-out region/object counts. The fused label archive's histogram does not preserve usable support for either rejection cohort, so the polygon annotations are required for this fixture lane.

`info.json.gz` has 44,557 DMS image records and Open Images IDs; it does not include a stable physical object-instance ID. `detected_objects` is only a list of category strings. Therefore image IDs cannot be substituted for manually verified distinct object instances, and the required five distinct object instances/class split has not been established. No object-disjoint evaluation manifest has been produced.

## Individual image license/attribution spot checks (no pixel fetch)

The following are candidate annotation records only, not accepted held-out examples. DMS photo IDs and Open Images IDs are kept here to make the checks reproducible.

| Cohort candidate | DMS photo / Open Images ID | DMS title / author | Original landing URL | Source result |
|---|---|---|---|---|
| `I cannot tell` (taxonomy ID 22) | `32189` / `854f36146aacfd23` | “27 push-ups later” / istolethetv | https://www.flickr.com/photos/istolethetv/3484451164 | Flickr page resolves to the matching title/author and displays “Some rights reserved”; Wikimedia Commons has a record for this exact Flickr item and explicitly verifies CC BY 2.0. The DMS per-image metadata also declares CC BY 2.0. Credit: istolethetv; title; source landing page; CC BY 2.0; note any resize/crop/preprocessing. No image downloaded. |
| `Multiple materials` (taxonomy ID 55) | `22563` / `1664c8fb02fb195b` | “its not really a ‘couch’ persay” / Chris Connelly | https://www.flickr.com/photos/c_conn/2690225703/ | DMS per-image metadata declares CC BY 2.0, author/title, and the exact original URL. An online educational image-attribution catalog independently lists this exact photo as Chris Connelly, CC-BY, and links to the same Flickr item. The Flickr landing page itself was inaccessible to the browser research tool during this check, so current license status is not directly confirmed there. Do not acquire pixels until a direct landing-page/API license check succeeds. |

Evidence sources: [matching Flickr page and title](https://www.flickr.com/photos/istolethetv/3484451164), [Wikimedia Commons license record for that Flickr item](https://commons.wikimedia.org/wiki/File:27_push-ups_later_(3484451164).jpg), and [attribution catalog entry for the ambiguous candidate](https://courses.lumenlearning.com/wm-spanish1/chapter/additional-attributions/). A “Some rights reserved” label by itself does not establish the exact license. DMS metadata plus secondary attribution is only a shortlist screen, not a substitute for the required direct per-image verification.

## Current gates and next safe action

- **Source provenance:** pass for DMS archive hashes/revision/license; Open Images metadata schema and per-image-check requirement verified.
- **Unknown/ambiguous label availability:** pass for existence in polygon annotations and raw-ID mapping; exact observed support above.
- **Image-license shortlist:** one candidate has direct corroboration via its item-specific Commons license record; one candidate still requires direct Flickr/API license confirmation. No pixels were fetched.
- **Held-out fixture readiness:** fail/not evidenced. Actual images and mask PNGs are absent from the fixture runtime cache; DMS annotations have no instance-level object IDs; five distinct object instances/class, object-disjoint split, preprocessing hashes, and the frozen region counts cannot yet be asserted.
- **Next safe CPU work:** resume per-image landing-page/license verification for candidate URLs, then acquire only verified image bytes into ignored runtime cache, validate against the published Open Images MD5/size metadata, reproduce the pinned DMS resize/orientation preprocessing, and manually establish object instances and a disjoint held-out manifest. Do not call the frozen classifier metric gates evaluable until actual image/mask fixtures exist.

## Alternate held-out evaluation route screen (2026-09-25)

No third-party image/annotation set found in this screen is currently ready to
replace DMS for the frozen classifier fixture. No files or pixels were
downloaded for this screen.

| Source | Published terms / availability | Label and split fit | Disposition |
|---|---|---|---|
| [MatSeg dataset](https://huggingface.co/datasets/FlyingFrog/MatSeg_Zero-Shot_Material_Segmentation_Dataset) and [publisher repository](https://github.com/sagieppel/MatSeg-Dataset) | Hugging Face declares CC0-1.0 and lists a 52.2 GB total dataset. Publisher README offers Zenodo and two alternate archives. The real-world benchmark is described as 820 images in the publisher README and 1,220 on the current HF card; this count discrepancy must be resolved against one immutable artifact manifest before use. | The publisher describes class-agnostic material/state segmentation, point-group and similarity annotations, including wet/dry, corrosion, spoilage, rocks, soils, liquids, etc. It does not publish the five required material-class labels (`Rubber/latex`, `Glass`, `Plastic, clear`, `Paint/plaster/enamel`, `Metal`) or object-instance IDs/disjoint split metadata needed by this classifier gate. | Not suitable for frozen class-identity scoring as-is. Its broad state/partial-similarity data might support separate exploratory unknown/ambiguity cases after reviewing the actual archive and its underlying image provenance; it cannot supply the required five-class supports based on published labels. No archive inspection or acquisition authorized/performed. |
| [Kyoto Local Materials Database](https://vision.ist.i.kyoto-u.ac.jp/codeanddata/localmatdb/) | Official lab page links a Dropbox archive but publishes no explicit dataset/image license on that page. Images come from PASCAL VOC, MS COCO, and ImageNet, whose source terms differ. | It provides pixel masks for Glass, Metal, Plaster, Plastic, and Rubber among 16 categories, but does not specify the required clear-plastic / latex subtypes, bare-vs-painted-metal truth cohorts, per-object support counts, or an object-disjoint evaluation split. | Candidate for further source audit only. License, immutable archive identity, source-image permissions, taxonomy reconciliation, and support/disjoint-object counts remain unverified. |
| [MINC](https://opensurfaces.cs.cornell.edu/publications/minc/) | The official page licenses annotations CC BY 4.0; photos have their own licenses. Full-resolution images are offered for non-commercial research only through a generated terms form. A MINC-2500 patch archive is listed separately, but the page does not establish that it contains the original five-class segmentation/identity fields needed for this test. | The published full dataset has 7,061 material segmentations in 23 categories; original-resolution images are needed for that resource. No source object-disjoint split or required five-class subgroup counts were established in this screen. | Not a cleared product evaluation source: access terms, photo-level rights, exact subset availability, and class/object support require further review. The non-commercial image condition also conflicts with an unqualified public/product evaluation route. |
| [OpenSurfaces extra annotations](https://www.robots.ox.ac.uk/~vgg/data/wildtex/) | Official VGG page publishes a downloadable tarball with images, masks, annotations, and splits, selected from OpenSurfaces. The OpenSurfaces paper says its image collection was limited to Creative Commons photos allowing sharing/remixing, but that does not establish the license/attribution for each proposed record. | Material segmentation and splits exist, but the required exact taxonomy/subtype and five-instance/class object-disjoint counts were not inspected. No individual photo IDs/attribution were verified. | Possible source for another per-record license audit, not a presently ready immutable held-out fixture. |
| [Describable Textures Dataset](https://www.robots.ox.ac.uk/~vgg/data/dtd/) | Official page offers a 625 MB image archive and a 1.4 MB label/split archive; published use is for research, without an explicit SPDX/CC license on the page. Images were collected from Google and Flickr. | 5,640 images are labeled by 47 perceptual texture attributes, not the required material taxonomy; it contains no object-instance IDs or material-region masks. | Not suitable for this gate. |

### Spec-permitted self-authored rendered fixture

The audited Ticket07 gate requires at least 20 **independently rendered**
regions/class, at least five distinct object instances/class, object-disjoint
development and held-out sets, plus fixed unknown and ambiguous cohorts. The
audited spec explicitly permits rendered multi-view crops/features for
material classification and includes synthetic golden fixtures. A
self-authored, physically shaded CPU fixture therefore provides a compliant
candidate-quality route without relaxing any numeric gate; synthetic metrics
remain distinct from real-image performance.

The fixture has now been implemented in
[`fixtures/render_fixture.py`](fixtures/render_fixture.py), generated into the
ignored project cache `.modly-amd-runtime/material-identity-fixture-v1/`, and
verified by [`fixtures/RENDERED_FIXTURE.md`](fixtures/RENDERED_FIXTURE.md).
Its RGB views use project-owned deterministic ellipsoid rasterization and
GGX/dielectric shading; all mesh, image, mask, face-map, input-manifest, and
truth-manifest bytes are hashed. It contains 145 unique objects and 580
rendered region views:

- Five supported classes have five development objects/class and 20
  object-disjoint held-out objects/class, with four separate calibrated
  camera/light renders per object: 80 regions/class held out.
- Unknown/out-of-taxonomy has five development and five held-out objects,
  four views/object: 20 regions/split. The five-object cohort contains two
  wood objects and one each ceramic, paper, and stone.
- Ambiguous multi-material has five development and five held-out objects,
  four views/object: 20 regions/split. Each single topology-bound region
  crosses separate rendered rubber and paint surfaces.
- Metal has bare and painted truth subcohorts; metrics score only generic
  `Metal`. Input features contain no class, split, cohort, subtype, label, or
  recipe IDs. The classifier sees opaque IDs plus image/mask/face/topology and
  camera evidence; split and truth are separate manifests.

The all-region held-out selective-coverage denominator is **440** examples:
400 supported plus 20 unknown plus 20 ambiguous. At least 352 must receive a
single accepted label for the unchanged >=80% coverage threshold. The frozen
>=90% unknown and >=90% ambiguous correct-abstention checks use their separate
20-example cohorts; they are not excluded from the coverage denominator.
Development has separate objects and the same 20 unknown and 20 ambiguous
region-view supports for calibration/policy work.

Exact generated digest evidence, renderer/runtime pins, and reproduction
command are recorded in `fixtures/RENDERED_FIXTURE.md`. The focused suite
passes 6/6 in 50.469 seconds, including two full generations with identical
input/render/mask/face/mesh/manifest digests and object/split/taxonomy checks.
This fixture is ready for an evaluator, but it has not been scored by SigLIP2.
No quality or coverage metric, model acceptance, RX 7900 GRE support,
MIGraphX parity, latency, or VRAM result is claimed. The fixture is a
controlled ellipsoid-render domain, not a claim of real-world generalization.
