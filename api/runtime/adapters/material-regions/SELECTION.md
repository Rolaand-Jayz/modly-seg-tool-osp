# Material-region candidate evaluation

Evaluation basis: `FINAL_AUDITED_SPEC.md` §§5, 7, 11, 15, 16, 25, 26–28,
and Ticket 06. This is a source-level/runtime-contract evaluation; no third-party
weights were downloaded and the shared RX 7900 GRE was not used for this work.

## Declared quality gates

The acceptance fixture is one glTF mesh/node with four adjacent triangles, a
single semantic part covering all four faces, and two ground-truth visible
material regions. It has no glTF material slots or PBR assertions. Two
calibrated 16×16 views provide independently generated 2D material-label masks
and camera projection matrices. The front/back views must agree after
projection. The implementation must achieve face-level mean IoU
≥0.90, face coverage ≥0.98, boundary F1 ≥0.85, and zero invalid/stale faces in
valid mappings. On this deliberately unambiguous fixture, the stricter expected
result is IoU=1.0, coverage=1.0, boundary F1=1.0. No material-region output may
depend on part IDs or glTF material/PBR slots. Unobserved faces must remain
unmapped and report unknown confidence rather than being filled by assumption.

The strategy rubric is conjunctive: (1) AMD-only operation, no CUDA/NVIDIA
dependency; (2) measured peak VRAM ≤14 GiB on the 16 GiB target, or a measured
CPU path; (3) each emitted mapping is bound to the input topology revision;
(4) declared quality gates pass on the same fixture; (5) provenance retains
view and upstream segmenter identities; and (6) integration does not replace
Modly's process-extension/workflow seam. A candidate without target runtime or
weight evidence cannot be described as hardware-qualified.

## Candidate A: MaterialSeg3D

The candidate is the upstream [MaterialSeg3D project](https://github.com/PROPHETE-pro/MaterialSeg3D),
described in its [paper](https://arxiv.org/abs/2404.13923) and
[project page](https://materialseg3d.github.io/). It is the dedicated material
segmentation candidate for 3D assets: it renders mesh views, predicts dense
2D material labels using its MIO-trained prior, projects them into UV space,
then fuses votes and unifies regions. Its output is suitable for generating
material/PBR maps, so its region-label stage must be separated from downstream
PBR assignment in Modly.

The upstream installation guide specifies Python 3.9.15, PyTorch 1.12.1 with
CUDA 11.3, PyTorch3D, xformers, NVIDIA CUB, nvdiffrast, GET3D/Text2Tex, a
Blender 2.90 Python environment, and separately downloaded model weights. The
authors report tests on A30/A100. These are concrete target incompatibilities
and missing immutable model-weight evidence, not evidence that the algorithm
cannot be ported. An AMD port would need a newly qualified 2D segmentation
checkpoint/runtime, replacement or validation of the CUDA-only rendering and
mesh operations, and adaptation from its OBJ/albedo-UV/category-specific input
to Modly's topology-bound sidecar contract. No upstream RX 7900 GRE/ROCm/MIGraphX
measurement or matching public weight digest is available from the inspected
setup documentation. Its target VRAM use therefore remains unmeasured and does
not pass the hardware qualification gate. This candidate is not selected as
the POC adapter.

## Candidate B: calibrated multi-view label projection and backprojection

This strategy takes per-view 2D material-label masks from a replaceable image
segmenter, camera clip matrices, and the existing GLB/glTF mesh. The 2D mask
producer is an explicit upstream dependency; this node returns an actionable
error when calibrated masks are not provided. It rasterizes
the source triangles into each calibrated view, associates visible pixels with
face IDs using a z-buffer, accumulates per-face label votes, and forms connected
face components from label and mesh-edge adjacency. The algorithm is CPU-only
and deterministic; it does not read `part_segments`, material indices, texture
slots, or PBR assertions. The established projection/fusion family is described
in MaterialSeg3D itself and independently in multi-view mesh projection work;
the implementation here keeps only Modly's explicit 2D evidence→topology
mapping and weighted-vote part of that strategy.

This strategy was run against the declared synthetic fixture using two views.
It produced exact ground truth (face mIoU 1.0, coverage 1.0, boundary F1 1.0),
two connected material regions within one geometric object, and valid mappings
on the current topology revision. The runtime performs no GPU allocation, so
its measured accelerator VRAM use is 0 bytes; CPU wall time and process memory
are recorded by the acceptance test. This does not qualify an upstream 2D
segmenter, and real-scene segmentation quality is not inferred from the
synthetic contract fixture.

## Rubric result

| Candidate | AMD / 16 GiB gate | Same-fixture mapping quality | Modly integration cost | Decision |
|---|---|---|---|---|
| MaterialSeg3D | Fails documented AMD-runtime gate; 16 GiB peak is unmeasured, so it cannot be accepted | Not measured: its required CUDA stack/checkpoint was neither installed nor executed | High: replace or port CUDA-oriented rendering/mesh dependencies, qualify an immutable weight file, adapt OBJ/albedo-UV/category inputs, then bind results to current topology | Reject for this POC; reconsider after AMD and VRAM proof |
| Multi-view projection/backprojection | Passes CPU-only policy; zero accelerator allocation, so no GPU-memory residency | Passes: mIoU 1.0, coverage 1.0, boundary F1 1.0 on the declared fixture | Low: standalone Modly process extension over current Structured Asset and headless JSON-lines seam; calibrated masks are an explicit input | Select as reference region-mapping/fusion adapter |

The candidates' quality values are not conflated: MaterialSeg3D did not run on
the common fixture, so no relative model-quality claim is made. Its rejection
is caused by documented runtime incompatibility and missing target-resource
evidence. Candidate B's fixture score qualifies only the topology mapping and
multi-view vote stage, not the upstream 2D segmenter or open-world accuracy.

## Decision

Select Candidate B for the Modly reference adapter because it passes the AMD,
16 GiB, fixture-quality, mapping, and integration gates without importing an
unqualified CUDA-bound model stack. MaterialSeg3D remains a documented future
candidate only after a separate AMD-compatible inference/weights and VRAM
qualification. View label inputs carry their own segmenter identity and
observation reference; Modly's adapter owns geometry projection, consensus,
topology mappings, confidence state, and invalidation. Material identity and
PBR remain independent downstream assertions. Ticket 06 validates this
multi-view mapping/fusion stage from supplied 2D masks; it does not claim raw
images have already been converted into material-label masks.

## Reproduction

Run in the project API Python 3.12 environment:

```sh
PYTHONPATH=api python -m unittest discover -s api/tests \
  -p test_ticket06_material_regions.py -v
```

The test builds a real GLB in a temporary workspace and invokes the Modly
process-extension entry over its JSON-lines protocol. It verifies two-view
projection, mapping, evidence, topology invalidation, input validation, and the
quality gates above. The stage artifact retains the exact view masks and camera
matrices under digest identity. Adapter-process wall time, peak host RSS, and
zero accelerator allocation are reported. No model weights or GPU are required
for this CPU adapter. Inputs are bounded to 64 views, 2 million total label
pixels, 2048 pixels per dimension, 128 characters per label, and 250,000 faces
per asset. Upstream segmenter revisions must be immutable commit/digest
identities or explicit `builtin:x.y.z` versions.
