# Native 3D part segmenter selection gates

This document was recorded before any target-GPU comparison for Ticket 04. These gates are fixed for this ticket; no threshold may be relaxed after candidate results are seen.

## Candidate set and rubric

1. **P3-SAM** (`Tencent-Hunyuan/Hunyuan3D-Part`, P3-SAM): required first candidate because the audited specification names it as the initial native-3D segmentation candidate and upstream states that it accepts arbitrary meshes.
2. **PartField** (`nv-tlabs/PartField`): alternate because upstream describes feed-forward feature-field inference on mesh inputs followed by hierarchical face clustering; it is the strongest currently inspectable alternate with a released inference path and checkpoint.
3. **SAMPart3D** (`Pointcept/SAMPart3D`) is screened out before target execution as a comparison candidate because its official install guide requires CUDA, PointOps, spconv, tiny-cuda-nn, NVIDIA RAPIDS/cuML and documents a 24 GB RTX 4090 target. PartSAM is likewise screened out because its official README says inference code and pretrained models are TODO/unreleased. These are documented screens, not GPU measurements.

Each executable candidate is evaluated in this order: (a) source, weight, and license identities can be pinned and acquired; (b) required custom/native dependencies have a supported ROCm/HIP path without CUDA/NVIDIA runtime; (c) runs on RX 7900 GRE gfx1100; (d) peak allocated plus reserved VRAM does not exceed the physical 16 GiB; (e) quality and deterministic mapping gates below; (f) end-to-end warm latency. A candidate that fails a preceding gate is not ranked as successful based on a later metric. If candidates both pass, choose highest fixture quality, then lower peak VRAM, then lower median warm latency. Preserve the capability contract independent of implementation.

## Frozen fixture and correctness thresholds

The fixture is a generated, watertight two-component mesh with 1,536 known face labels (two disconnected, three-times-subdivided box shells with 768 faces each), stored with exact source bytes and a topology digest in the accompanying probe evidence. It tests two native geometric parts; it does not assert semantic names. Prediction region IDs are matched to truth IDs by maximum-weight bipartite assignment over face intersections. The primary score is face-level macro IoU over the two truth parts.

Acceptance requires all of the following:

- face-level macro IoU **at least 0.90**;
- every input face receives exactly one label (coverage **1.00**);
- overlap policy is **disjoint partition**: one region per face; overlaps and duplicate face assignment are rejected;
- repeated execution with identical mesh bytes, adapter revision, weights, parameters, and seed yields identical per-face labels;
- all mapping indices are in range and bound to the exact Structured Asset topology revision;
- target execution uses no CUDA runtime or NVIDIA-only binary dependency, and peak GPU allocation plus reservation is **at most 16 GiB**;
- no confidence number is fabricated: absent candidate-native calibrated confidence is represented as uncalibrated/unknown according to its source.

The 0.90 threshold is intentionally exacting for this deliberately separable fixture. It is a smoke gate for topology extraction and part grouping, not a substitute for broad benchmark generalization.

## Primary-source dependency notes (reviewed 2026-09-24)

- P3-SAM upstream describes arbitrary mesh input and automated native-3D part masks. Its setup README documents Python 3.10, PyTorch 2.4.0+CUDA 12.1, Sonata dependencies, and building `utils/chamfer3D`; the demo obtains `p3sam.safetensors` from Hugging Face. These are declared requirements to verify on hardware, not proof of AMD incompatibility. Sources: [official P3-SAM README](https://github.com/Tencent-Hunyuan/Hunyuan3D-Part/blob/main/P3-SAM/README.md), [official project README](https://github.com/Tencent-Hunyuan/Hunyuan3D-Part).
- PartField upstream describes feed-forward part feature fields, downloadable Objaverse checkpoint, mesh feature inference, and hierarchical agglomerative clustering. Its published setup pins PyTorch 2.4.0 CUDA 12.4 and a CUDA-specific torch-scatter wheel, so AMD support must be demonstrated rather than assumed. Source: [official PartField README](https://github.com/nv-tlabs/PartField).
- SAMPart3D upstream documents a 24 GB RTX 4090, CUDA 12.1, custom PointOps, spconv, tiny-cuda-nn, and RAPIDS/cuML installation. Source: [official SAMPart3D install guide](https://github.com/Pointcept/SAMPart3D/blob/main/INSTALL.md).
- PartSAM upstream currently lists inference code and pretrained models as unreleased TODOs. Source: [official PartSAM README](https://github.com/czvvd/PartSAM).
- The Modly P3-SAM extension currently documents NVIDIA CUDA-capable hardware and a 24 GB target; it is an adapter and does not change upstream dependencies. Source: [Modly P3-SAM extension README](https://github.com/DrHepa/Hunyuan3D-Part-modly-extension).

## Immutable model identities and target probe envelope

The initial P3-SAM candidate is pinned to upstream source commit `e96be065375438962375b55326416291342958a7`. Its P3-SAM head weights are Hugging Face revision `ee23ed4fb349b9144a061297acff7122927faf5d`, file `p3sam/p3sam.safetensors`, 450,968,044 bytes, SHA-256 `eb76550cfbe06f154c6e9b17167ccfc28222bb4a216ec7b12ac2bf7d762de38c`. Its separate Sonata backbone weight is revision `0e6f990a2f7bfdffc9d2f58139cfa2b883d8e8cb`, file `sonata.pth`, 434,008,287 bytes, SHA-256 `c5ced5acdae30d1c469713398073a866e25e6e414e23feed5dc025373657ac50`. The two downloaded checkpoints total 884,976,331 bytes before framework and extraction overhead. The official P3-SAM runner defaults to 100,000 sampled points, 400 prompts, and prompt batches of 32; a first target run must record actual peak VRAM before these defaults can be retained. No VRAM estimate from tensor arithmetic substitutes for the device measurement.

The alternate PartField model is pinned to source commit `373025dbd283bb44cc4a6dc78c99994dbc91de32` and weight repository `mikaelaangel/partfield-ckpt`, model revision `f8cda8fd7dcef0596654015a482cc89407977a29`, file `model_objaverse.ckpt`, 1,243,631,761 bytes, SHA-256 `463efc8a3afd3913142aa025e0125c00f16ef452b8de6a132ebe32bbe7877ee4`. The Hugging Face file scanner detects Python pickle imports in this checkpoint; treat it as executable and load only from the verified project-managed cache after its provenance review. Its upstream installation uses PyTorch 2.4/CUDA 12.4 and a CUDA-specific torch-scatter wheel, so any ROCm success must be measured with a clean dependency report. The published PartField license restricts third-party use to noncommercial research/education; P3-SAM code and weights use Tencent's community license. Target/runtime evidence does not override those use terms.

The bounded target probe uses only the 1,536-face known-truth GLB fixture, seed 42, one candidate at a time, one loaded model at a time, and records exact source/weight hashes, ROCm/PyTorch versions, device name, native libraries, warm median latency, allocated and reserved VRAM peaks, per-face mask digest, face coverage, overlap count, and best-match macro IoU. It runs first with the explicitly documented low-memory point/prompt batch parameters so the 16 GiB device is protected; only after that run succeeds under 16 GiB may a default-settings run be considered. Every tested mode must independently pass all frozen quality gates. The other candidate is not loaded until the first model process exits and GPU memory returns to its pre-run baseline. Any source/weight download occurs within the project-owned adapter cache, and actual transfer bytes and SHA-256 are recorded before loading.

P3-SAM's dense feed-forward `mlp`, stage-1 and stage-2 segmentation MLPs, and IoU MLPs are individually qualified with Modly AMD Runtime, which then routes each call through the selected MIGraphX or PyTorch ROCm callable. Sonata's sparse/native point-processing operations remain on their installed framework dependency path and are included in stage VRAM/latency measurements; no ROCm support is inferred from upstream CUDA instructions. No CUDA-specific custom operator is treated as AMD-compatible until the target environment proves it.

Any change to candidate scope or these gates requires a written reason that does not lower an acceptance threshold.
