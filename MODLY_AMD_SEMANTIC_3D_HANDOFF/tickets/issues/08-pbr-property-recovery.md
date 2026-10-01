# 08 — PBR property recovery with synthetic validation

**What to build:** Estimate supported physically based rendering properties for material regions through `Estimate PBR Properties`, probing Material Anything first and selecting a replacement if its AMD/resource/dependency cost fails the target gates. Preserve only properties the accepted adapter actually produces, and validate recovery against synthetic ground truth plus novel-light rendering.

**Blocked by:** 01 — Structured Asset headless round-trip; 02 — AMD Runtime proof through Modly; 06 — Material-region segmentation independent of parts.

**Status:** in progress; candidate acceptance blocked

### MatMart official-source screen (2026-09-28)

The CVPR 2026 paper describes estimating albedo, roughness, and metallic from
RGB views plus known geometry, including completion of unseen UV areas. The
official publication trail reviewed here does not establish an
author-linked implementation, deployable inference package, pinned weights,
applicable license terms, preservation of caller topology/material-region
IDs, or AMD/RX 7900 GRE 16 GB operation. Frozen channel and novel-light quality
evidence is also absent. Rejected before acquisition/probing; no code, weights,
data, fixture, or truth were accessed and no gates changed. Evidence:
[`TICKET08_MATMART_OFFICIAL_SOURCE_SCREEN_2026-09-28.md`](../../../api/runtime/adapters/pbr/evidence/TICKET08_MATMART_OFFICIAL_SOURCE_SCREEN_2026-09-28.md).

### MatLat official-source screen (2026-09-28)

MatLat accepts a mesh and generates base-color, roughness, and metallic maps,
but is not eligible for acquisition/evaluation: released weights are CC BY-NC
4.0; the documented inference route uses nvdiffrast/CV-CUDA and has no
RX 7900 GRE ROCm or <=14 GiB evidence; and preservation of the caller's
topology, UV mapping, and material-region IDs is unestablished. Immutable code,
weight, and complete component identities are also incomplete. No code,
packages, weights, fixture, observations, or truth were acquired or run. No
acceptance gate changed. Details and direct official sources:
[`ticket08-matlat-official-source-screen-2026-09-28.md`](../../../api/runtime/adapters/pbr/evidence/ticket08-matlat-official-source-screen-2026-09-28.md).

### Fresh primary-source discovery (2026-09-28)

A bounded search found no new Ticket 08 candidate. The official 2026
MaterialSeg3D++ article confirms that its 30 material classes each map to a
separately assigned roughness/metallic pair; it remains a class-to-preset
workflow rather than independent continuous PBR estimation. M-XR I2M and
LumiTex were also surfaced but already have source screens, so no duplicate
screen was added. No assets, model weights, packages, fixture, or truth were
accessed; no quality or AMD evidence was generated and no gates changed.
Details and direct official sources:
[`ticket08-fresh-primary-source-discovery-2026-09-28.md`](../../../api/runtime/adapters/pbr/evidence/ticket08-fresh-primary-source-discovery-2026-09-28.md).

### MaterialMVP topology/runtime recheck (2026-09-28)

MaterialMVP had already been identified as a direct mesh-input lead; this
source recheck adds a concrete topology blocker. Its official pipeline defaults
to remeshing, then wraps UVs and saves a downsampled mesh with generated PBR
maps. The `use_remesh=False` branch does not establish that UV wrapping,
baking, and export preserve the caller's original faces or Modly material
region IDs. Its CUDA 12.4 PyTorch/custom-renderer stack, unresolved complete
model/component identities and terms, and missing frozen channel/novel-light
quality evidence also fail the gates before evaluation. No download, execution,
fixture, or truth access occurred. This is a recheck of the existing candidate,
not a newly selected model. Sources and prior evidence:
[`MaterialMVP official pipeline`](https://github.com/ZebinHe/MaterialMVP/blob/main/textureGenPipeline.py),
[`MaterialMVP source screen`](../../../api/runtime/adapters/pbr/evidence/additional-candidate-screen-2026-09-25.md).

### LightSwitch official-source screen (2026-09-28)

LightSwitch is a multi-view, material-guided relighting route, but its
documented outputs are relit views and a Gaussian-splat appearance model, not
separate PBR maps attached to the caller's unchanged mesh/topology/material
regions. Its MIT code license does not establish complete model-weight terms
and identities. The documented A100 setup provides no ROCm/RX 7900 GRE or
<=14 GiB evidence; frozen channel/novel-light quality and Modly provenance are
also unproven. Rejected before acquisition or evaluation; no model, fixture,
or truth was accessed and no gates changed. Details:
[`ticket08-lightswitch-official-source-screen-2026-09-28.md`](../../../api/runtime/adapters/pbr/evidence/ticket08-lightswitch-official-source-screen-2026-09-28.md).

### Additional image-space model lead: MatPredict (2026-09-28)

The official MatPredict README describes albedo, roughness, and metallic image outputs, but says its dataset will be released after review and does not identify released pretrained weights. It also does not define how to attach image-space predictions to Modly's supplied mesh, current topology revision, or material-region IDs. Rights, an AMD route, provenance/confidence behavior, frozen channel metrics, and novel-light quality remain unresolved. The candidate is rejected before acquisition or evaluation; no assets or fixture/truth data were accessed and no gate changed. Source-backed findings: [`ticket08-matpredict-official-source-screen-2026-09-28.md`](../../../api/runtime/adapters/pbr/evidence/ticket08-matpredict-official-source-screen-2026-09-28.md).

- [ ] Before final comparison, claimed-channel correctness/quality thresholds and the selection rubric are declared; Material Anything is then probed for AMD execution, Blender/PyTorch3D/native dependencies, peak VRAM, latency, supported output channels, and quality, with another estimator allowed if it wins the declared gate without changing the capability contract.
- [ ] The accepted adapter writes only supported PBR assertions/maps (for example base color/albedo, roughness, metallic, bump/normal/detail) and leaves unsupported channels unknown rather than fabricating them.
- [ ] Bump, normal, and other detail representations are identified correctly instead of being mislabeled as interchangeable channels.
- [ ] Each PBR assertion/map records evidence kind, confidence state where meaningful, adapter/weights/runtime provenance, source observations, parameters, and topology/material-region target.
- [ ] Synthetic fixtures with known PBR ground truth measure each claimed channel with declared task-appropriate numeric metrics/tolerances and expose lighting baked into albedo or other obvious decomposition failures.
- [ ] A novel-light render comparison is produced for at least one fixture so recovered properties are tested outside the estimator's input illumination.
- [ ] The accepted path fits the 16 GB target or uses an explicit supported low-memory mode; heavy Blender/preprocessing requirements remain isolated inside the adapter rather than becoming Modly viewer/runtime dependencies.

## Current evidence (2026-09-25)

### Material Anything target-operation feasibility recheck (2026-09-27)

The exact project target image is PyTorch 2.11.0+rocm7.14.0 / HIP 7.14.60850. Material Anything's active mesh path needs PyTorch3D OBJ/GLB handling, UV textures, mesh rasterization and face-attribute interpolation, plus Kaolin indexing and `kal.render.mesh.rasterize`; the current image lacks both libraries. Kaolin's official documentation describes its complete feature set as CUDA/NVIDIA-only and its published wheel matrix does not cover this ROCm/PyTorch target. PyTorch3D's official CI has a ROCm 7.2.3/PyTorch 2.11 build-only job, but it does not test rasterization and does not qualify ROCm 7.14 or the RX 7900 GRE. This makes the current image path unavailable and Kaolin the main compatibility blocker, but does not prove an isolated PyTorch3D build impossible. No weights were acquired and no source or packages were changed. If Material Anything remains under consideration after the active Ticket 04 GPU run, next test a pinned PyTorch3D build in a separate project-local overlay on a tiny synthetic mesh; a separate AMD-compatible replacement or explicit rejection is still required for Kaolin rasterization. Sources: [Kaolin installation notes](https://github.com/NVIDIAGameWorks/kaolin/blob/master/docs/notes/installation.rst), [PyTorch3D installation notes](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md), and [PyTorch3D build workflow](https://github.com/facebookresearch/pytorch3d/blob/main/.github/workflows/build.yml).

### SuperMat base-model release and terms screen (2026-09-27)

SuperMat is not eligible for acquisition or inference on current evidence. The pinned SuperMat code snapshot is MIT and publisher metadata labels the SuperMat weights MIT, but the required SD 2.1 base is an unaffiliated mirror that claims CreativeML Open RAIL++-M and links the Stability license endpoint, which returned HTTP 401. The mirror card also describes direct use as research-purpose; commercial deployment and redistribution scope for Modly is not established. The mirror's principal Diffusers component hashes are available from prior metadata but the complete consumed tree is not yet hashed. No model/code/weights/package/fixture/truth were downloaded or run. Required clearance and full tree-lock conditions are recorded in `api/runtime/adapters/pbr/evidence/SUPERMAT_LICENSE_AND_RELEASE_SOURCE_GATE_2026-09-27.md`. This screen does not change the PBR contract, candidate selection, or acceptance gates; SuperMat remains unselected.

### PBRnxt official-source screen (2026-09-27)

PBRnxt is a distinct 2D texture enhancement model, but is not eligible for
Ticket 08 evaluation: it takes one existing diffuse texture, omits metallic,
and provides no Modly topology/material-region correspondence. The inspected
source pins its code revision and MIT license but not the checkpoint digest or
weight-specific terms; its documented environment is CUDA-oriented, with no
ROCm/RX 7900 GRE or <=14 GiB evidence. Normal and displacement are kept
distinct and were not treated as bump/height. No artifacts, packages, weights,
Modly assets, fixture, or truth were downloaded or accessed; no inference or
scoring was done and no gate changed. Evidence:
`api/runtime/adapters/pbr/evidence/ticket08-pbrnxt-official-source-screen-2026-09-27.md`.

### Spec-Gloss Surfels official-source screen (2026-09-28)

Spec-Gloss Surfels is a relevant multi-view relighting method, but it reconstructs a 2D Gaussian-surfel scene rather than preserving a caller mesh/topology/material-region contract. It documents albedo, roughness, and F0, not an explicit metallic map; novel-light relighting claims do not supply the frozen per-channel or held-out-light score. Its official environment uses NVIDIA packaging, `nvdiffrast`, and custom rasterization/ray-tracing submodules, with no AMD/RX 7900 GRE qualification. Although code is MIT, complete identities/terms for linked StableNormal and StableDelight prior models and a pinned executable component set are not established. Rejected before acquisition/evaluation. No source artifacts, weights, datasets, fixtures, truth, or GPU were accessed; gates and thresholds remain unchanged. Evidence: `api/runtime/adapters/pbr/evidence/ticket08-specgloss-surfels-official-source-screen-2026-09-28.md`.

- The DualMat official-source screen found a relevant multi-view albedo/metallic/roughness research lead, but no accessible official source repository, immutable code/weight identities, complete compatible license chain, AMD/resource qualification, or caller-topology/region binding. It was rejected before acquisition or evaluation; no assets or fixture/truth data were accessed. Details: `api/runtime/adapters/pbr/evidence/ticket08-dualmat-official-source-screen-2026-09-25.md`.

- The candidate-independent frozen fixture/scorer suite passes 6/6 with
  `PYTHONPATH=api .modly-amd-runtime/api-test-venv/bin/python -m unittest
  discover -s api/tests -p 'test_ticket08_*.py' -v`. This covers fixture
  determinism/input-truth isolation, numeric channel metrics, and the
  held-out-light renderer. It does not establish model quality or acceptance.
- Material Anything remains dependency-blocked in the pinned target image:
  PyTorch3D and Kaolin are absent. Its pinned active mesh path imports
  `TexturesUV`, calls Kaolin mesh indexing and rasterization, and includes a
  CUDA-device allocation. The fixed multi-gigabyte weights have not been
  fetched; see `api/runtime/adapters/pbr/evidence/material-anything-cpu-source-screen.md`.
- SuperMat remains an unqualified alternative. Target-image CPU dependency
  evidence records `xformers` and `einops` absent from the base image. A
  hash-recorded `einops==0.8.1` wheel exists in the TripoSR cache and passed a
  local networkless zipimport NumPy smoke; no xformers package is staged. A
  source-hash-locked overlay moves the eager import to the xFormers-only
  processor call. In the networkless no-device project image, the full
  overlaid attention module import remains blocked at missing Diffusers. The
  actual pinned PyTorch SDPA processor class body was isolated from the
  verified AST and matched an independent CPU reference exactly (max error
  0.0; input/output `[4,5,8]`; peak RSS 763,604,992 B; elapsed 481.59 ms).
  This does not establish Diffusers integration. Multi-view inference writes
  separate per-view maps; its UV refine script requires prebuilt UV maps,
  positions, and masks and does not implement their projection/fusion from
  views. No weights were fetched. See
  `api/runtime/adapters/pbr/evidence/supermat-cpu-source-screen.md` and
  `api/runtime/adapters/pbr/evidence/supermat-attention-cpu-probe-result.json`.
- Neither candidate has a model output, AMD execution, MIGraphX parity,
  channel quality, novel-light, latency, or VRAM result. Keep all candidate
  and target acceptance criteria open; the thresholds in
  `api/runtime/adapters/pbr/SELECTION.md` are unchanged.
- Combined fixture/scorer/overlay/UV projection suite passes 16/16 with
  `PYTHONPATH=api .modly-amd-runtime/api-test-venv/bin/python -m unittest
  discover -s api/tests -p 'test_ticket08_*.py' -v`.
- A topology-bound multi-view projection/fusion component is implemented in
  `api/runtime/adapters/pbr/view_projection.py`. It consumes per-pixel face
  IDs and barycentrics against canonical face UVs, binds observations to a
  topology revision, weights observed samples, and preserves texels without
  evidence as unknown. Its focused tests pass 8/8. The frozen PBR fixture
  does not yet include the required per-pixel correspondence inputs, so this
  component is not integrated into candidate inference. A subsequent
  source-interface review corrected the earlier view-to-UV characterization:
  SuperMat's same-view spatial maps can be paired with raster face IDs and
  barycentrics for that calibrated camera and topology revision. That pairing
  is technically plausible but not yet implemented or measured; see
  `api/runtime/adapters/pbr/evidence/supermat-frozen-source-topology-gate-2026-09-25.md`.
- A dependency-ready alternate screen found no stronger license-clear,
  pinned-weight candidate with a topology-ready PBR path. DualMat has no
  actionable official code/checkpoint source in the screen; Material Palette
  is an image-and-mask material workflow and does not bind recovered maps to
  the supplied mesh topology. SuperMat remains the shortest technical route,
  pending review of the pinned Stable Diffusion mirror's OpenRAIL++ terms,
  staging required assets only after that review, and target dependency and
  quality measurements. No alternative was selected and no acceptance gate
  changed.


## Additional candidate screen (2026-09-25)

No newly screened estimator is ready for the frozen target probe. Hunyuan3D-Paint 2.1 has the closest mesh-input seam, but its upstream README reports 21 GB texture-generation VRAM, beyond the 14 GiB PBR gate, and its stack is CUDA-tested. IntrinsiX exceeds the target memory gate and does not provide topology-bound mesh maps. PBR_Boost_3DGen expects an already albedo-textured UV mesh and estimates a narrower set of channels; its published stack uses CUDA/nvdiffrast. TexGaussian advertises mesh-baked maps, but immutable code/checkpoint identity, license, and AMD rasterizer support remain unresolved. Source details and citations are recorded in `api/runtime/adapters/pbr/evidence/supermat-cpu-source-screen.md`. No assets were fetched, no weights loaded, and no acceptance criteria changed.

### Additional released-model screen

A disjoint read-only scan found no additional candidate cleared for evaluation. MaterialMVP is the closest newly identified mesh-input estimator: its official inference interface accepts a mesh and image prompt and its data schema includes albedo and metallic-roughness outputs. Its documented install pins CUDA 12.4 PyTorch, a custom rasterizer, and a compiled differentiable renderer; immutable checkpoint revision/hashes, a license statement in the inspected project README, and bump/normal output are not established. MeshGen is a PBR image-to-3D generation pipeline with a CUDA-specific install and roughly 25.3 GB of published weights, not a material-only path for an arbitrary existing mesh. TRELLIS.2 documents shape-conditioned PBR texturing and base-color/roughness/metallic outputs, but its official requirements specify NVIDIA GPU >=24 GB and CUDA extensions, exceeding the RX 7900 GRE <=14 GiB AMD gate. None was downloaded or executed; no held-out or quality result is claimed. Full source evidence and direct links are recorded in `api/runtime/adapters/pbr/evidence/additional-candidate-screen-2026-09-25.md`. Ticket 08 remains acceptance blocked; thresholds and gates are unchanged.

### Calibrated inverse-rendering candidate (2026-09-25)

A truth-isolated CPU candidate using mesh UVs, training observations/masks, camera transforms, and training lights was measured against the frozen scorer and rejected. It passed 2 focused tests, but failed base-color MAE (0.2593 vs <=0.08), base-color SSIM (0.4865 vs >=0.85), roughness MAE (0.3078 vs <=0.10), and metallic MAE (0.4134 vs <=0.10). It passed metallic bias (0.0160 vs <=0.08) and novel-light image MAE (0.0545 vs <=0.08) using geometric mesh normals; no normal map is claimed. The scorer received truth only after estimation. The candidate was not tuned against held-out evidence and was not run on the RX 7900 GRE because its development map gates failed. Full details, hashes, environment, command, and exact values: `api/runtime/adapters/pbr/evidence/inverse-render-candidate-cpu-rejection-2026-09-25.md`. This does not change any Ticket 08 threshold or acceptance criterion.

A separate primary-source screen found no additional eligible model: PBR3DGen has not released checkpoints or a runnable inference contract, while SF3D emits object-level metallic/roughness scalars rather than topology-bound maps for an existing mesh. See `api/runtime/adapters/pbr/evidence/released-pbr-model-frontier-2026-09-25.md`.

### Fixed geometric-normal inverse-render candidate (2026-09-25)

A separate candidate held face normals fixed from the supplied mesh topology and fit base color, roughness, and metallic using only calibrated training observations/lights. It was frozen before one invocation of the scorer. It failed base-color MAE (0.12363 vs <=0.08), base-color SSIM (0.64076 vs >=0.85), roughness MAE (0.41812 vs <=0.10), metallic MAE (0.31749 vs <=0.10), and conductor/dielectric bias (0.24721 vs <=0.08). Geometric-normal held-out-light MAE passed at 0.03914. The single focused test passed under system Python 3.14.7/NumPy 2.4.2/SciPy 1.18.1; this is candidate-development evidence, not target acceptance. No RX 7900 GRE run followed the hard map-gate failures. Full details and hashes are in `api/runtime/adapters/pbr/evidence/fixed-geometry-inverse-render-cpu-rejection-2026-09-25.md`. Gates remain unchanged.

Project-runtime reproducibility: the fixed-normal candidate was rerun under the project-owned Python 3.12 API test environment using SciPy 1.16.1 installed from a checksum-verified staged wheel into a separate project-local overlay (the API venv was unchanged). Its metrics matched within 0.0002 on every gate. The complete Ticket08 suite passes 19/19 with this overlay; without it, three rejected research-candidate tests are skipped because SciPy is optional and absent. Exact command and metric reproduction: `api/runtime/adapters/pbr/evidence/fixed-geometry-inverse-render-cpu-rejection-2026-09-25.md`.

Material Palette was also screened against the existing topology-bound projection contract. It generates and decomposes new texture patches, which are not pixel-aligned with calibrated source-view face/barycentric maps; the existing projection utility cannot bind those maps to UVs without inventing correspondence. Its decomposition head also lacks metallic. Checkpoint hash/runtime/AMD qualification and component licenses remain unresolved. This candidate was not downloaded or executed. No-go evidence and official source links: `api/runtime/adapters/pbr/evidence/material-palette-topology-gate-2026-09-25.md`.
### Additional official candidate screens (2026-09-25)

### New candidate frontier (2026-09-25)

A disjoint source-only screen evaluated TexGaussian, Seed3D 2.0, and ExMesh++. TexGaussian has mutable source/ambiguous release assets, missing verified checkpoint terms, and a CUDA/nvdiffrast stack; Seed3D is presented as a hosted API rather than a local AMD adapter; ExMesh++ is a research paper without deployable assets and its reconstruction changes topology. None demonstrates the unchanged-topology, observation-conditioned PBR contract or frozen synthetic/novel-light gates. No code, weights, packages, fixture inputs, or truth were accessed. Ticket 08 remains blocked with thresholds unchanged. Exact findings and official sources: `api/runtime/adapters/pbr/evidence/ticket08-new-candidate-frontier-2026-09-25.md`.

SuperMat's source interface can pair same-view PBR maps with existing
face/barycentric render maps and topology revisions. It is not candidate-ready:
the SD2.1 base model terms/intended-use clearance remain unresolved, model
identity is incomplete, albedo color-space semantics are not established, and
no target memory, quality, latency, or novel-light evidence exists. Evidence:
`api/runtime/adapters/pbr/evidence/supermat-frozen-source-topology-gate-2026-09-25.md`.

SfD exports albedo, roughness, and metallic BRDF maps, but rebuilds a canonical
SDF mesh rather than preserving arbitrary input topology and its calibrated
view correspondences. Its source revision, code/checkpoint licensing, and AMD
route are also unresolved. It is not eligible for the current scorer.
Evidence: `api/runtime/adapters/pbr/evidence/sfd-primary-source-gate-screen-2026-09-25.md`.

Neither candidate was downloaded or executed, no held-out truth was accessed,
and no PBR gate changed.

### SuperMat official base-model provenance follow-up (2026-09-25)

The official Stability AI release confirms SD 2.1 publication and its visible
model card identifies CreativeML Open RAIL++-M, but the official model file,
immutable commit listing, and full terms were inaccessible (401) in this
environment; the `bf16` branch reference is mutable. The third-party mirror
was not substituted. The SuperMat model/base snapshot therefore remains
unqualified for acquisition, intended-use terms, and execution. The project
image lacks Diffusers and xformers, and its upstream dependency declarations
are floating. Source/interface feasibility remains distinct from model
qualification. Evidence: `api/runtime/adapters/pbr/evidence/supermat-official-base-provenance-2026-09-25.md`.

### MatSpray topology/AMD screen (2026-09-25)

MatSpray's per-view maps are fused into a relightable Gaussian representation;
the official interface does not bind outputs to an existing mesh's UV/topology
or preserve measured source-view correspondences through conversion. Its
documented stack requires CUDA extensions and OptiX 7.4; no AMD route or RX
7900 GRE evidence exists. Predictor checkpoint identity and rights are also
unresolved. No assets were downloaded. Evidence:
`api/runtime/adapters/pbr/evidence/matspray-topology-amd-gate-screen-2026-09-25.md`.

### Frozen-fixture correspondence integration (2026-09-25)

Added `api/runtime/adapters/pbr/fixture_correspondence.py` to generate deterministic, versioned per-pixel face-ID and barycentric sidecars for all four calibrated training views. The SHA-256 topology revision binds the exact fixture positions, UVs, and indexed faces; embedded sidecar metadata defines schema, dimensions, view count, sentinel, and barycentric ordering. A focused test sends the rendered training RGB views and those correspondences through `project_and_fuse_views`, checking visibility alignment, valid barycentrics, projected coverage, unknown texels, topology binding, and provenance. The original frozen fixture archive and GLB were not changed, and scoring-only material maps were not used as estimator inputs. The complete Ticket 08 suite passes 21/21 with the isolated SciPy 1.16.1 overlay; the same suite without that optional overlay passes 18 tests with 3 research-candidate skips. This closes the fixture-side correspondence integration gap for source-view maps, but does not establish any candidate output, AMD execution, or PBR acceptance.

The generated sidecar is retained at `api/runtime/adapters/pbr/evidence/ticket08-three-region-pbr-correspondence-v1.npz` with manifest `ticket08-three-region-pbr-correspondence-v1.json`. Its SHA-256 is `cfbfc146f65451137da58887ef969ec3c06b0fef4c893be29486cfd392933474`, size 463,511 bytes, and topology revision `sha256:77922db4079e22f05d6649baa6483218559ca8ccb18ab16c9d51b32ff03315c2`.

### SuperMat official artifact and license refresh (2026-09-25)

A source-only refresh established an immutable official SD 2.1 single-file checkpoint at repository revision `0b4891198ec223dff1b3989f17537695fb1bdbe8`, with upstream LFS SHA-256 `ad2a33c361c1f593c4a1fb32ea81afce2b5bb7d1983c6b94793a26a3b54b08a0` and size 5,214,865,159 bytes. This `.ckpt` does not establish the complete immutable Diffusers component snapshot used by SuperMat's `DiffusionPipeline.from_pretrained` path. The official model card exposes a CreativeML Open RAIL++-M label, but the exact terms endpoints return HTTP 401 in this research context; no intended-use conclusion is possible. No artifact was downloaded or loaded. Do not convert the checkpoint, substitute a mirror, or proceed without the exact terms, component file identities, and owner-confirmed use scope. Full request outcomes and links: `api/runtime/adapters/pbr/evidence/supermat-official-assets-license-refresh-2026-09-25.md`.

### Registered fixed-geometry v2 candidate result (2026-09-25)

A new CPU-only candidate consumes the frozen face-ID/barycentric sidecar directly and checks its topology revision against the indexed mesh. It does not reconstruct raster alignment from mask bounds, does not read scoring arrays during estimation, leaves unsupported/unobserved channels unknown, and labels its residual confidence as uncalibrated. The training-only probe used resolution 32, `max_nfev=25`, `min_samples=3`, SciPy `soft_l1` loss with `f_scale=0.02`; it reproduced **70,688/70,688** training pixels (100% coverage), with forward-render RGB MAE **0.0729694**. Training output SHA-256 is `5a8b5b3e532c9b022f0694ea75feb3738f6c74b91920b7da1e6160716de4382a`; frozen training report SHA-256 is `af8d625ec1159ebf4cf99d9474995ff64c13306e4f976b8112a50158a68eea3d`. Estimator source SHA-256 is `bcb39942b4e62a416aabe9aed50ec03773028627872cd598ede90d9b2a31bd79`.

The valid allowlisted scorer opened scoring arrays only after the training-only freeze. Visible-texel coverage measured 1.0 (reported evidence; the audited rubric defines no separate coverage cutoff), and geometric-normal novel-light MAE passed at **0.0395586** (<=0.08). Base-color MAE **0.133329** (>0.08), base-color SSIM **0.516634** (<0.85), roughness MAE **0.403555** (>0.10), metallic MAE **0.326869** (>0.10), and metallic bias **0.283427** (>0.08) failed. The candidate is rejected and was not tuned or run on the RX 7900 GRE. CPU scorer telemetry: 0 accelerator devices, peak host RSS 74,940 KiB, scorer elapsed 0.0926 s; this is not target-hardware qualification. Valid score report SHA-256 `04a954e93f3d4de7aeafad62536015463b6a8cad2f5e8b3e0a39849c44d2ae93`; training-only report SHA-256 `e434f78edd871fdc9e580ed55ff4234f961f1feb789c1fd3be850269c630f119`. The allowlisted training scene is `ticket08-three-region-pbr-training-scene-v1.json` (SHA-256 `a460ee742fefb0cd1f29c55bf8f4290de0bbe08556fca61daabb200c28e52dc2`). Candidate runner SHA-256 `5ce95631be33758652c44291c8ddea9a062b3f733ea086d3e28e148f1e742a1b`; scorer source SHA-256 `fb171facd7f916e22a7f4b2860196feb7fca19cbf3c0ada0fa0c44bbb43414c6`. No Ticket 08 threshold changed.

Evidence-integrity correction: before the exact-key allowlisted training-scene file was added, one exploratory scorer invocation used a runner that parsed the broader fixture metadata JSON (which also contains material names). Although the estimator code consumed only camera and training-light fields and the numerical output was identical, that invocation is not valid isolation evidence. Its report is preserved as `ticket08-registered-fixed-geometry-v2-quality-score-metadata-contaminated.json` (SHA-256 `6aa9bb3cc9c4cee5481ee271688d68dd62ce6496cdfb2c2b8ea58ac823c34e45`) and excluded from acceptance. The training output was regenerated with the exact allowlist, confirmed byte-identical, and the valid scorer was then invoked. The corrected process preserves the same candidate parameters and does not use the prior failed metrics to tune the estimator.

Scoring-rubric correction: the first allowlisted scoring script version also encoded an extra 100% visible-coverage cutoff, which is not a separate hard gate in the audited `SELECTION.md`. That intermediate report is preserved as `ticket08-registered-fixed-geometry-v2-quality-score-unreviewed-coverage-threshold.json` (SHA-256 `d97c654786e235d0d2abb81df1584965cd5722d87b9bbf224218134523ac63c4`) and excluded. The final report uses only the audited channel, metallic-bias, and novel-light thresholds; coverage remains informational. Removing the extra cutoff does not alter the failed metrics or candidate decision.

### Registered latent-normal v3 candidate result (2026-09-25)

A separate candidate used exact face-ID/barycentric correspondences and fitted a bounded world-space latent shading normal per UV cell as a nuisance variable. It does not export or assert tangent normals, bump/height, opacity, or emissive. The allowlisted training-only run froze resolution 32, `max_nfev=60`, `min_samples=6`, `normal_prior_weight=0.01`, robust `soft_l1` loss, and `f_scale=0.02`. It covered **70,688/70,688** registered training pixels (100%) and had mean fitted training RGB MAE **0.0591110**. Output SHA-256 `115c0cf90f60e6df7a81c1d770f56dd71dd4f03a8f280af5a4b2f939e885e96e`; training report SHA-256 `55507a6cd42bc045ffd648b3be8c54f7744b9dc20836e248f7f9546ab2059d0d`; estimator source SHA-256 `d841da1d14773354f387fda7476a84df4616dbe14bac4f638c138c0e16ff0f54`.

After freezing, the separate scorer measured visible coverage 1.0, novel-light MAE **0.0404010** (<=0.08), base-color MAE **0.161017** (>0.08), base-color SSIM **0.425811** (<0.85), roughness MAE **0.251079** (>0.10), metallic MAE **0.201517** (>0.10), and metallic bias **0.520005** (>0.08). It is rejected; no tuning or RX 7900 GRE run followed. CPU scorer telemetry: zero accelerators, peak host RSS 74,296 KiB, score time 0.0835 seconds; estimator training time was 81.0 seconds, with candidate-stage peak host RSS not captured. Full report: `api/runtime/adapters/pbr/evidence/ticket08-registered-latent-normal-v3-quality-score.json`; source, runner, output, and training report are adjacent. Probe source SHA-256 `1369b0f0c9fa239c0a0b6dd3acf7de4f7f20072763badd7225a454c69736fabe`; scorer source SHA-256 `2bb17e161658c0805432abd468048332fc9ba2a0afa0fda798787753f1c92c40`. Frozen thresholds are unchanged.

### Further candidate frontier and registered-input feasibility (2026-09-25)

The official-source screen of Materialist, MaterialFusion/StableMaterial, and Intrinsic Image Fusion found no candidate ready for the frozen probe: the public interfaces either rebuild meshes or lack a caller-topology contract, immutable weight identities/terms, or a qualified AMD route. Details and primary-source links are in `api/runtime/adapters/pbr/evidence/inverse-rendering-candidate-frontier-2026-09-25.md`.

A separate feasibility review did not freeze a third registered inverse-rendering candidate. Current allowlisted inputs contain no truth-isolated Ticket 06 material-region IDs, and the registered views provide only a narrow repeated lighting set. Deriving regions from the same shaded RGB or imposing unvalidated spatial/normal priors would not add independent evidence. The review defines evidence required to reopen this branch in `api/runtime/adapters/pbr/evidence/registered-global-candidate-feasibility-screen-2026-09-25.md`. No scorer was invoked and no gates changed.

The complete Ticket 08 suite passes **25/25** in the project Python 3.12 environment with the isolated, checksum-verified SciPy overlay. This is regression evidence only; no PBR estimator passes the frozen quality gates or has RX 7900 GRE acceptance evidence.

### License-cleared candidate frontier update (2026-09-25)

A primary-source screen found no independently license-cleared, immutable candidate ready for Ticket 08 evaluation. LumiTex is the closest technical fit for an existing mesh and albedo/metallic/roughness maps, but its own README identifies non-commercial FLUX and Tencent renderer components, documents CUDA-specific dependencies, and does not pin/checksum a complete weight set. MatLat expressly licenses its weights CC BY-NC 4.0 and documents CUDA/nvdiffrast; Material Palette lacks metallic and its named SD 1.5 checkpoint is reported unavailable by its own README; MatMart's paper did not yield an official executable/checkpoint/license/runtime contract in this bounded screen. No assets were fetched or run, no fixture truth was accessed, and no frozen gate changed. Evidence and direct official source links: `api/runtime/adapters/pbr/evidence/license-amd-cleared-pbr-estimator-frontier-2026-09-25.md`.

### Registered inverse-render repair review (2026-09-25)

No third candidate was frozen or scored. The registered training inputs provide calibrated cameras, repeated three-light illumination, and topology correspondences, but no independently accepted topology-bound region IDs. A region map from fixture construction metadata or any after-the-fact spatial/normal prior would add answer-bearing or unvalidated assumptions. Existing fixed-geometry and latent-normal registered fits already fail the frozen gates. Reopen only with an accepted Ticket 06 region map and training-only multi-start stability/parameter-conditioning evidence before freezing one dev evaluation. No heldout data, prototype, or scorer was accessed. Details: `api/runtime/adapters/pbr/evidence/ticket08-analytic-research-2026-09-25.md` and `api/runtime/adapters/pbr/research/ticket08-registered-inverse-render-repair-screen-2026-09-25.md`.

### Bounded primary-source refresh (2026-09-25)

A bounded screen of four additional released research systems found no candidate ready for acquisition or target probing. FlashTex emits PBR texture files for an input mesh, but is prompt-conditioned generation, documents baked-light leakage into albedo, and documents a CUDA/PyTorch3D stack. Uni-Renderer describes image-space albedo/metallic/roughness decomposition but its official repository marks weights unreleased and specifies CUDA/NVIDIA A800. PBR-NeRF jointly estimates geometry and materials and requires NVIDIA OptiX in its setup. Neural-PBIR reconstructs shape and reflectance jointly and the official Meta Digital Twin Catalog license is CC BY-NC 4.0. No code, model, package, fixture, or truth was downloaded or run; no score is claimed. Exact primary-source links and per-candidate blockers: `api/runtime/adapters/pbr/evidence/ticket08-new-source-screen-flashtex-unirenderer-pbrnerf-neuralpbir-2026-09-25.md`. Ticket 08 remains acceptance-blocked, with all thresholds unchanged.

### Intrinsic Image Diffusion source screen (2026-09-25)

Intrinsic Image Diffusion (IID) reports albedo, roughness, and metallic outputs for indoor single-view material estimation and states a minimum 10 GB GPU requirement. It is not ready for probing: its published test/runtime path is NVIDIA CUDA, its interface does not document mapping to an unchanged caller topology, and its mutable README does not provide immutable model-weight identities/digests or clear terms covering the checkpoint assets. No code, weights, packages, fixture inputs, or truth were accessed. No PBR criterion changed. Source report: `api/runtime/adapters/pbr/evidence/intrinsic-image-diffusion-primary-source-screen-2026-09-25.md`.

### MatNet and ShadeNet source-only screen (2026-09-25)

A primary-source-only screen of two released local image-to-material models found no new candidate that clears source, resource, rights, and topology gates. MatNet advertises albedo/roughness/metallic (plus normal/depth) predictions and a 434 MB checkpoint, but its current model listing does not bind a full immutable weight revision/hash or complete code/depth-component/training-data terms; Materialist's published workflow requires CUDA-compatible GPU hardware and reconstructs a mesh instead of documenting caller-topology preservation. ShadeNet publishes a pinned full source revision with PyTorch and ONNX inference artifacts and required map channels, but its card is `CC-BY-NC-4.0`, which fails the rights gate; its image-only interface has no topology contract, and AMD memory/quality are unmeasured. No code/assets/packages were fetched or executed; no fixture/truth was accessed and no thresholds changed. Do not predeclare either candidate. Full direct-source details: `api/runtime/adapters/pbr/evidence/ticket08-models-matnet-shadenet-screen-2026-09-25.md`.

### NDJIR and DiffReg-PBIR source-only screen (2026-09-25)

Two further primary-source candidates fail the fixed PBR prerequisites. NDJIR reconstructs geometry and reports specular reflectance rather than the required metallic channel. An additional official-Dockerfile review confirms its base image is CUDA 11.0, it installs NNabla's CUDA extension and native components, and its README estimates 3.5 hours to train on 100 images with an A100; this supplies no AMD target qualification. DiffReg-PBIR declares CC-BY-NC 4.0, depends on CUDA/OptiX CUDA extensions, and reconstructs its own SDF mesh rather than estimating materials on unchanged caller topology. No code, weights, fixture inputs, or truth were accessed; no acceptance gates changed. Details and direct sources: `api/runtime/adapters/pbr/evidence/ticket08-ndjir-diffreg-pbir-primary-source-screen-2026-09-25.md`.

### NeRO and NeROIC source-only screen (2026-09-25)

Two additional inverse-rendering systems were screened from official repositories only. NeRO jointly reconstructs geometry and BRDF from posed views, extracts its own mesh/material maps, and documents nvdiffrast plus a CUDA ray-tracing extension; it does not establish unchanged caller-topology material transfer, immutable checkpoint/weight hashes, or an AMD/14 GiB route. NeROIC likewise reconstructs geometry and material/lighting from photos rather than accepting an unchanged caller mesh; it documents CUDA 10.2, PyTorch 1.7.1, and old PyTorch3D, and its license is noncommercial academic absent separate permission. Neither clears Ticket 08's source, rights, runtime, or topology gates. No assets were downloaded or executed, no fixture/truth was accessed, and no thresholds changed. Official sources: [NeRO](https://github.com/liuyuan-pal/NeRO) and [NeROIC](https://github.com/snap-research/NeROIC).

### NeuMaTex source-only screen (2026-09-25)

NVIDIA NeuMaTex accepts meshes and corresponding multiview images, but predicts diffuse base color plus a 6D neural specular latent rather than the required standard base-color, roughness, and metallic maps. Its public sources do not identify immutable implementation/checkpoint digests or checkpoint-specific terms. The paper reports an RTX 5090 setup without an AMD or <=14 GiB target measurement. Its predicted aleatoric uncertainty supports optimization and does not establish the required PBR channel quality or separate unknown/ambiguous material identity outcomes. No implementation/weights were acquired or run, no fixture/truth was accessed, and no criteria changed. Official sources: [project page](https://nvlabs.github.io/neumatex/), [paper](https://arxiv.org/abs/2606.26715), [paper HTML](https://arxiv.org/html/2606.26715v2).

### CHORD (Ubisoft La Forge) source-only screen (2026-09-25)

CHORD is a new source-screen candidate for image-to-material decomposition. Its official materials describe base color, normal, derived height, roughness, and metalness outputs, but the released tool estimates maps for a texture image and does not document transfer onto an unchanged caller mesh, UV/topology binding, or preservation of caller material regions. The official GitHub `main` source is mutable; the model requires gated Hugging Face access and explicit terms acceptance, and both source and weights use Ubisoft's Research-Only Copyleft license, which strictly prohibits commercial use. The published setup installs CUDA 12.8 PyTorch; no ROCm/MIGraphX/HIP path or <=14 GiB RX 7900 GRE memory result is reported. Ubisoft explicitly says artist feedback finds results not production-ready, notes weak metalness accuracy on strongly specular imagery, and gives no result against Ticket 08's frozen channel/novel-light scorer. These are material blockers to the project contract, despite channel overlap. No source, weights, package, fixture, or truth was downloaded/accessed, and no criteria changed. Official sources: [Ubisoft release and limitations](https://www.ubisoft.com/en-us/studio/laforge/news/1i3YOvQX2iArLlScBPqBZs/generative-base-material-an-opensource-prototype-for-pbr-material-estimation-debuting-at-siggraph-asia-2025), [CHORD repository](https://github.com/ubisoft/ubisoft-laforge-chord), [model card and access terms](https://huggingface.co/Ubisoft/ubisoft-laforge-chord), [license](https://github.com/ubisoft/ubisoft-laforge-chord/blob/main/LICENSE), [paper/project page](https://ubisoft-laforge.github.io/world/chord/). No candidate is selected; Ticket 08 remains acceptance-blocked.

### VideoMatGen source-only screen (2026-09-25)

VideoMatGen is a relevant geometry-conditioned PBR generator, but it does not clear the immutable-asset, AMD/resource, or acceptance-quality gates for Ticket 08. Its method assumes a known untextured mesh with valid UVs, predicts base color, roughness, metallicity, and height, and projects generated multi-view material buffers into that mesh's UV texture space; this is compatible in principle with retaining the supplied mesh topology. However, its primary task is text-to-material generation (image-conditioning is described as an extension), rather than evidence-grounded estimation of the observed asset's PBR properties. The paper names the base Cosmos-1.0-Diffusion-7BVideo2World checkpoint but does not provide a pinned VideoMatGen implementation or immutable fine-tuned model/VAE/T5 checkpoint identities, file digests, or complete use terms in the official sources reviewed. It reports inference at 2–3 minutes per asset on 8×A100 GPUs and training on 64×A100; no ROCm/HIP route or RX 7900 GRE ≤14 GiB measurement is reported. Published CLIP-FID/CMMD/LPIPS and VAE map-reconstruction metrics are not Ticket 08's frozen topology-bound map-quality/held-out-light acceptance evidence. No code, weights, packages, fixtures, or truth were accessed; no gates changed. Official sources: [NVIDIA research page](https://research.nvidia.com/labs/rtr/publication/hasselgren2026videomatgen/), [arXiv paper](https://arxiv.org/abs/2603.16566), [paper HTML](https://arxiv.org/html/2603.16566).


### Additional official-source screen: SVBRDF Uncertainty (2026-09-25)

The distinct SIGGRAPH 2025 frequency-analysis optimizer was screened from its official repository. It documents base-color, metallic, roughness, and entropy outputs from multi-view material captures, but its inspected checkout has no immutable source revision in this environment, no learned checkpoint identity, an unpinned `nvdiffrast` Git dependency with no documented AMD backend, and no established caller-topology/region-binding contract; bundled benchmark data terms also remain unreviewed. It is therefore rejected before acquisition or evaluation. No fixtures/truth, weights, code, or packages were accessed. Full source evidence: `api/runtime/adapters/pbr/evidence/ticket08-svbrdf-uncertainty-official-source-screen-2026-09-25.md`. Ticket 08 remains blocked; all acceptance criteria and selection thresholds are unchanged.


A bounded official-source screen found no additional ready candidate. NI-Tex requires an NVIDIA CUDA/H100-or-H200 stack and lists a 39.5 GB mutable-main model tree without a complete immutable lock; MatE extracts tileable material textures but lacks topology-bound mesh-region transfer, metallic, and released code. Neither is eligible for Ticket 08 probing. No assets or fixtures/truth were accessed and no thresholds changed. Evidence: `api/runtime/adapters/pbr/evidence/NI_TEX_MATE_SOURCE_SCREEN_2026-09-25.md`.

### Marigold IID Appearance v1.1 source gate (2026-09-26)

The official ETH Zurich Marigold IID Appearance v1.1 release is a technically relevant new lead: its model card specifies sRGB albedo plus linear roughness and metallicity outputs, and the official source describes local Diffusers inference. It is **not eligible** for package/model acquisition or fixture evaluation yet. The linked official `LICENSE-MODEL.txt` ends Attachment A with the literal `[insert use restrictions]` placeholder; the card's Stable Diffusion model-license endpoint returns HTTP 401. The reviewed Hugging Face listing exposed only abbreviated revision `e7280a0`, without a verified full immutable revision and selected-file hash manifest. AMD/14 GiB execution, topology projection, candidate preregistration, channel quality, and novel-light evidence remain absent. No code, package, checkpoint, fixture, or truth was accessed. Frozen thresholds remain unchanged. Full source-only gate record: `api/runtime/adapters/pbr/evidence/marigold-iid-appearance-v1-1-source-gate-2026-09-26.md`.

### IDArb and LINO-UniPS primary-source screen (2026-09-26)

A bounded official-source review found two channel/input-shape-relevant leads but neither clears pre-evaluation gates. IDArb's official README still marks release of inference code and pretrained checkpoints pending and documents a CUDA 11.8/A100 environment. LINO-UniPS now has PBR evaluation code, but it expects `./ckpt/lino_pbr.pth`; the only identified official HF checkpoint is tagged for normal estimation, and the PBR requirements pin a CUDA 12.4 wheel index plus `spconv_cu124` and xformers. Neither has an immutable PBR checkpoint/file-hash and complete PBR asset-rights lock, or RX 7900 GRE/14 GiB evidence. No assets, packages, datasets, fixture inputs, predictions, or truth were accessed. Full source record: `api/runtime/adapters/pbr/evidence/ticket08-idarb-lino-unips-primary-source-screen-2026-09-26.md`. Thresholds remain unchanged.

### ReLi3D primary-source candidate screen (2026-09-26)

ReLi3D is not eligible for Ticket 08 probing under the existing caller-topology/material-region contract. Its published path takes posed views through UV unwrapping and new-mesh reconstruction rather than emitting PBR maps bound to the supplied Modly topology/material regions; documented dependencies include CUDA-specific native components and no AMD route is provided. Full immutable identity and applicable conditional license terms are also unresolved. Weight file size was not treated as a VRAM measurement. No assets, packages, weights, fixtures, or truth were downloaded/accessed; no inference or quality scoring occurred. Frozen gates remain unchanged. See `api/runtime/adapters/pbr/evidence/ticket08-reli3d-primary-source-screen-2026-09-26.md`.

### MatForge-App source screen (2026-09-27)

MatForge-App's official v1.0 release names normal, roughness, and metallic outputs, but not required base color/albedo. The documented single-RGB-image interface does not bind maps to an existing caller mesh, material region, or UV/topology revision. Checkpoint digests/complete weight terms and an AMD/RX 7900 GRE route are not established; the README identifies a research-only restriction on pretrained PVT-v2 weights. Rejected before acquisition; no package, weight, fixture, or truth access and no gate change. Evidence: `api/runtime/adapters/pbr/evidence/TICKET08_MATFORGE_APP_SOURCE_SCREEN_2026-09-27.md`.

### Learned-gradient SVBRDF source screen (2026-09-27)

The official Xuejiao-Luo single-image SVBRDF repository is not eligible for Ticket 08 probing. Its documented outputs are diffuse, specular, roughness, and normal; it does not predict metallic, and specular cannot be relabeled as metallic. It also lacks a caller-topology/material-region interface, an immutable checkpoint digest and checkpoint-specific terms, and AMD/RX 7900 GRE/resource or frozen-quality evidence. Rejected before acquisition; no code, packages, checkpoint, fixture, or truth was accessed and no gates changed. Evidence: `api/runtime/adapters/pbr/evidence/ticket08-learned-gradient-svbrdf-screen-2026-09-27.md`.

### RGB-X and TextureWorks source screen (2026-09-27)

RGB-X predicts image-space PBR channels but uses mutable weight IDs with unresolved terms, is documented NVIDIA-only, lacks caller-mesh/topology inputs, and has no frozen-quality or RX 7900 GRE result. TextureWorks uses NVIDIA PTX image heuristics, does not estimate base color, and has no topology-bound transfer or required quality evidence. Both were rejected before acquisition/evaluation. No files, packages, checkpoints, fixture inputs, or truth were accessed; no criteria changed. Details and official links: `api/runtime/adapters/pbr/evidence/ticket08-rgbx-textureworks-source-screen-2026-09-27.md`.

### MaterialSeg3D source screen (2026-09-27)

MaterialSeg3D requires an albedo UV map and predicts material labels before assigning fixed roughness/metalness values by class; it does not demonstrate independent spatial PBR recovery for the required channels. Its official setup is CUDA/NVIDIA focused, its weight identity/terms are incomplete, and no AMD or frozen-fixture quality evidence exists. Rejected before acquisition; no packages, weights, fixture, or truth were accessed. Details: `api/runtime/adapters/pbr/evidence/ticket08-materialseg3d-official-source-screen-2026-09-27.md`. Ticket 08 remains blocked.

### SVBRDF Uncertainty optimization source screen (2026-09-27)

`svbrdf_uncertainty` is a conditional multi-view optimization lead: its public
code emits base-color, metallic, roughness, and entropy maps. However, the
official pinned optimizer hard-selects Mitsuba `cuda_ad_rgb`. Official
Mitsuba documentation describes `llvm_ad_rgb` for parallel CPU use and
`cuda_ad_rgb` for GPU use; its documented prebuilt variant set has no Vulkan
backend. This gives the current candidate no AMD GPU path, and a CPU-only route
does not meet the RX 7900 GRE acceptance gate. No code/package/model/fixture was
acquired or run. Keep the candidate unselected unless a justified, measured
AMD implementation is available. Report: `api/runtime/adapters/pbr/evidence/ticket08-svbrdf-uncertainty-source-screen-2026-09-27.md`; primary sources:
[official optimizer at pinned revision](https://github.com/rubenwiersma/svbrdf_uncertainty/blob/c58d1124f4f5316a6baa67e6467291bda145efcc/opt_sh.py),
[Mitsuba variants](https://mitsuba.readthedocs.io/en/stable/src/key_topics/variants.html).

DiffusionRenderer is a relevant image-space channel lead for base color, roughness, and metallic, but its official interface does not bind predictions to Modly's caller topology or material-region revisions. Its documented GPU use exceeds 22 GB, above the 14 GiB limit, and the setup is CUDA-specific with no AMD route. Its code/model license chain also needs qualification. Rejected before acquisition; no code, package, checkpoint, fixture or truth data was accessed, and no gates changed. Report: `api/runtime/adapters/pbr/evidence/ticket08-diffusion-renderer-official-source-screen-2026-09-27.md`.

DiffMat emits PBR maps while fitting a supplied Substance graph to a target texture, but it does not recover calibrated properties bound to Modly's mesh and material regions. The official source warns complex materials may need at least 16 GB VRAM, documents CUDA-default execution without ROCm evidence, and states a noncommercial license with practical dependence on Adobe Substance tools. Rejected before acquisition/evaluation; no repository, package, graph, fixture, truth, or GPU use occurred and gates remain unchanged. Report: `api/runtime/adapters/pbr/evidence/ticket08-diffmat-official-source-screen-2026-09-27.md`.

Meshy AI Texturing's official web application can accept OBJ/FBX/GLB assets and produce albedo, normal, roughness, and metallic textures. The reviewed source documents no local model/weight route, immutable model identity, Modly topology/material-region provenance, AMD runtime, or <=14 GiB evidence; the hosted path also fails the local/no-required-cloud gate. Rejected before account or asset use; no upload/download occurred and no gates changed. Report: `api/runtime/adapters/pbr/evidence/ticket08-meshy-ai-texturing-official-source-screen-2026-09-27.md`.

### Synthetic light-excitation conditioning probe (2026-09-27)

A CPU-only finite-difference probe of the existing GGX forward model found a modest local identifiability improvement when each of four synthetic views used a rotated three-light rig: the Jacobian condition number decreased from 19.19 with the same rig repeated to 12.27 with varied rigs, at full rank in both cases. This is an input-design observation only; it does not measure estimator accuracy or justify retuning/re-scoring rejected candidates. The current contract accepts one shared light set, so any future use requires genuinely calibrated per-view light metadata. No fixture/truth, estimator fit, scorer, model, or GPU was used. Script and method: `api/runtime/adapters/pbr/research/ticket08_light_excitation_conditioning.py`; report: `api/runtime/adapters/pbr/research/ticket08-light-excitation-conditioning-2026-09-27.md`. Gates remain unchanged.

### Per-view capture-metadata prerequisite audit (2026-09-27)

The frozen synthetic PBR scene provides camera transforms and one declared shared three-light rig; the fixture applies that same rig to every view, and the registered inverse-render input contract accepts this shared-rig record. It provides no evidence about varied lighting. In the current source-image workflow, observations carry artifact identity/path/digest/media type but no measured camera or lighting calibration, and the teapot source image contains no metadata. Per audited spec §15, missing exposure, white balance, camera, or lighting values must stay unknown and cannot be inferred. A real-image calibrated-light estimator therefore needs capture-time per-view light direction/radiance, declared coordinate frame/units, and calibration provenance bound by digest to each observation; known camera/exposure/white balance should also be retained. No fixture truth, scorer, model, GPU, or file was accessed/changed in this audit. Relevant current seams: `api/runtime/adapters/pbr/evidence/ticket08-three-region-pbr-training-scene-v1.json`, `api/runtime/adapters/pbr/probe_registered_fixed_geometry_v2.py`, `api/runtime/adapters/pbr/view_projection.py`, and `api/schemas/structured_asset.py`. This is an input prerequisite, not an acceptance pass or gate change.


### ARM official-source screen (2026-09-28)

ARM is scientifically relevant to albedo, roughness, metalness, and relighting, but the official release does not establish an interface that consumes Modly's caller-supplied topology/material regions, nor official pinned deployable weights/terms or an AMD inference route. It is rejected before acquisition/evaluation; no code, weights, fixture, truth, or GPU were accessed and no gates changed. Report: [`ticket08-arm-appearance-official-source-screen-2026-09-28.md`](../../../api/runtime/adapters/pbr/evidence/ticket08-arm-appearance-official-source-screen-2026-09-28.md).


### Redner official inverse-render source screen (2026-09-28)

Redner is a relevant differentiable-rendering foundation, but its documented path does not supply the Modly topology/material-region estimator contract or metallic recovery, and its published GPU package is CUDA-based. RX 7900 GRE resource and frozen quality evidence are absent. Related SVBRDF Estimation uses image-space methods and a custom Redner fork, with the same geometry/channel/AMD gaps. Rejected before acquisition/evaluation; no code, weights, fixtures, truth, or GPU accessed and no gates changed. Report: [`ticket08-redner-official-inverse-render-screen-2026-09-28.md`](../../../api/runtime/adapters/pbr/evidence/ticket08-redner-official-inverse-render-screen-2026-09-28.md).


### Fixed-geometry inverse-render interpolation improvement (2026-09-28)

A truth-isolated CPU development change replaced blockwise coarse-grid assignment with bilinear UV-cell-center expansion. This improves base-color MAE/SSIM from 0.1236/0.6408 to 0.1031/0.7203, but remains outside frozen limits (<=0.08/>=0.85); roughness MAE is 0.4152 (<=0.10), metallic MAE 0.3176 (<=0.10), metallic bias 0.2471 (<=0.08), and novel-light MAE 0.0416 (<=0.08). The changed fixed-normal fit cannot explain the fixture's height-derived shading normals, so material channels remain confounded under the available repeated-light observations. Candidate is not selected; no thresholds, fixture, or scorer changed; no target run. Full evidence: [`fixed-geometry-bilinear-expansion-cpu-development-2026-09-28.md`](../../../api/runtime/adapters/pbr/evidence/fixed-geometry-bilinear-expansion-cpu-development-2026-09-28.md). Full Ticket 08 CPU suite passes 30/30 in the project API venv with local SciPy installed; this verifies code contracts, not acceptance.

### PyTorch3D ROCm build preflight (2026-09-28)

The official PyTorch3D ROCm port (`b73d735ecf194c31de812feffef3a55cc3726128`)
successfully built in the exact project PyTorch 2.11.0+rocm7.14.0 image for
`gfx1100`, using a project-local overlay and the image-packaged SDK include and
library paths. The generated extension is pinned by digest in
`api/runtime/adapters/pbr/evidence/material-anything-pytorch3d-rocm-build-preflight-2026-09-28.md`.
This closes only the PyTorch3D package-build feasibility uncertainty; no
Material Anything model, fixture, source image, held-out input, or truth was
accessed during the build. After Ticket 04 became terminal, one isolated
16x16 generated triangle raster/interpolation operation ran on the RX 7900 GRE.
The 82-pixel face mask matched the independent CPU reference exactly; maximum
barycentric and interpolated-attribute absolute errors were 1.0523e-7 and
1.0448e-7. Peak allocated/reserved memory was 15,872 B / 2 MiB, and both
returned to a zero-byte baseline after cleanup. Full runtime, memory, command,
and artifact hashes: [`material-anything-pytorch3d-rocm-synthetic-op-preflight-2026-09-28.md`](../../../api/runtime/adapters/pbr/evidence/material-anything-pytorch3d-rocm-synthetic-op-preflight-2026-09-28.md).
This tests only a core PyTorch3D primitive. It does not test the Material
Anything UV-atlas route or establish Kaolin parity; its model weight
identities/terms also remain unresolved. No candidate selection, PBR gate, or
threshold changed.

### Material Anything source-reference contract probe (2026-09-28)

Generated CPU-only contract tests now cover the pinned source's separate UV
and geometry face-index gathers, face-row alignment across a UV seam, literal
post-raster clamp/scale/sentinel expression, uint8 truncation, and a bounded
binary mask dilation footprint. The focused suite passes 8/8. This is not a
Kaolin comparison: no Kaolin version/source revision is selected and no
Kaolin module or raster result is available in the project image/cache. Atlas
orientation, pixel centers, edge rules, out-of-range clipping, overlap/tie
selection, Kaolin barycentrics, and full OpenCV dilation output remain
unknown. The literal pinned postprocess also maps clamped XYZ endpoints -1
and +1 to -0.5 and 1.5 before uint8 casting; the report records the resulting
wrap behavior without altering upstream behavior. Full scope and evidence:
[`material-anything-uv-reference-contract-2026-09-28.md`](../../../api/runtime/adapters/pbr/evidence/material-anything-uv-reference-contract-2026-09-28.md).
This does not select Material Anything, pass the Kaolin bridge, or change any
Ticket 08 quality/AMD/resource gate.

### Kaolin immutable-reference runtime screen (2026-09-28)

The official immutable candidate reference is Kaolin `v0.18.0`, commit
`06ffb7d955ca26b608c60a9e862327c56b226921`, whose library states Apache-2.0.
Its default rasterizer dispatches to a custom CUDA extension; Kaolin's docs
require NVIDIA/CUDA for full functionality and say CPU installs omit CUDA
operations. The existing project image is PyTorch `2.11.0+rocm7.14.0` on
RX 7900 GRE and has no Kaolin module. Kaolin's documented tested PyTorch range
tops out at 2.8.0, and its listed requirement range at 2.5.1. The optional
nvdiffrast backend is not Material Anything's default and is absent from the
current image; using it as the reference would require separately proving its
parity with default Kaolin. No package was installed or built and no GPU,
model weights, fixture, observations, or truth were accessed. This screen
narrows the next decision: exact default-operator parity requires a separate
supported NVIDIA reference environment; otherwise a different AMD-compatible
reference must first be qualified against Kaolin. Ticket 08 remains open and
no gates changed. Full source links and environment identities are in
`api/runtime/adapters/pbr/evidence/material-anything-kaolin-reference-runtime-screen-2026-09-28.md`.

### MIRReS official-source screen (2026-09-28)


### VideoNeuMat official-source screen (2026-09-28)

VideoNeuMat's official project page says code is forthcoming and describes a
neural material built from 17 frames using Wan 2.1 14B / Wan 1.3B backbones.
The available source does not establish separate basecolor/roughness/metallic
maps bound to Modly's caller topology and material regions, immutable fine-tuned
checkpoint identity or terms, AMD qualification, or a frozen-fixture route.
Rejected before acquisition or evaluation. No packages, weights, fixtures,
observations, or truth were accessed; no acceptance gate changed. Evidence:
`api/runtime/adapters/pbr/evidence/ticket08-videoneumat-official-source-screen-2026-09-28.md`
(SHA-256 `3e12662b91e166a5d1f0af49ed507b3b45c1540fc8c5ededda178cbea9d892c9`).

### PM-PMVS official-source screen (2026-09-28)

The official README's calibrated multi-view, camera-attached-light route assumes
one uniform material and reconstructs a new oriented point cloud. It does not
establish separate base-color, roughness, and metallic recovery per region on
Modly's supplied topology. The documented runtime is NVIDIA CUDA and the terms
are academic-use with commercial use by author permission. Rejected before
acquisition/evaluation. No code, packages, weights, data, fixtures, truth, or
GPU were accessed; no gate changed. Evidence:
`api/runtime/adapters/pbr/evidence/ticket08-pmpmvs-official-source-screen-2026-09-28.md`
(SHA-256 `adac48fabf46b4f57c1c7fa907d6fbfa42babd7443ed9c25e5ff169f8ca2d3c0`).

### Source-shaped Kaolin gather operation (2026-09-28)

Added a narrowly scoped `index_vertices_by_faces` port matching the pinned
Kaolin v0.18.0 gather expression, plus a caller helper that preserves separate
UV and geometry face indices and original face-row order. Six generated CPU
tests pass in the project-owned AMD image. This does not port Kaolin
rasterization, establish UV raster parity, integrate Material Anything, or
qualify PBR quality/rights/weights; no acceptance gate changed. Exact source,
commands, hashes, and limitations:
`api/runtime/adapters/pbr/evidence/material-anything-kaolin-gather-port-2026-09-28.md`.

### Material Anything provenance and weight-identity hold (2026-09-28)

A fresh source/terms review confirms the pinned Material Anything source declares
MIT but its README says a significant portion of the code is based on Text2Tex.
Text2Tex identifies its own code under CC BY-NC-SA 3.0, and Material Anything
issue #18 asking for commercial-provenance clarification is still open. This is
an unresolved intended-use/provenance question, not a legal conclusion; the
MIT file alone is insufficient to advance this candidate. The estimator and
refiner revisions still lack selected file SHA-256 manifests and locally
verified bytes, and the inherited Stable Diffusion 2.1 asset terms/identity
also need review. No weights were downloaded or executed. The source-shaped
Kaolin gather is not an inference or raster bridge. Details and official source
links: `api/runtime/adapters/pbr/evidence/material-anything-text2tex-provenance-rights-screen-2026-09-28.md`.
Ticket 08 remains open; no threshold or acceptance gate changed.

### Live primary-source candidate qualification review (2026-09-28)

No candidate is dependency-ready for Ticket 08 acceptance evaluation; Ticket 08
remains blocked. The source/rights review did not fetch or execute model weights,
inspect fixture truth, change any frozen gate, or run a GPU process. Full
candidate-by-candidate findings and current primary-source links are recorded
in [`ticket08-live-primary-source-qualification-2026-09-28.md`](../../../api/runtime/adapters/pbr/evidence/ticket08-live-primary-source-qualification-2026-09-28.md).

Material Anything remains the required first candidate, but its source-derived
Kaolin UV raster contract has no pinned reference parity, its inherited
Text2Tex provenance/use terms remain unresolved, and full estimator/refiner/base
weight identities are incomplete. SuperMat remains blocked on base-model terms,
consumed-artifact identities, dependency closure, and topology-bound fusion.
Existing fixed-geometry inverse-render candidates fail frozen channel gates.
Freshly reviewed MatLat, Hunyuan3D-2.1 extension, and MyMeshy routes fail the
frozen AMD/topology/quality or estimator-contract requirements. No candidate
advances to weights or inference, and no acceptance status or threshold changed.

### Independent development fixture and one-way scorer (2026-10-01)

Added `api/runtime/adapters/pbr/development_fixture_v1.py` and the concrete
sidecars under `api/runtime/adapters/pbr/evidence/ticket08-development-fixture-v1/`.
This separately seeded 96x96 three-region scene is generated without importing
the frozen acceptance fixture builder. Candidate inputs and scoring targets
are separate NPZ files with SHA-256 identities bound by the v1 JSON manifest;
the candidate input loader enforces its field allowlist and manifest hash.
PBR truth, region truth, and the development novel-light render remain in the
target file. The one-way scorer accepts only three estimated maps and an
observed mask, runs no estimator, checks the target manifest/hash, and reports
map and development novel-light errors. Its output explicitly says it is not
acceptance or generalization evidence and applies no frozen gate. The focused
isolation/scorer suite passes 3/3. Frozen thresholds and held-out scoring paths
are unchanged; no GPU was used and no prior raw output was overwritten.

### View-diverse development v2 and bounded multi-start candidate (2026-10-01)

The v1 development run returned weak channel maps and inspection showed its
three training observations were the same combined-light render repeated.
That made it a poor development probe for multi-view recovery. It remains
preserved as v1 evidence; no frozen acceptance data were used.

Added `development_fixture_v2.py` with a new seed, three distinct calibrated
camera poses, matching mesh/UV correspondences and per-view visibility masks.
Candidate inputs and scoring targets are separate hash-bound NPZ files with an
explicit field allowlist. The one-way development scorer now accepts either
versioned dev manifest, expands smaller candidate maps using the same nearest
map rule as the registered fixed-geometry scorer, preserves `observed=false`
as unknown, and never invokes an estimator. Focused tests pass 5/5, including
versioned target isolation and different per-view observations.

The region inverse renderer now uses three fixed deterministic initial points
for bounded GGX least-squares fitting and selects the lowest robust objective.
On v2, fitting at 32x32 and expanding to 96x96 yielded 100% visible-map
coverage on 2,408 pixels: base-color MAE 0.03331 and SSIM 0.58207, roughness
MAE 0.02587, metallic MAE 0.03274, and novel-light MAE 0.00316. Base-color
SSIM remains below the unchanged 0.85 quality limit. A 48x48 run reached
61.96% coverage and did not improve SSIM. No frozen gate was applied, no
heldout data were read, and no GPU was used. The 32x32 estimate, score, and
provenance are stored in
`api/runtime/adapters/pbr/evidence/ticket08-development-fixture-v2/`.
This remains development-only synthetic evidence; Ticket 08 is not accepted.

### Normal-detail development domain review (2026-10-01)

The region-inverse v2 lambda sweep's development fixture uses a planar scene
with constant +Z normals. It therefore does not exercise the height-derived
surface detail present in the frozen PBR scene. The frozen v2 training report
records large fit errors in two of three regions, and the terminal score fails
base-color MAE/SSIM, roughness, metallic, and conductor/dielectric bias. The
previous latent-normal candidate also fails the material-map gates and has no
spatial or height-integrability model. Further base-color prior tuning is not a
justified next step.

The next development candidate needs a distinct seeded fixture with
spatially-varying training normals derived from a documented height field,
separate hash-bound input and target files, and a coherent normal/detail
nuisance model. The scorer and all frozen limits remain unchanged. The terminal
v2 scorer was not rerun and frozen target pixels were not inspected in this
review. Full diagnosis and evidence boundary:
`api/runtime/adapters/pbr/evidence/ticket08-normal-detail-domain-gap-2026-10-01.md`.
