# Modly AMD-Native Semantic 3D Asset Pipeline — Specification

**Status:** Audited specification; ready for implementation tickets  
**Spec revision:** 2.0  
**Date:** 2026-09-24  
**Target host:** Modly  
**Primary hardware target:** AMD Radeon RX 7900 GRE (`gfx1100`, 16 GB VRAM)  
**Primary execution strategy:** Torch-MIGraphX + PyTorch ROCm + HIP, with Modly's existing renderer for the POC and optional Vulkan-native work where justified

## Problem Statement

Modly already provides local 3D generation, workflows, external model/process extensions, headless workflow execution, and asset viewing. The useful 3D-AI stack is nevertheless fragmented: geometry generation, native-3D part segmentation, semantic naming, material-region segmentation, material classification, PBR recovery, validation, and export are supplied by different projects with different data models, GPU assumptions, dependencies, and failure modes.

The project needs Modly to become a modular AMD-first semantic 3D reconstruction and authoring environment without making the product architecture dependent on any individual upstream model or vendor. It must accept source observations or an existing mesh and produce a structured, editable 3D asset whose geometry, semantic parts, material regions, material classifications, render properties, confidence state, provenance, corrections, and intermediate artifacts remain explicit and independently replaceable.

The reference workflow must run on the RX 7900 GRE without an NVIDIA driver, CUDA Toolkit, CUDA runtime, or NVIDIA-only binary dependency. Ordinary PyTorch model code should not be rewritten simply because it originated in a CUDA-first repository. Dense neural modules should be evaluated for Torch-MIGraphX execution; PyTorch ROCm remains the fallback for modules that cannot be usefully lowered; genuinely CUDA-specific 3D operators should be translated, replaced, or implemented as reusable HIP primitives.

The first proof of concept must prove the end-to-end contracts and AMD execution path before substantial custom UI or model-training work is attempted.

## Solution

Build an extension-driven semantic 3D pipeline on top of Modly's existing workflow, process-extension, asset, and headless execution surfaces. Modly remains the control plane. New inference and processing behavior is supplied by replaceable adapters plus a shared AMD Runtime deep module.

The reference workflow is:

1. Acquire one or more source observations or import an existing mesh.
2. Generate geometry when no mesh is supplied.
3. Segment the geometry into candidate object parts.
4. Assign semantic identities to part segments.
5. Segment surface regions by material where evidence supports a distinction.
6. Classify material identity independently of render-property estimation.
7. Estimate PBR properties and preserve uncertainty/provenance.
8. Fuse compatible evidence without pretending incomparable confidence scores are equivalent.
9. Validate geometry, mappings, material assignments, and metadata.
10. Allow targeted user correction and targeted stage re-runs.
11. Export GLB/glTF plus a versioned structured sidecar.

The initial candidate stack is Hunyuan3D for geometry generation, P3-SAM from Hunyuan3D-Part for native-3D segmentation, a separate semantic resolver for part names, a distinct material-region segmentation strategy, and Material Anything or a comparable system for PBR recovery. X-Part is not treated as the segmentation engine: it is an optional part-generation/decomposition refinement stage downstream of P3-SAM and is not required for the initial POC.

The reference generator is not permanently fixed by this specification. Hunyuan3D 2 Mini/Turbo is the first candidate because Modly already has extension support and it is more plausible on the 16 GB target. Newer Hunyuan variants or another generator may replace it if the AMD compatibility probe shows a better quality/performance fit.

## User Stories

1. As a Modly user, I want to generate a 3D asset from an image locally, so that I do not need a hosted generation service.
2. As a Modly user, I want to provide multiple observations when an adapter supports them, so that reconstruction or material inference can use more evidence than one image.
3. As a Modly user, I want to import an existing mesh instead of generating one, so that segmentation and material recovery can operate independently of generation.
4. As a Modly user, I want generation to be a replaceable workflow stage, so that I can switch models without rebuilding downstream steps.
5. As a Modly user, I want a generated or imported mesh to be decomposed into coherent 3D parts, so that the result is structurally editable rather than monolithic.
6. As a Modly user, I want part segmentation to operate on 3D geometry or consistent multi-view evidence, so that boundaries remain meaningful across viewpoints.
7. As a Modly user, I want segmented parts to receive semantic labels, so that the asset contains meaningful object structure rather than anonymous region IDs.
8. As a Modly user, I want geometric segmentation and semantic naming to remain separate operations, so that a strong segmenter is not rejected merely because its labels are weak.
9. As a Modly user, I want multiple segmentation or semantic engines to be able to evaluate the same region, so that the system can compare evidence rather than trust one model blindly.
10. As a Modly user, I want every model-derived assertion to expose its confidence state, including an explicit unknown or uncalibrated state when no trustworthy numeric score exists, so that the UI never invents certainty.
11. As a Modly user, I want uncertain or conflicting regions to be flagged instead of silently converted into authoritative structure, so that errors are correctable.
12. As a Modly user, I want material regions to be distinct from semantic parts, so that one part may contain multiple materials and one material may appear across multiple parts.
13. As a Modly user, I want material-region boundaries to be recoverable independently of PBR estimation, so that a material estimator is not forced to double as a segmenter.
14. As a Modly user, I want material regions to receive material identity candidates, so that downstream tools can distinguish categories such as rubber, glass, plastic, painted metal, or bare metal.
15. As a Modly user, I want material identity classification to remain separate from PBR property estimation, so that either system can be replaced or corrected independently.
16. As a Modly user, I want base color/albedo estimated separately from illumination where the selected estimator supports it, so that lighting is not permanently baked into the material description.
17. As a Modly user, I want roughness estimates for material regions, so that surfaces respond plausibly under new lighting.
18. As a Modly user, I want metallic estimates for material regions, so that metal and dielectric behavior remain physically meaningful.
19. As a Modly user, I want normal, bump, or other supported detail information preserved or recovered explicitly, so that fine surface structure survives export without mislabeling one representation as another.
20. As a Modly user, I want opacity, emissive, and later PBR channels represented only when supported by evidence or the selected estimator, so that unsupported channels remain unknown rather than fabricated.
21. As a Modly user, I want every derived property to record provenance, so that I can determine which model, weights, parameters, source observation, pipeline stage, and backend produced it.
22. As a Modly user, I want intermediate artifacts preserved, so that a failed late stage does not force regeneration from the beginning.
23. As a Modly user, I want to re-run only a selected stage and its dependents, so that I can iterate without repeating unaffected expensive stages.
24. As a Modly user, I want to swap one extension implementation for another while preserving compatible upstream and downstream stages, so that workflows survive model churn.
25. As a Modly user, I want workflow validation before execution, so that incompatible asset types, schema versions, or missing capabilities fail before consuming GPU time.
26. As a Modly user, I want each stage to expose meaningful domain controls rather than requiring knowledge of vendor-specific implementation details, so that workflows remain understandable.
27. As a Modly user, I want AMD execution to be the normal path on my system, so that I do not need NVIDIA hardware or an NVIDIA runtime.
28. As a Modly user, I want dense PyTorch neural modules to be evaluated for Torch-MIGraphX execution, so that AMD inference can use MIGraphX optimizations without requiring ONNX conversion.
29. As a Modly user, I want a module that fails MIGraphX correctness, compatibility, or performance gates to use PyTorch ROCm deliberately, so that compiler coverage does not block a useful workflow.
30. As a Modly user, I want specialized 3D operators to use reusable HIP implementations where practical, so that the same AMD work can benefit multiple extensions.
31. As a Modly user, I want the runtime to report which backend executed each major module or stage, so that MIGraphX, PyTorch ROCm, HIP, CPU, and any native rendering path are observable.
32. As a Modly user, I want peak VRAM and execution time recorded per heavy stage, so that I can understand whether a workflow fits my hardware.
33. As a Modly user, I want models that cannot coexist in 16 GB VRAM to load sequentially and release resources between stages, so that the reference workflow remains viable on the RX 7900 GRE.
34. As a Modly user, I want adapters that cannot fit the 16 GB target to be rejected or placed behind an explicit offload/low-memory mode rather than failing unpredictably.
35. As a Modly user, I want a failed extension to produce bounded stage-specific diagnostics, so that failures are actionable rather than generic subprocess errors.
36. As a Modly user, I want failed or repaired extension processes to restart cleanly, so that stale workers do not preserve broken model state.
37. As a Modly user, I want the complete workflow to run headlessly, so that it can be driven by Modly CLI, scripts, agents, and automated evaluation.
38. As a Modly user, I want headless and visual workflow execution to use the same underlying contracts, so that automation does not become a second implementation.
39. As a Modly user, I want the final GLB/glTF to preserve usable part and material boundaries where the interchange format permits, so that downstream engines can use the structure directly.
40. As a Modly user, I want structured sidecar metadata alongside the GLB, so that semantics, evidence, confidence state, provenance, corrections, and metadata not representable in glTF are not lost.
41. As a Modly user, I want the system to distinguish observed source data, model inference, deterministic derivation, and user correction, so that the asset history remains auditable.
42. As a Modly user, I want manual corrections to survive compatible re-runs, so that the system does not repeatedly overwrite confirmed information.
43. As a Modly user, I want corrections whose target geometry is no longer compatible to be marked orphaned for review instead of silently transferred to the wrong region.
44. As a Modly user, I want a low-confidence region to be targetable for a different model or additional observation, so that uncertainty can drive selective refinement.
45. As a Modly user, I want imported or generated assets to pass deterministic structural validation before export, so that broken geometry, invalid mappings, or material-reference errors are caught early.
46. As a Modly user, I want units, coordinate basis, transforms, UV sets, and texture color-space conventions represented explicitly, so that the asset does not change meaning between stages.
47. As a Modly user, I want source observations to retain known camera and capture metadata when available, so that later inverse-rendering or multi-view extensions can use it.
48. As a Modly user, I want the first useful pipeline to work before a large custom UI rewrite is attempted, so that technical value is proven first.
49. As a developer, I want each extension to declare the capabilities it provides, so that Modly can validate workflows without knowing vendor-specific details.
50. As a developer, I want capability inputs and outputs to use versioned asset contracts, so that extensions remain composable as the schema evolves.
51. As a developer, I want legacy Modly extensions and workflows to remain usable when the structured-asset contract is added, so that the new work does not unnecessarily break existing generation workflows.
52. As a developer, I want the AMD Runtime hidden behind a small interface, so that individual extensions do not each reinvent backend detection, compilation, profiling, fallback, and memory cleanup.
53. As a developer, I want reusable HIP 3D primitives to be shared across extensions, so that KNN, neighborhood search, grouping, sampling, interpolation, rasterization, voxelization, and mesh extraction are not repeatedly ported.
54. As a developer, I want an adapter compatibility report, so that each imported model clearly shows MIGraphX status, ROCm fallback, HIP work, native dependencies, VRAM use, quality status, and known blockers.
55. As a developer, I want the reference pipeline to save deterministic fixtures and stage outputs, so that backend changes can be regression-tested.
56. As a developer, I want quality comparisons between AMD execution and a known-good upstream path where one is available, so that acceleration work does not silently alter model behavior.
57. As a developer, I want the system to tolerate new generators, segmenters, semantic resolvers, material segmenters, classifiers, and PBR estimators without changing Modly core unless the public workflow contract itself lacks a required concept.
58. As a developer, I want the structured-asset schema to evolve by explicit version and migration rules rather than ad-hoc fields, so that older projects remain interpretable.
59. As a developer, I want extension failures to preserve enough provenance to reproduce the failing stage independently, so that debugging can occur outside the full workflow.
60. As a developer, I want stage cache keys to include the exact input artifacts, adapter/version, weights, parameters, and relevant runtime version, so that cached outputs are never reused after a semantically significant change.
61. As a developer, I want weights and extension revisions pinned by immutable identity for reproducible runs, so that an upstream update cannot silently change an old project.
62. As a developer, I want arbitrary downloaded extensions treated as executable code rather than trusted data, so that the application can make trust and provenance visible.
63. As a future extension author, I want physical-property assertions such as density or hardness to fit the same evidence/provenance model later, so that adding engineering properties does not require redesigning the asset schema.

## Implementation Decisions

### 1. Modly remains the host and control plane

Modly is not replaced with a new monolithic application for the POC. Its workflow graph, external model/process extension system, headless workflow execution, project/asset behavior, and existing viewer remain the user-facing control plane.

Core Modly changes are allowed when the current public seam cannot faithfully represent a required concept. The rule is **minimal host change, not zero host change**.

### 2. The primary external seam is the Modly workflow/extension execution seam

The highest-value test seam is a workflow or process run receiving versioned inputs and producing registered artifacts plus bounded diagnostics. New model behavior should enter Modly through this seam instead of through a second orchestration system.

Current Modly releases already support external model/process extensions and canonical `workflow-run`, `capability`, and `process-run` headless concepts. The existing extension manifest is narrower than the Structured Asset contract in this specification, so the POC may need an additive host/manifest extension for richer artifact descriptors or multi-output metadata. Existing single-image/image-to-mesh behavior must remain compatible.

### 3. Do not assume named multi-view ports already exist everywhere

Current Modly work has an open design for stable named model input ports. The semantic pipeline must therefore not depend on named multi-view ports until that host capability is present and tested. The POC may begin with single-image generation and mesh-based downstream stages, then enable named observations as an additive contract.

### 4. The Structured Asset is the canonical data model

The canonical asset is not merely a mesh. It is a versioned record containing at least: asset/schema identity; geometry artifact and digest; topology revision; coordinate basis, handedness, units, and transforms; UV/texture-space conventions; object/component hierarchy; part mappings; semantic assertions; material-region mappings; material-identity assertions; PBR assertions; optional future physical-property assertions; confidence/evidence metadata; observations; provenance; user corrections; validation state; and intermediate stage artifacts.

GLB/glTF is the primary portable 3D artifact. A versioned sidecar stores richer semantics and run history that glTF cannot represent reliably or ergonomically.

### 5. Geometry mappings must be explicit and invalidatable

A part or material region must map back to concrete geometry using a representation appropriate to the stage, such as primitive/mesh IDs, face IDs, vertex IDs, masks, UV regions, or a derived correspondence table. Mappings are always tied to a topology revision.

When topology changes, downstream mappings are not presumed valid. The system either remaps them with a proven correspondence transform or invalidates/orphans them for re-computation or user review.

### 6. Stable IDs are semantic identities, not guesses about unchanged topology

Part/material IDs should remain stable across a targeted re-run only when a deterministic correspondence demonstrates that the underlying region is the same. Stable IDs must not be preserved merely because two regions occupy a similar location or have a similar label.

### 7. Evidence kind is distinct from confidence

Each assertion declares an evidence kind such as observed, deterministic-derived, model-inferred, or user-confirmed. Confidence describes uncertainty in an inference; it does not convert an inference into an observation.

A model may supply a native score, Modly may derive a quality score, or no trustworthy numeric confidence may exist. The contract therefore requires a confidence **state**, not a fabricated number. A numeric score, when present, records its source and whether it is native, calibrated, derived, or uncalibrated.

### 8. Provenance is sufficient for reproduction

Provenance for a derived assertion or stage result includes, where applicable: adapter identity and revision, upstream repository/commit, model/weights identity and digest, runtime versions, backend, input artifact digests, parameters, random seed, source observations, device identity, and stage/run identity.

### 9. Part segmentation and semantic identification are separate capabilities

Part segmentation answers which geometric elements form a coherent candidate part. Semantic identification answers what that part represents. They remain independently replaceable.

P3-SAM is the initial Hunyuan3D-Part segmentation candidate because it is explicitly a native-3D part segmenter. X-Part is a distinct optional part-generation/decomposition refinement stage and is not required to satisfy `Segment Parts`.

### 10. X-Part is optional in the POC

The currently public X-Part release is described upstream as a light version. The initial POC must not make full X-Part behavior a dependency of successful semantic segmentation. If X-Part proves useful and fits the AMD/hardware constraints, it may be added behind a separate capability after P3-SAM output is available.

### 11. Material-region segmentation is independent of part segmentation and PBR estimation

Material boundaries are not assumed to equal semantic-part boundaries. Material-region segmentation is a separate capability that may use geometry, source observations, multi-view features, PBR discontinuities, or a combination of evidence.

Material Anything is not assigned responsibility for semantic material-region segmentation unless an adapter explicitly demonstrates that behavior. Its initial role is PBR material generation/estimation/refinement.

### 12. Material identity and PBR properties are separate assertion families

A material classifier may assert `rubber`, `painted steel`, `ABS`, `glass`, or another semantic identity. A PBR estimator may assert base color/albedo, roughness, metallic, bump/normal/detail, opacity, or emissive information. Neither output is treated as a substitute for the other.

### 13. Material Anything is a PBR candidate, not an all-purpose material oracle

Material Anything is a candidate adapter for generating/estimating PBR material maps from 3D assets. Its documented outputs include albedo, roughness, metallic, and bump maps. Any support for additional channels must be adapter-specific and proven rather than inferred from the generic asset schema.

Its Blender-based rendering/preprocessing requirements may be used internally by the adapter for the POC. That does not make Blender the Modly viewport or the product's architectural renderer.

### 14. Future physical properties use the same assertion model but are not POC requirements

The schema may represent later assertions such as density, hardness, elasticity, thermal conductivity, or refractive index with units/ranges/evidence/provenance. The first POC does not claim to infer these reliably.

### 15. Source observations are preserved as evidence

Source images, image sets, video frames, scans, or other observations are referenced by immutable artifact identity. Known camera pose, intrinsics, exposure, white balance, lighting metadata, or capture ordering should be retained when available; absent metadata remains absent rather than guessed.

### 16. Arbitration occurs above individual adapters

No upstream model becomes the authoritative object model. The fusion layer combines compatible assertions and exposes conflicts.

Raw confidence numbers from different adapters are not directly compared unless they have been normalized or calibrated onto a common interpretation. The POC may use deterministic rules, within-adapter rankings, provenance quality, geometric consistency, and explicit conflict states without pretending heterogeneous scores are commensurate.

### 17. User corrections outrank model inference but remain topology-bound

A user-confirmed correction supersedes competing model inference for the same valid target. If the target mapping becomes invalid after geometry changes, the correction becomes orphaned or pending-remap rather than silently attaching to a different region.

### 18. AMD-native is defined operationally

The reference path is AMD-native when it runs on the target Radeon system without NVIDIA drivers, CUDA Toolkit, CUDA runtime libraries, or NVIDIA-only compiled extensions.

Using PyTorch APIs named `torch.cuda` is not itself disqualifying because ROCm intentionally exposes the CUDA-shaped PyTorch device API. The requirement concerns the actual runtime/backend and binary dependencies, not symbol names in upstream Python.

### 19. Torch-MIGraphX is the preferred compiler path for suitable dense neural modules

PyTorch remains the frontend. Candidate modules are compiled through Torch-MIGraphX using the supported `torch.compile(..., backend="migraphx")` path or FX lowering when appropriate.

MIGraphX use is gated by correctness and measured usefulness. A module is not forced through MIGraphX merely to increase compiler coverage.

### 20. Mixed MIGraphX/ROCm execution is explicit; automatic partition fallback is not assumed

The runtime must not assume that Torch-MIGraphX will automatically partition every unsupported graph into ideal MIGraphX and PyTorch regions. Compilation boundaries are chosen at module/submodule seams that can be tested independently.

If a candidate module fails to lower, produces incorrect results, or performs worse than the accepted fallback policy, that module runs through PyTorch ROCm and the compatibility report records the reason.

### 21. PyTorch ROCm is the standard neural fallback

Ordinary PyTorch inference that does not benefit from or cannot use MIGraphX remains on PyTorch ROCm. CPU fallback is permitted only when required by an upstream operation or when GPU execution is not useful; it must be visible in telemetry if it materially affects latency.

### 22. HIP owns reusable CUDA-specific 3D primitives

CUDA-origin operations that are sparse, topology-dependent, custom-kernel-heavy, or poor graph-compiler targets should be translated, replaced, or implemented as reusable HIP/C++ primitives where practical.

Candidate shared primitives include KNN, radius/neighborhood search, sampling, grouping, interpolation, scatter/gather, rasterization, voxelization, sparse neighborhood operations, and mesh/surface extraction. A shared primitive is introduced only after at least two concrete consumers or a clearly reusable dependency justify the seam.

### 23. The existing Modly viewer remains the POC viewer

Modly currently renders its UI through an Electron/Three.js stack. Replacing that viewer with Vulkan is not required for the POC and would contradict the goal of proving the inference/data pipeline with minimal host churn.

Vulkan remains available for native offscreen rendering, validation, custom compute, or a future dedicated viewport if a measured need justifies it. The Structured Asset and extension contracts must not depend on a particular UI renderer.

### 24. The reference generator is selected by a target-hardware probe

Hunyuan3D 2 Mini/Turbo is the initial generation candidate because Modly already supports these variants and they are plausible on the 16 GB target. Before implementation tickets lock the generator, a probe records VRAM, native dependencies, AMD compatibility, generation quality, and required porting work.

Newer Hunyuan variants or another generator may replace the candidate without changing downstream contracts.

### 25. The segmentation stage has a hard 16 GB viability gate

Public Hunyuan3D-Part Modly work currently advertises substantially more VRAM than the 16 GB target. P3-SAM therefore remains a candidate, not a presumed fit.

The POC accepts P3-SAM only if the AMD port fits within the target through native execution, precision changes, staged loading, offload, or another explicitly supported low-memory strategy without unacceptable quality loss. Otherwise a different native-3D segmenter becomes the reference adapter.

### 26. The AMD Runtime is a deep module

Extensions do not individually reimplement device detection, Torch-MIGraphX compilation, ROCm fallback, backend policy, execution telemetry, VRAM accounting, deterministic cleanup, or backend/version reporting.

The AMD Runtime exposes a small interface and hides backend complexity. Removing it should force substantial duplicated backend logic to reappear across multiple adapters; otherwise it is too shallow.

### 27. Runtime environments are version-pinned and isolated

ROCm, PyTorch, MIGraphX, Torch-MIGraphX, Python packages, custom native extensions, and model weights are treated as a compatibility set. Each adapter records the supported set used for a successful run.

Adapters use isolated/persistent environments where necessary rather than mutating one global Python environment. Modly packaging must not assume the AppImage or embedded Python environment is appropriate for compiling native ML dependencies.

### 28. Model residency is sequential by default on the 16 GB target

Heavy adapters may load, execute, serialize results, release model state, synchronize as required, clear framework caches where useful, and exit/restart the worker before the next heavy stage. Persistent residency is an optimization only after measured headroom exists.

### 29. Intermediate artifacts are content-addressed and resumable

Every major stage emits a durable artifact or structured result with a digest. Stage cache identity is derived from semantically relevant inputs: input artifact digests, adapter revision, weights digest, parameters, schema version, and runtime factors that can change output semantics.

A targeted re-run invalidates only dependent stages whose cache keys or geometry mappings are no longer valid.

### 30. Headless execution is the POC acceptance path

The reference pipeline is not considered complete until it can execute through Modly's canonical headless workflow/process surface. The visual editor consumes the same contracts but is not required to prove the first vertical slice.

### 31. Diagnostics are stage-aware, bounded, and reproducible

A failure identifies the workflow stage, adapter, adapter revision, backend, model/weights identity where available, input artifact identity, error class, and detailed log location. User-facing output contains a bounded summary rather than unbounded subprocess logs.

A failed or repaired worker must not remain alive with stale model state.

### 32. Adapter trust and reproducibility are visible

Installing an extension means executing third-party code. The POC does not need a full sandbox, but it must distinguish trusted/pinned reference adapters from arbitrary local or GitHub-installed extensions. Reference workflows pin adapter revisions and weight identities rather than silently following upstream `main` or mutable model tags.

### 33. Distribution/legal packaging is separate from capability architecture

Third-party redistribution, notices, model terms, and release packaging are a separate release workstream. They do not determine the internal capability vocabulary or force vendor-specific dependencies into the Structured Asset model.

### 34. Cloud services are never required by the reference workflow

An extension may optionally add a cloud implementation later, but the reference generation, segmentation, semantic, material, validation, and export path must remain locally executable.

### 35. glTF 2.0 defines the POC interchange frame and metallic-roughness export convention

The Structured Asset may preserve source/native coordinate metadata internally, but GLB/glTF export normalizes to glTF 2.0 conventions: right-handed coordinates, +Y up, +Z forward, and meters for linear distance. Any conversion is recorded in provenance.

The POC export target is glTF 2.0 metallic-roughness PBR. Base-color/emissive textures follow glTF color-space semantics; scalar metallic/roughness/occlusion and normal data use linear interpretation as defined by glTF. Normal and bump are distinct representations and must not be silently relabeled. Adapter-native material data that cannot be represented losslessly remains in the sidecar or a later extension format.

### 36. Semantic and material labels are open-vocabulary assertions with optional normalization

The POC does not impose a universal ontology. An assertion stores the model/user label exactly enough for provenance plus an optional normalized label, category identifier, or ontology reference when a resolver can provide one.

Tests use fixture-specific expected labels or a declared local ontology. The system must not collapse distinct material concepts merely to force all adapters into one taxonomy.

## Testing Decisions

### Testing philosophy

Tests validate externally observable behavior through the highest stable seam available. They should not depend on helper-function call counts, private class layouts, or exact compiler partition internals unless the test is specifically for the backend adapter itself.

The primary end-to-end seam is: **Modly workflow/process execution in → registered artifacts, Structured Asset state, telemetry, and bounded diagnostics out**.

### Host/extension contract tests

The host must prove that legacy single-image/image-to-mesh extensions still execute after structured-asset support is added. New contracts must validate schema version, required inputs, output artifact identity, unsupported asset versions, and malformed metadata before dispatch.

If named multi-view ports are introduced, tests must cover stable names, required/optional ports, ordering, duplicate/unknown names, legacy compatibility, and transport into the extension worker.

### Structured Asset contract tests

Tests verify serialization/round-trip behavior for geometry identity, topology revision, coordinate conventions, UV references, part mappings, material mappings, assertions, confidence state, provenance, corrections, and stage artifacts.

A topology-changing fixture must prove that stale region mappings and corrections are invalidated or explicitly remapped rather than silently reused.

### AMD Runtime tests

The runtime must:

- identify the target AMD architecture correctly;
- execute a representative supported module through Torch-MIGraphX;
- validate MIGraphX output against the PyTorch ROCm baseline within declared tolerance;
- reject or fall back from a deliberately unsupported/incorrect compilation case;
- report the actual backend used;
- prove that no NVIDIA runtime library is required for the reference path;
- release enough resources between heavy stages to prevent stale-residency OOMs;
- record stage/module timing and peak memory;
- preserve the exact version identities used for a run.

### Generator tests

The reference generator must accept its declared observation inputs, emit a valid geometry artifact, preserve run provenance, and fail without corrupting previously registered artifacts. Single-image fixtures explicitly acknowledge that unseen geometry is inferred rather than observed.

### Part-segmentation tests

A segmenter must map each result back to the input topology revision and explicitly declare overlap policy. Fixtures with known segmentation ground truth should use task-appropriate metrics such as face-level IoU or another declared partition metric. Unsupported geometry must fail with an actionable diagnostic.

### Semantic-resolver tests

A semantic resolver must support `unknown`/ambiguous output, preserve multiple candidates when configured, and never require a numeric confidence if the model does not provide one. Evaluation uses a declared label ontology for the fixture rather than free-form string similarity alone.

### Material-region tests

Fixtures must include at least one part containing multiple materials and one material spanning multiple parts. Region output must map back to geometry/UV space and remain separate from material-class and PBR assertions.

### Material-classification tests

The classifier must be able to return unknown/other, multiple candidates where appropriate, evidence/provenance, and a confidence state. It is tested separately from PBR-map quality.

### PBR recovery tests

Synthetic ground-truth fixtures provide known base color/albedo, roughness, metallic, and bump/normal data where supported. Tests compare predicted maps/properties with task-appropriate numeric metrics and include novel-light rerender checks so that lighting baked into albedo is exposed.

An adapter is evaluated only for channels it claims to produce. Unsupported channels must remain absent/unknown.

### Evidence-fusion tests

Fusion tests verify deterministic handling of compatible evidence, conflicting evidence, incomparable confidence scales, and user-confirmed overrides. Raw scores from unrelated adapters must not be ranked against each other unless the test includes an explicit calibration mapping.

### Correction tests

Corrections survive a stage re-run only when their target identity remains valid. A topology-changing re-run must move incompatible corrections to an orphaned/pending-remap state.

### Workflow resumability and cache tests

A late-stage failure must preserve successful earlier artifacts. Re-running with identical semantic inputs may reuse the stage cache; changing weights, parameters, topology, adapter revision, or relevant runtime identity must invalidate the affected cache entry and downstream dependents.

### Export tests

The final GLB/glTF must open in an independent validator/viewer. Geometry/material structure represented in glTF must survive round trip. The sidecar must reference exported geometry/material identities correctly and preserve semantics, assertions, evidence kind, confidence state, provenance, and correction status.

### Coordinate, unit, and PBR interchange tests

Round-trip fixtures verify that export normalization preserves scale, orientation, transforms, UV references, metallic/roughness values, color-space interpretation, and tangent-space normal convention. A fixture with a known non-glTF source basis must prove that the conversion and inverse interpretation are reproducible from provenance.

### Label normalization tests

Open-vocabulary labels must round-trip without destructive normalization. When a normalized label or ontology identifier is supplied, it remains an additional assertion field rather than replacing the source label.

### Golden fixtures

The initial fixture set includes a simple single-material object; a multi-part single-material object; a single-part multi-material object; an object containing plastic, rubber, metal, and painted surfaces; an ambiguous part boundary; an ambiguous material boundary; a low-confidence/unknown semantic case; an imported mesh with known part/material truth; a synthetic asset with known PBR ground truth; a topology-changing re-run fixture; and a multi-view fixture with known camera metadata when named observation ports are available.

Each fixture exists to answer a specific contract or quality question, not merely to look impressive.

### AMD equivalence and performance testing

Every reference adapter receives a compatibility report containing target hardware/software versions, peak VRAM and host RAM, first-run and warm latency, Torch-MIGraphX acceptance, MIGraphX correctness/tolerance, modules left on PyTorch ROCm and why, custom HIP/native work, material CPU fallbacks, output-quality comparison to a known-good path where available, and unresolved blockers.

Quality/performance tolerances are declared per adapter before an optimization is accepted. “Looks close enough” is not an acceptance criterion.

## Out of Scope

- Training a new foundation 3D generation model for the first POC.
- Replacing Modly's workflow editor or Electron/Three.js viewer as a prerequisite.
- Requiring Vulkan for the POC viewport.
- Hard-forking Modly into a separate product unless an essential concept cannot be represented through additive host changes.
- Requiring ONNX as the primary PyTorch-to-MIGraphX route.
- Assuming automatic graph partition fallback between MIGraphX and PyTorch.
- Rewriting entire PyTorch models in C++ or HIP.
- Porting every CUDA-based 3D project before the reference pipeline works.
- Keeping every model resident in VRAM simultaneously.
- Making X-Part a mandatory dependency of part segmentation.
- Treating Material Anything as the material-region segmenter or semantic material classifier without evidence.
- Full automatic retopology suitable for every production-art pipeline.
- Rigging and animation in the first vertical slice.
- Reliable inference of engineering properties such as density, strength, elasticity, or thermal behavior in the first vertical slice.
- A universal semantic/material ontology across all object classes.
- Learned multi-model arbitration in the first milestone.
- Cloud APIs as required dependencies.
- A production-grade extension marketplace, sandbox, or package-distribution UX.
- Final legal/distribution policy decisions for every third-party dependency.

## Further Notes

### Canonical domain language

**Observation** — An immutable source image, image set, video frame set, scan, or other evidence from which geometry or properties may be inferred.

**Structured Asset** — The canonical Modly-side representation combining geometry identity, coordinate conventions, part structure, semantics, material regions, material/PBR assertions, evidence, provenance, corrections, and stage artifacts.

**Topology Revision** — An identity for a specific mesh topology to which face/vertex/region mappings are bound.

**Part Segment** — A geometry-based region hypothesized to represent a coherent object part. It does not inherently imply a semantic name or material identity.

**Semantic Label Assertion** — A candidate human-meaningful description of a part segment plus evidence kind, confidence state, and provenance.

**Material Region** — A surface region believed to share a material assignment. Material regions are independent of part segments.

**Material Identity Assertion** — A candidate semantic classification of a material, such as rubber, painted steel, anodized aluminum, ABS, glass, or fabric.

**PBR Property Assertion** — A render-oriented property such as base color/albedo, roughness, metallic, normal/bump/detail, opacity, or emissive data, with its representation and evidence explicitly recorded.

**Physical Property Assertion** — A future evidence-backed property with units/ranges such as density, hardness, elasticity, thermal conductivity, or refractive index. Not required for the first POC.

**Evidence Kind** — Whether an assertion is observed, deterministically derived, model-inferred, or user-confirmed.

**Confidence State** — The uncertainty representation attached to an inference. It may contain a numeric score with calibration metadata or explicitly state unknown/unavailable.

**Provenance** — Reproduction metadata describing source artifacts, adapter/model/weights identity, parameters, runtime/backend, device, and run/stage identity.

**Capability** — A vendor-neutral operation an adapter can provide to a Modly workflow.

**Adapter** — A concrete implementation satisfying a capability interface at a seam.

**AMD Runtime** — The deep module that hides AMD device detection, Torch-MIGraphX compilation, PyTorch ROCm fallback, HIP primitive access, profiling, memory lifecycle, and execution diagnostics.

### Capability vocabulary

The initial capability vocabulary is:

- Acquire Observations;
- Generate Geometry;
- Import Geometry;
- Segment Parts;
- Generate/Refine Parts (optional, e.g. X-Part);
- Identify Part Semantics;
- Segment Material Regions;
- Classify Material Identity;
- Estimate PBR Properties;
- Fuse Evidence;
- Validate Structured Asset;
- Repair Geometry;
- Export Structured Asset.

Future physical-property inference extends the assertion family without changing the existing material-region or PBR capability semantics.

### Candidate probes for unresolved adapter choices

The architecture is fixed; several implementation choices intentionally remain probe-driven because target-hardware viability matters more than brand preference.

For `Segment Material Regions`, the first probe compares at least one dedicated 3D material-segmentation path such as MaterialSeg3D with a multi-view SAM2-style projection/back-projection strategy similar to the material-grouping approach used in Phys4DGen. MaterialSeg3D is older and CUDA-heavy, so it is evidence that the capability exists rather than a guaranteed reference choice. The selected adapter must fit the AMD/runtime and quality gates in this spec.

For `Identify Part Semantics` and `Classify Material Identity`, the POC may use rendered multi-view crops/features plus an open vision-language or embedding model. The chosen model must support explicit unknown/ambiguous output and must not be coupled to the geometry segmenter.

### Verified external assumptions as of 2026-09-24

- Modly v0.4.x supports external model and process extensions, workflows, canonical headless workflow/process commands, and an Electron/Three.js viewer. This specification therefore preserves Modly as the host instead of replacing those systems.
- Modly's stable extension contracts are still evolving; stable named multi-image model input ports have been discussed as an additive contract and must be verified before the semantic pipeline relies on them.
- Hunyuan3D-Part explicitly separates P3-SAM (native-3D part segmentation) from X-Part (part generation/decomposition). The public X-Part release is described as a light version; P3-SAM can accept arbitrary input meshes.
- Material Anything documents generation of albedo, roughness, metallic, and bump maps and uses Blender-based rendering/preprocessing in its released pipeline. It is therefore treated as a PBR adapter, not automatically as material-region segmentation or semantic classification.
- Current MIGraphX documentation supports Torch-MIGraphX integration through PyTorch `torch.compile`, so ONNX is not required as the normal PyTorch-to-MIGraphX path.

Reference sources:

- https://github.com/lightningpixel/modly
- https://github.com/lightningpixel/modly/issues/228
- https://github.com/Tencent-Hunyuan/Hunyuan3D-Part
- https://github.com/3DTopia/MaterialAnything
- https://github.com/PROPHETE-pro/MaterialSeg3D
- https://github.com/JiajingLin/Phys4DGen
- https://rocm.docs.amd.com/projects/AMDMIGraphX/en/develop/install/install-torch-migraphx.html

### Architectural seam decision

The primary seam remains the Modly workflow/extension execution contract. The secondary internal seam is the AMD Runtime. Additional seams are introduced only when concrete implementations genuinely vary there.

The POC explicitly permits a small additive Modly host change for structured artifacts or richer I/O if the current extension contract cannot carry the required data. That is preferable to smuggling structured state through arbitrary strings or vendor-specific parameters.

### Hardware viability gates

A reference adapter is accepted on the RX 7900 GRE only when:

1. it completes its declared reference fixture without an NVIDIA runtime;
2. peak VRAM fits the target or an explicit supported low-memory/offload path fits it;
3. output passes adapter-specific correctness/quality thresholds;
4. repeated execution does not leak enough GPU memory to break the next stage;
5. native dependencies are reproducibly buildable/installable for the target environment;
6. any CPU fallback with material latency impact is disclosed.

Failure of one candidate adapter does not fail the architecture; it forces selection of another adapter for that capability.

### Proof-of-concept acceptance gate

The POC is complete when a canonical Modly headless workflow can, on the RX 7900 GRE without an NVIDIA runtime:

1. accept at least one observation or an existing mesh;
2. generate or ingest a valid geometry artifact;
3. produce native-3D part segmentation with topology-bound mappings;
4. assign semantic labels to a useful subset of segments while supporting unknown/ambiguous output;
5. produce material-region assignments independently of part segmentation;
6. produce material-identity candidates for at least a useful subset of regions;
7. produce supported PBR estimates without fabricating unsupported channels;
8. attach evidence kind, confidence state, and reproducible provenance to derived assertions;
9. preserve intermediate artifacts and correct dependency invalidation for targeted re-runs;
10. export a valid GLB/glTF plus a versioned structured sidecar;
11. report the backend, peak VRAM, and latency of every heavy stage;
12. execute at least one representative dense neural module through Torch-MIGraphX successfully, while allowing other modules to remain on PyTorch ROCm when justified;
13. complete all accepted reference adapters within the 16 GB target or their explicitly supported low-memory modes;
14. preserve legacy Modly generation behavior covered by regression tests.

### Initial implementation sequence

**Tracer 0 — Host contract probe**  
Map the exact current Modly model/process manifest, artifact registry, workflow I/O types, and headless execution contract. Implement only the smallest additive host changes required for a versioned Structured Asset reference and/or richer artifact output. Preserve legacy extensions.

**Tracer 1 — AMD Runtime proof**  
Compile representative dense PyTorch modules through Torch-MIGraphX on `gfx1100`, validate results against PyTorch ROCm, exercise explicit module-level fallback, record telemetry, and prove deterministic cleanup between sequential heavy loads.

**Tracer 2 — Geometry**  
Probe Hunyuan3D 2 Mini/Turbo and the strongest alternative that plausibly fits 16 GB. Select the reference generator using AMD compatibility, VRAM, quality, and porting cost. Run it through the Modly extension seam and emit a canonical geometry artifact plus provenance.

**Tracer 3 — Native-3D part segmentation**  
Probe P3-SAM on AMD and the 16 GB target. If viable, integrate it as `Segment Parts`; if not, select the best alternate segmenter without changing the Structured Asset contract. Keep X-Part optional.

**Tracer 4 — Semantics and material regions**  
Integrate a semantic resolver and a separate material-region segmentation strategy. Prove that a semantic part can contain multiple material regions and that one material can span multiple parts.

**Tracer 5 — Material identity and PBR recovery**  
Add material classification plus Material Anything or the selected PBR estimator. Preserve only channels the adapter actually produces, with evidence/provenance and synthetic-ground-truth validation.

**Tracer 6 — Evidence, corrections, resumability, export**  
Implement deterministic fusion/conflict behavior, correction precedence/orphaning, content-addressed stage caching, full sidecar serialization, independent GLB validation, and the backend compatibility report.

**Tracer 7 — Interactive workflow polish**  
Expose proven capabilities in Modly's existing visual workflow/viewer, display ambiguity/confidence/provenance, and add targeted re-run/correction UX without replacing the POC renderer.

### Primary risks and mitigations

**P3-SAM may not fit 16 GB.** Treat fit as a hard probe, not an assumption. Use precision/offload/staged execution only when quality remains acceptable; otherwise select another segmenter.

**Torch-MIGraphX coverage may be incomplete.** Compile at independently testable module seams, keep PyTorch ROCm as the explicit fallback, and upstream missing operators/optimizations only where the payoff is justified.

**Material Anything may bring Blender/PyTorch3D and other heavy dependencies.** Isolate them in the adapter environment, measure resource cost, and replace the adapter if the dependency burden exceeds its quality value.

**Material-region segmentation has no locked reference adapter yet.** Treat MaterialSeg3D and multi-view SAM2-style grouping as probe candidates; select based on AMD viability, quality, and how cleanly the result maps back to mesh topology.

**Current Modly I/O types may be too narrow for Structured Assets.** Add a minimal versioned artifact descriptor or sidecar registration path rather than encoding structured state into arbitrary strings or vendor-specific parameters.

**Single-image generation invents unseen geometry.** Provenance distinguishes inferred geometry from observed source evidence; multi-view acquisition is an additive capability when host input ports and adapters support it.

**Upstream mutable dependencies threaten reproducibility.** Pin adapter revisions, model-weight digests, and runtime compatibility sets in each stage provenance record.

### Success metric

The project succeeds when Modly can treat modern 3D generation, segmentation, semantic, material, and PBR systems as replaceable tools inside one durable Structured Asset pipeline, while the shared AMD Runtime makes `gfx1100` a first-class local execution target and preserves correctness, provenance, resumability, and user control.