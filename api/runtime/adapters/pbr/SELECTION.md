# PBR estimator selection and synthetic acceptance gates

This rubric is frozen before any Ticket 08 candidate is installed, executed, or
compared. Upstream quality claims are screening evidence only. The first
candidate is Material Anything; any alternative must preserve the
`Estimate PBR Properties` capability and pass these same fixtures on the
project RX 7900 GRE runtime.

## Fixed synthetic fixture and inputs

The fixture is a UV-mapped GLB with one mesh, one connected part, and three
independent material regions: a dielectric polymer, a painted dielectric metal,
and a bare conductor. The GLB carries no material slots or texture maps. The
fixture archive holds deterministic 256×256 linear-space ground-truth maps and
four calibrated orthographic views rendered by the same fixed CPU GGX reference
renderer. Each training view uses the same three declared directional-light
sources from a different calibrated camera pose; one separate directional-light
environment is held out for the novel-light render. The geometry, UVs, view and
surface masks, camera matrices, light parameters, seed, and every ground-truth
texture are versioned and SHA-256 identified in test evidence. The fixture also
has a known scalar height/bump field. It has no normals texture unless a
candidate explicitly claims a normal output.

The declared observation illumination and view transforms are inputs. A
candidate may not read the ground-truth maps, fixture IDs, filenames that
encode the material answer, glTF material slots, or hidden render metadata.
Only regions visible in the supplied source views are scored.

## Hard gates

1. Execute without an NVIDIA driver, CUDA runtime/toolkit, CUDA-only binary,
   hosted service, or required cloud account. The measured peak allocated plus
   reserved VRAM is at most **14 GiB** on the 16 GiB RX 7900 GRE; a CPU-only
   stage must report zero accelerator use and measured host memory/runtime.
2. Every emitted property is a supported assertion bound to a current
   topology revision and material-region ID. Unsupported channels remain
   absent/unknown. Preserve the source maps/observations, calibrated-view
   inputs, exact weights/model identities, runtime/backend, parameters, seed,
   and stage/run provenance.
3. On the fixed visible texels, claimed channels meet all relevant thresholds:
   - base color/albedo: linear-RGB masked MAE ≤ **0.08** and masked SSIM ≥
     **0.85**;
   - roughness: normalized masked MAE ≤ **0.10**;
   - metallic: normalized masked MAE ≤ **0.10** and conductor/dielectric
     region mean absolute bias ≤ **0.08**;
   - scalar bump/height, if claimed: normalized masked MAE ≤ **0.10**;
   - tangent-space normal, if claimed: mean angular error ≤ **12°** and
     hemisphere validity ≥ **99.9%**.
4. The held-out novel-light render uses the estimated properties on the same
   fixed geometry/UVs and must achieve masked linear-RGB MAE ≤ **0.08** against
   the ground-truth PBR render. The pinned renderer implements direct-light
   GGX, with fixed orthographic cameras, normalized linear-sRGB radiance,
   background, and no tone mapping. The renderer inputs and reference output
   are retained as evidence; no input-light bake may be relabeled as albedo.
5. Candidate modules and render/preprocessing dependencies are isolated
   inside the project adapter/runtime. A candidate that cannot reach AMD
   execution or its quality/resource gates is rejected; no threshold is
   relaxed after measurements.

## Candidate comparison order

1. Record immutable repository/checkpoint revisions and hashes, license,
   dependency/native-library requirements, output-channel semantics, and
   target-hardware import result.
2. Material Anything is the required first candidate. Check its Blender,
   PyTorch3D, renderer, model, and weight seams. If the dependency gate is
   technically passable, run it on the frozen fixture on the target and record
   first-run/warm latency, host RAM, peak VRAM, output-channel metrics, and
   novel-light error. Do not claim unmeasured metrics when a preceding hard
   dependency gate fails.
3. Probe at least one plausible alternative when Material Anything fails a
   gate. Candidate alternatives currently include SuperMat (single/multi-view
   PBR estimation) and any stronger source-and-weight-pinned implementation
   that can run in the project ROCm image. CUDA-oriented launch instructions
   are not themselves proof of AMD incompatibility; imports, native
   dependencies, and execution are measured where feasible.
4. Reject every candidate failing a hard gate. Among passing candidates,
   prefer lower novel-light MAE, then lower aggregate normalized map error;
   differences below 0.01 MAE are ties. Next prefer lower peak VRAM (within
   512 MiB is a tie), then lower warm latency (within 10% is a tie), then
   lower AMD-specific porting and isolated preprocessing cost.
5. If no candidate passes, leave Ticket 08 blocked and report measured
   failures; do not substitute constants, material labels, synthetic success,
   or a placeholder estimator.

## Channel semantics

Albedo/base color is linear reflectance, not a source-light image. Roughness
and metallic remain separate scalar channels. A candidate's map named “bump”
is stored and validated as bump/height only after its encoding and semantics
are confirmed; it is not treated as a tangent-space normal. Normal data is
stored as a vector field only when the estimator actually emits a normal map
and its tangent basis/handedness are known. Unsupported opacity/emissive or
other channels stay unknown.

## Probe status

No candidate has been downloaded or run against this rubric yet. This document
is the pre-comparison threshold record; the candidate table and decision must
be added only after measured evidence exists.

### Material Anything pre-probe dependency screen

The official repository's inference instructions require PyTorch3D, recommend
Blender 3.2.2, and list `bpy` and `triton` in its requirements. Its documented
pipeline generates albedo, roughness, metallic, and bump maps. These findings
identify dependencies that must be inspected inside the project AMD runtime;
they do not establish AMD incompatibility or any quality/resource result. The
repository and checkpoint revisions, exact checkpoint hashes, license details,
and target import/inference outcome are still unrecorded. Resolve those items
before the target probe and do not infer support from upstream CUDA launch
instructions.

Sources checked 2026-09-24: [official Material Anything repository](https://github.com/3DTopia/MaterialAnything)
and its [requirements](https://github.com/3DTopia/MaterialAnything/blob/main/requirements.txt).
