# Geometry generator selection rubric

This rubric was declared before target-GPU inference or candidate comparison.
Model size and upstream claims are screening evidence only; they are not a
selection result. The measured comparison must use the same fixed input images,
seed where supported, output validation, renderer, and host environment for each
candidate.

## Hard acceptance gates

1. Run in the project's AMD ROCm environment on the RX 7900 GRE (`gfx1100`)
   without an NVIDIA driver, CUDA toolkit/runtime, or NVIDIA-only extension.
2. Generate geometry from a Modly source-observation artifact through the
   process-extension seam; write a valid glTF 2.0 binary (`.glb`) with a
   non-empty triangle surface, finite positions, finite non-degenerate bounds,
   and no partial output registered on failure.
3. Every successful result records immutable source and weight identities,
   adapter/upstream revision, runtime/backend, parameters, seed, device,
   elapsed time, and measured peak VRAM. Single-image-derived hidden geometry
   is marked model-inferred; the input image remains observed evidence.
4. Peak GPU memory must be at most 14 GiB during generation, preserving a 2 GiB
   reserve on the 16 GiB device for driver/display and host runtime overhead.
   Latency must be at most 180 seconds per 512-pixel source image after one
   cold model load; model loading time is reported separately.
5. At least three fixed source images must complete successfully. For each,
   render the generated mesh from the input camera using the same orthographic
   framing rule and compare the rendered foreground silhouette to the
   foreground mask extracted from that source. Median intersection-over-union
   must be at least 0.50 and no per-image IoU may be below 0.35. The masks,
   render settings, and computed scores are retained as evidence. This is a
   minimum image-conditioned geometry gate, not a claim of hidden-surface
   ground truth.

## Selection among passing candidates

Candidates that fail any hard gate are ineligible. Among eligible candidates,
select by this fixed ordering:

1. Higher median silhouette IoU on the same fixture set.
2. Lower peak VRAM; a difference under 512 MiB is treated as a tie.
3. Lower median per-image generation latency; a difference under 10% is a tie.
4. Lower AMD-specific porting burden, ordered as: no custom CUDA/native op;
   upstream native op with supported AMD build; adapter-local replacement or
   HIP port; upstream code rewrite.

No candidate may be selected using upstream benchmark claims, model parameter
count, or image-only aesthetic preference in place of the declared measurements.
If no candidate passes, Ticket 03 remains blocked and no generator is selected.

## Candidate screening (not target acceptance)

* Hunyuan3D 2 Mini/Turbo is the required primary candidate. Its upstream
  repository describes it as a 0.6B step-distilled shape model and lists 6 GB
  VRAM for shape generation in the Hunyuan3D 2 system. Shape-only generation
  avoids conflating geometry selection with the separate texture/material
  pipeline. The upstream application exposes a `--device` default of `cuda`,
  so the Modly adapter must explicitly route tensor execution through the
  approved ROCm PyTorch runtime and audit any custom kernels before acceptance.
* TripoSR is the plausible 16 GB alternative. Its upstream repository reports
  about 6 GB VRAM for its default single-image path and releases source and
  checkpoints under MIT. Its official troubleshooting documents a CUDA-built
  `torchmcubes` extraction dependency. The AMD probe must establish whether the
  model itself runs with ROCm and whether that custom CUDA extractor can be
  replaced with a validated AMD-compatible or CPU extraction path without
  unacceptable quality or latency loss.

Screening references: [Hunyuan3D-2 upstream README](https://github.com/Tencent-Hunyuan/Hunyuan3D-2),
[Hunyuan3D-2 inference entrypoint](https://github.com/Tencent-Hunyuan/Hunyuan3D-2/blob/main/gradio_app.py),
and [TripoSR upstream README](https://github.com/VAST-AI-Research/TripoSR).

## Probe status

**Completed on the RX 7900 GRE (`gfx1100`) through Modly's process-extension
seam.** Hunyuan3D-2 Mini/Turbo passed the unchanged gates on all three fixed
512-pixel fixtures using explicit PyTorch ROCm denoising with upstream CPU
offload. Chair, flamingo, and teapot generation took 99.367 s, 99.116 s, and
98.639 s, respectively. Sampled board peaks were 6,537,334,784 B,
6,494,683,136 B, and 6,502,207,488 B; adapter-reported model peaks were
2,609,977,344 B. Silhouette IoU was 0.778680, 0.619006, and 0.956428
(median 0.778680), satisfying the <=14 GiB, <=180 s, median >=0.50, and
per-image >=0.35 gates. Outputs were valid GLB files and Structured Assets
with pinned source/weight provenance. The measured denoiser MIGraphX route was
slower than ROCm (692.82 ms vs 418.04 ms), so the full-module ROCm route was
used consistently for all fixtures.

TripoSR completed the same fixture screen but failed the frozen quality gates:
chair/flamingo/teapot IoU was 0.33790/0.18581/0.59528 (median 0.33790), and
its MIGraphX output failed the declared parity tolerance. It is therefore
ineligible. Hunyuan is the sole candidate that passed all hard gates and is
selected under the ordering above. Full commands, source and weight identities,
per-run telemetry, output hashes, and failure-path checks are recorded in
[`03-reference-geometry-generation.md`](../../../../MODLY_AMD_SEMANTIC_3D_HANDOFF/tickets/issues/03-reference-geometry-generation.md)
and its linked per-fixture evidence directories.
