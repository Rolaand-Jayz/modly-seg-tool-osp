# Project AMD runtime

The AMD runtime is built and stored locally under `.modly-amd-runtime/`. The
entry point always passes this project-specific Podman root and runroot, so it
does not use another Podman image store. It uses the pinned AMD PyTorch base
image and compiles MIGraphX for `gfx1100`.

Requirements: Linux with Podman, an AMD ROCm-compatible GPU exposed as
`/dev/kfd` and `/dev/dri`, and membership in the `video` group. Container GPU
access uses those device nodes. The build downloads pinned source repositories
and AMD's ROCm development wheel; the full transitive build inputs are inventoried
but are not yet hash-locked. See [`LOCK.md`](LOCK.md).

From the project root:

```sh
scripts/modly-amd-runtime.sh build
scripts/modly-amd-runtime.sh status
scripts/modly-amd-runtime.sh probe
scripts/modly-amd-runtime.sh memory
```

`build` creates `localhost/modly-amd-migraphx:ticket02` in the project-local
store. `status` prints images in that store. `probe` runs Modly's shared
`AMDInferenceRuntime` on dense inference, checks numerical agreement with
eager PyTorch ROCm, and exercises the explicit fallback policy. `memory` runs
two regions sequentially and calls `release_region` between them. JSON output
and detailed diagnostics are written under the ignored local directory
`.modly-amd-runtime/results/`.

The probe uses deterministic synthetic model weights and input tensors. It
validates the adapter/device path and runtime policy; it does not qualify any
production segmentation model or its weights. Full semantic 3D ticket and
release acceptance remains governed by the audited tickets and progress state.

## Shared display GPU budget

The geometry and GeoSAM2 adapters read live free and total VRAM before model
load. They reserve 4 GiB of the *currently free* memory for the display and
other applications, cap their own PyTorch caching allocator at the smaller of
the remaining free memory and 14 GiB, and refuse to start a stage if less than
2 GiB remains for that allocator. A missing or rejected allocator control
fails closed. GeoSAM2 records the chosen limit in `gpu-budget.json` and its
provenance; geometry records it in generation parameters.

This is an allocator limit, not a VRAM partition. Other ROCm allocations,
graphics use, later changes in another application's demand, and GPU compute
time are outside its guarantee. It has only CPU-side contract checks so far;
the project GPU pause marker remains in force after the desktop and game
impact report. Do not describe this policy as shared-GPU coexistence proof.

The pinned GeoSAM2 predictor also supports storing video frames and prediction
state in system RAM. Set `MODLY_GEOSAM2_CPU_OFFLOAD=1` on the workflow launcher
to enable both upstream controls for a candidate run. The adapter verifies that
the returned state reports both controls active, records the choice in the
proposal audit, and restores the predictor after the run. This candidate has
CPU-only wrapper tests; it has no RX 7900 GRE quality, memory, or desktop
coexistence acceptance evidence yet. The pause marker still blocks the
launcher even when the flag is set.
