# Real ROCm stage profiler smoke — 2026-09-25

## Scope

This smoke validates the shared `AMDInferenceRuntime.profile_stage` API using
one synthetic FP16 matrix multiplication on the RX 7900 GRE. It is runtime
telemetry evidence only: no model, candidate adapter, model weights, or
candidate inference was involved. It does not change Ticket 02 status or claim
any later ticket acceptance.

## Run

The existing project-owned `localhost/modly-amd-migraphx:ticket02` image was
run with `/dev/kfd` and `/dev/dri` exposed. The test script is
`.modly-amd-runtime/results/profile-stage-smoke.py`; its outputs are
`.modly-amd-runtime/results/profile-stage-smoke.json` and
`.modly-amd-runtime/results/profile-stage-smoke-details.jsonl`.

Exact successful command, run from the project root:

```sh
podman --root /mnt/workdrive/modly-seg-tool-osp/.modly-amd-runtime/storage \
  --runroot /mnt/workdrive/modly-seg-tool-osp/.modly-amd-runtime/run run --rm \
  --userns=host --device /dev/kfd --device /dev/dri --group-add video \
  --ipc=host --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  --volume /mnt/workdrive/modly-seg-tool-osp/api:/modly/api:ro \
  --volume /mnt/workdrive/modly-seg-tool-osp/.modly-amd-runtime/results:/results:rw \
  --env PYTHONPATH=/modly/api:/opt/rocm/lib \
  --env MODLY_AMD_RUNTIME_LOG=/results/profile-stage-smoke-details.jsonl \
  localhost/modly-amd-migraphx:ticket02 python /results/profile-stage-smoke.py
```

Exit code: **0**. The profile reported status `passed`, exactly one closure
invocation, finite FP16 output with shape `1024×1024`, backend `pytorch_rocm`,
and device `AMD Radeon RX 7900 GRE (gfx1100, index=0)`. Captured runtime
versions: Python `3.12.3`, PyTorch `2.11.0+rocm7.14.0`, ROCm/HIP `7.14.60850`,
MIGraphX `2.16.0.dev+20250912-17-575-g4bcfe75b2`, and Torch-MIGraphX `1.2`.
Measured latency was `1181.6702369978884 ms`; peak allocated VRAM was
`39,845,888` bytes and peak reserved VRAM was `54,525,952` bytes. Run ID:
`c1e1c93d-01c8-4648-aa3f-01727c257587`.

The first attempt used relative Podman root/runroot arguments and exited 125
before container start because the store records an absolute static path. The
successful command above used those same project paths as absolute arguments;
no image was rebuilt and no global container store was used.

## Artifact hashes

The SHA-256 values, independently recomputed after the run, are:

| Artifact | SHA-256 |
| --- | --- |
| `.modly-amd-runtime/results/profile-stage-smoke.py` | `8d9468e74adbced729e1e3a292c716689d825a73f6ca522f7eb450ff508af436` |
| `.modly-amd-runtime/results/profile-stage-smoke.json` | `5b9582daa6bd233c6a7e013e432490cacd34921edf3f65f8b2886d2c8b2005a5` |
| `.modly-amd-runtime/results/profile-stage-smoke-details.jsonl` | `53f9baddabfe79030c87dbc8211ac059ceaf6a38f2c38a8997b7adb13666c15a` |

The ignored runtime results mount permits container-created outputs but rejects
host-side writes in this task context, so the hashes are recorded in this
tracked evidence note rather than a companion file in the result directory.
