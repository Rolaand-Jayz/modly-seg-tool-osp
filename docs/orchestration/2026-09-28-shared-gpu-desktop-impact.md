# Shared GPU desktop impact during the SUV segmentation run

## Observed incident

The user reported that the Modly GPU workload affected other GPU work,
including desktop rendering, and asked that all task processes stop. The
GeoSAM2 SUV run `0d0cd967-f220-4a02-b348-89c06a92ca45` was interrupted.
Its workflow log reported a PyTorch ROCm `OutOfMemoryError`: one attempted
allocation was 15.05 GiB on a GPU with 15.98 GiB total and 10.69 GiB free;
PyTorch reported 4.44 GiB already allocated. The run had rendered twelve
views and created topology-bound correspondence, but emitted no accepted
segmentation result. The pipeline exited 130 after interruption.

The launcher's GPU container uses host `/dev/kfd` and all of `/dev/dri`, plus
`--ipc=host`. Host inspection mapped `/dev/dri/card1` and `renderD129` to the
RX 7900 GRE. `fuser` found Xwayland and desktop applications using that same
card/render node. This is a direct shared-device path, not evidence that the
container escaped its process namespace. It is sufficient to explain how
compute and graphics can contend; it does not by itself prove which low-level
event caused each desktop symptom. The inspected kernel messages did not show
an AMD GPU reset around this run.

After the stop, project Podman `ps` returned no running container, no task
process matched the host process check, and `rocm-smi --showpids` returned no
KFD process. The RX 7900 GRE still had about 2.84 GB of VRAM in use by other
desktop applications at inspection time. No unrelated process was killed.

## Immediate guard

`.modly-amd-runtime/GPU_RUNS_PAUSED` now blocks every GPU-running command in
`scripts/modly-amd-runtime.sh` before it starts a container. The reference
geometry and part-segmentation process extensions also return
`AMD_GPU_RUNS_PAUSED` before loading a model when the marker exists in their
workspace. `bash -n` passed; the launcher returned 78 with the pause message;
both workflow entry points emitted the expected error code from a minimal
CPU-only request; and their Python files compiled. These checks did not start
GPU work. The marker remains in place.

## Remaining engineering boundary

The RX 7900 GRE is shared with desktop graphics. A Podman host-memory limit or
private IPC namespace would not, on its own, reserve VRAM or compute time for
the compositor. Before removing the pause marker, the AMD workflow needs a
measured low-memory and scheduling strategy that preserves desktop usability,
plus a review of the broad `/dev/dri`, host IPC, ptrace, and unconfined seccomp
flags. The audited RX 7900 GRE POC gates remain open; partial SUV output is not
acceptance evidence.

## Game impact report and read-only follow-up

The user reported that Golf With Your Friends became unusable and required a
reinstall. This is a reported impact, not a proven disk overwrite. The game
resided in `/home/rolaandjayz/SteamLibrary`, outside the Modly container's
writable project-workspace bind mount. The Modly run's retained output paths
were under its unique `StructuredAssets/runs/` directory and run logs; no
cross-project write was identified in those inspected paths. This does not
rule out graphics-resource disruption or every possible file-system path.

Steam's local logs show the native Linux game process ended at 21:34, a later
native launch ended at 21:43, and Steam changed the installed depot from the
Linux one to the Windows one at 21:44 before downloading/staging about 5.7 GB
and verifying it at 21:46. That log sequence documents the reinstall; it
does not determine why the game was unusable beforehand. Kernel logs in the
inspected window show an AMD GPU VM-stat warning when the game process ended,
but no logged GPU reset or fault. No game file, Steam setting, or process was
modified by this investigation.

The failed SUV trace recorded 59 accepted proposals on view 0, 70 on view 1,
and 27 on view 2, then no completed view 3 proposal/lift. The exact operator
behind the 15.05 GiB allocation is not in the saved failure record. Pinned
GeoSAM2 exposes `offload_video_to_cpu` and `offload_state_to_cpu` in its video
predictor; the failed run used neither. They are now an opt-in low-memory
candidate, but have not yet been proved output-equivalent or safe for desktop
use.

The retained container stderr ends after the view-2 postprocessed export; its
next line is not a Python traceback. The proposal audit has the OOM type and
allocator message but no call frames. The pinned seed loop proceeds to a
view-3 automatic proposal call after view 2, but the saved record cannot prove
which internal operation made the 15.05 GiB request. The adapter now stores
the last 24 Python call frames (file, line, function only) in the existing
failure audit for a future authorized run. A CPU-only simulated failure check
passed 8/8 focused audit tests, and Python compilation passed. It did not run
the model or remove the GPU pause.

## CPU-only containment work after the report

The geometry and GeoSAM2 adapters now apply a live-free-memory policy before
model load: 4 GiB is held back from the observed free memory, the PyTorch
caching allocator is capped at the smaller of the remainder and 14 GiB, and
stages fail when less than 2 GiB of allocator headroom would remain. The policy
and scope are described in `api/runtime/amd/RUNTIME.md`. It does not cap
non-PyTorch GPU allocation or reserve future memory for another process.

GeoSAM2 can now opt into its pinned predictor's native video and state CPU
offload with `MODLY_GEOSAM2_CPU_OFFLOAD=1`. The adapter records the candidate
setting in its audit and rejects a returned predictor state that did not honor
both flags. The project launcher forwards the flag. CPU-only budget, offload,
and focused adapter checks passed 23/23; Python compilation and launcher
syntax checks passed. No game files were changed and no GPU workload was
started for this work. The pause marker remains in place. A representative
shared-GPU run and segmentation-quality comparison are still required before
these controls can be accepted as a fix.

CPU-only verification commands from the project root:

```sh
python -m unittest api.tests.test_amd_gpu_budget api.tests.test_geosam2_cpu_offload api.tests.test_ticket04_geosam2_adapter api.tests.test_ticket04_geosam2_diagnostic_audit_return -v
python -m py_compile api/runtime/adapters/parts/geosam2.py api/runtime/adapters/parts/geosam2_cpu_offload.py api/runtime/amd/gpu_budget.py api/runtime/adapters/geometry/runner.py
bash -n scripts/modly-amd-runtime.sh
scripts/modly-amd-runtime.sh probe
```

The first three commands exited zero; the last exited 78 with the expected
pause refusal before starting a container. An initial Ticket 03 test command
could not load because putting `api` ahead of installed packages exposed its
local `typing_extensions.py` to Pydantic. Preloading the installed module and
then adding `api` to the import path resolved that test-harness conflict; the
Ticket 03 and GeoSAM2 focused regression set passed 18/18 CPU-only tests.

The part-segmentation output path was also hardened. Both P3-SAM and GeoSAM2
create run directories through directory handles that reject symlink redirects,
reject an existing run ID, turn arbitrary schema-valid asset IDs into safe
filename components, and publish sidecars only if the final name is new.
These changes address potential cross-asset writes in the adapter; they do not
establish that Modly caused the reported game damage. The six focused path
checks and ten budget/offload checks passed 16/16 after integration, with
Python compilation passing. No Steam files or GPU workloads were touched.

## Direct process-extension pause boundary

A later entry-point audit found that direct process-extension calls checked a
marker only in the input workspace. Geometry and part segmentation now check
both their source project root and the input workspace before importing an
adapter. The launcher now permits only `build`, `build-geosam2`, and `status`
while the project marker exists; an unlisted future command fails closed.
The project/workspace marker cases across both processors and eight focused
GeoSAM2 audit checks passed 11/11 CPU-only tests. Processor compilation and
launcher `bash -n` passed; invoking an unlisted command returned 78 before
Podman or GPU access. This does not make the marker a hardware isolation
boundary, but it closes the identified Modly entry-point bypass.
Separate CPU-only subprocess calls with a temporary external workspace
confirmed that both registered processors return `AMD_GPU_RUNS_PAUSED` from
their real JSON-lines entry points before loading a model.

Ticket 07's material-identity process also used a shared Structured Asset
filename with replacing publication. It now writes each result under its
unique run ID and refuses an existing final filename or redirected output
parent. Its five focused output tests and existing CPU contract suite passed
(the latter with one optional real-model integration skip). This is another
preventive cross-run write correction, not evidence that this process ran
during or caused the game incident.
