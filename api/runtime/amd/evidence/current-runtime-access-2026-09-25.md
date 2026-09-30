# Current task AMD runtime access check — 2026-09-25

This is an environment snapshot from the current task invocation. It is not a
replacement for the historical RX 7900 GRE/Ticket 02 execution evidence, and it
does not change any ticket acceptance state.

## Commands and observed results

From the project root (`/mnt/workdrive/modly-seg-tool-osp`):

```sh
ls -l /dev/kfd /dev/dri
```

Result: both paths are absent (`No such file or directory`, exit 2). This task
container cannot access ROCm device nodes.

```sh
scripts/modly-amd-runtime.sh status
```

Result: exits 1 before listing images because the script's `chmod 700
.modly-amd-runtime/run` receives `Read-only file system`. The injected project
runroot is read-only in this invocation.

```sh
.modly-amd-runtime/api-test-venv/bin/python - <<'PY'
import torch
PY
```

Result: `ModuleNotFoundError: No module named 'torch'`. The API test virtualenv
does not provide a host-side ROCm runtime.

## Impact

The focused AMD runtime unit suite validates the fail-closed behavior and
telemetry shape with an isolated fake Torch implementation. This task
invocation cannot produce a real-GPU smoke result for
`AMDInferenceRuntime.profile_stage`; no candidate inference or ticket acceptance
run was attempted. Historical Ticket 02 evidence remains the accepted source
for the project-owned RX 7900 GRE container and MIGraphX path; consult
`../LOCK.md` and the associated probe artifacts for its pinned image/runtime
identities and measured results.

The smallest next step for real profiler validation is to rerun the helper
inside the already-built project AMD image with `/dev/kfd` and `/dev/dri`
exposed, writing the returned profile and its run log under the project-owned
`.modly-amd-runtime/results/` directory. No dependency-ready downstream ticket
was advanced by this check.
