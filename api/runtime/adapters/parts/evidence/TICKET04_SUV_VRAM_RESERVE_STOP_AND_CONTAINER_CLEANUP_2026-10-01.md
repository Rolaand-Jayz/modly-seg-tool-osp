# Ticket 04 SUV reserve stops and exact container cleanup (2026-10-01)

## Decision

The bounded GeoSAM2 box-reduction candidate remains unqualified. Two isolated
75,000-face user-SUV workflow runs were stopped when board-wide free VRAM fell
below the existing 4 GiB reserve. Neither wrote a final face-label array,
Structured Asset sidecar, or segmentation manifest, so neither was scored.
No Ticket 04 quality or resource gate changed.

The attempts exposed a cleanup gap: terminating the launcher process group,
and even receiving a stopped state from Podman, did not always stop the actual
container task. Both identified task processes were explicitly killed by their
run-specific host PID; GPU memory returned to the pre-run level. The GeoSAM2
launcher now records a unique container name and cidfile. A monitored runner
uses the full container ID to resolve the exact delegated cgroup, kills that
cgroup, and verifies that it is empty before reporting the run stopped.

## SUV input

Both attempts used an isolated copy of the same Modly-imported user SUV asset:

- Geometry SHA-256: `39eae91adc8dc42f5ab896579d666aa00c806b80ce1d9a4f0fdadcaea5dbed68`
- Topology revision: `sha256:f65ea5a3000e4b366a8628a7360c7164f8b6fa4524b4186d5ba1ce1e0ae0a057`
- 75,000 faces, 40,184 vertices
- Box reduction opt-in enabled; 32 prompts per GPU batch; all 12 views and M2M retained
- No source image or truth labels were read during these segmentation attempts

## Run 13: process-group stop was insufficient

Workspace: `.modly-amd-runtime/suv-box-reduction-run13/`.
Run ID: `d3ba1d04-6a77-4d44-8c9b-f00000000013`.
The two-second board monitor sampled 386 times over 772.2 seconds. Peak observed
board use was 12,889,935,872 bytes (12.00 GiB); minimum free memory was
4,273,156,096 bytes (3.98 GiB). It stopped the launcher when the reserve was
crossed. The container task outlived that process-group signal. After verifying
the exact run command and KFD PID, only that task PID and its Podman launcher
were force-stopped. KFD became empty and board use returned to 3,647,815,680
bytes.

Five per-view label arrays and a partial 9,707,474-byte proposal audit exist.
There is no final canonical label file, segmentation manifest, or output
Structured Asset. Monitor SHA-256:
`797f407aede7ceac6f914ad95ed932b7fc14214947b3572982f361bdcf58a3f0`.

## Run 14: Podman stopped state did not prove process exit

Workspace: `.modly-amd-runtime/suv-box-reduction-run14/`.
Run ID: `d3ba1d04-6a77-4d44-8c9b-f00000000014`.
The run used a unique container name and cidfile and the first version of the
new supervisor. Its two-second monitor sampled 401 times over 816.6 seconds.
Peak observed board use was 12,887,764,992 bytes (12.00 GiB); minimum free
memory was 4,195,817,216 bytes (3.98 GiB). Podman returned an error to the
initial exact-ID kill, then reported the container state as stopped. The task
Python process nevertheless remained active in the run's delegated cgroup and
continued consuming CPU and GPU memory. The supervisor incorrectly treated the
Podman state response as sufficient; it exited before cgroup cleanup was
added. The exact KFD PID and Podman process were then force-stopped. KFD became
empty and board use returned to 3,647,795,200 bytes.

Five per-view label arrays and a partial 9,853,960-byte proposal audit exist.
There is no final canonical label file, segmentation manifest, or output
Structured Asset. Monitor SHA-256:
`47d52c9bb68b5d7d6745ed553e3eb177d0dc2ad55febe88e3bb0cb8aa2b27cf3`.

## Cleanup correction and validation

`scripts/modly-amd-runtime.sh workflow-geosam2` now assigns the run a unique
`modly-geosam2-<run_id>` name and writes Podman's full container ID into a new
workspace-local cidfile. `scripts/monitored_geosam2_workflow.py` samples the
largest unique host DRM VRAM device every two seconds and starts only when the
4 GiB reserve plus 2 GiB start margin is available. On a reserve crossing it
reads only that run's validated 64-character container ID, resolves the
matching `libpod-<id>.scope`, writes to that cgroup's `cgroup.kill`, confirms
that `cgroup.procs` is empty, and only then reports the container stopped. It
will retry exact cleanup while the reserve is crossed instead of leaving a
possibly detached process behind.

Validation:

- Focused supervisor unit tests: **5/5 passed**.
- Python compilation and launcher shell syntax: passed.
- A detached CPU-only container was killed using the exact-ID cgroup path;
  the supervisor reported `cgroup.kill`, zero remaining processes, and
  Podman `rm --force` exit 0.
- After each SUV stop, `rocm-smi --showpids` showed no KFD process and board
  use returned to the measured pre-run baseline.

This validates the exact cleanup primitive on the project runtime. It does not
yet prove a reserve-triggered RX 7900 GRE end-to-end stop under inference, nor
does it establish box-reduction parity, quality, or Ticket 04 acceptance. The
partial workspaces and runtime monitor logs remain ignored and are excluded
from Git publication.

## Run 15: partial completion without a reserve crossing

Workspace: `.modly-amd-runtime/suv-box-reduction-run15/`.
Run ID: `d3ba1d04-6a77-4d44-8c9b-f00000000015`. This retry used the updated
exact-container supervisor and the same 75,000-face imported SUV and opt-in
settings. Its 414 board-wide samples covered 828.3 seconds. Minimum free VRAM
was 5,692,243,968 bytes (5.30 GiB), so the supervisor did not cross its 4 GiB
stop threshold. Maximum sampled board use was 11,471,826,944 bytes (10.68
GiB), and use returned to the 3,649,896,448-byte pre-run baseline after exit.
The monitor log SHA-256 is
`57c9536ca435796968baac9628a3ed9c58a725da8133d7cee69fa901951a5ca4`.

Seven of the twelve automatic view outputs were written. The workflow status
file reports `container=137` and `event_filter=1`; input preparation and stdout
capture returned zero. The monitor ended with `stop_reason: null`, so the
available evidence does not identify why the container exited with 137. It
does not support attributing this run to the board-wide VRAM reserve. No final
face-label array, segmentation manifest, scored result, or run output
Structured Asset was produced. At the post-exit check, `rocm-smi --showpids`
reported no KFD processes and board use had returned to baseline. The partial
run remains unqualified; the cause of exit 137 and a successful end-to-end
supervisor run remain unresolved. Monitor and partial output files remain
ignored and excluded from Git publication.
