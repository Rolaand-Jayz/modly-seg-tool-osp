#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PODMAN_ROOT="$ROOT_DIR/.modly-amd-runtime/storage"
PODMAN_RUNROOT="$ROOT_DIR/.modly-amd-runtime/run"
IMAGE="localhost/modly-amd-migraphx:ticket02"
GEOSAM2_IMAGE="localhost/modly-amd-geosam2:ticket04"
mkdir -p "$PODMAN_ROOT" "$PODMAN_RUNROOT" "$ROOT_DIR/.modly-amd-runtime/results"
chmod 700 "$PODMAN_RUNROOT"
export XDG_RUNTIME_DIR="$PODMAN_RUNROOT"

# GPU containers use host AMD device nodes. A project pause must fail closed
# before any container can touch the GPU, including diagnostic subcommands.
GPU_RUN_PAUSE_FILE="$ROOT_DIR/.modly-amd-runtime/GPU_RUNS_PAUSED"
if [[ -e "$GPU_RUN_PAUSE_FILE" ]]; then
  case "${1:-}" in
    build|build-geosam2|status|budget-check)
      ;;
    *)
      printf 'AMD GPU runs are paused for this project: %s\n' "$GPU_RUN_PAUSE_FILE" >&2
      exit 78
      ;;
  esac
fi

podman_local() {
  podman --root "$PODMAN_ROOT" --runroot "$PODMAN_RUNROOT" "$@"
}

case "${1:-}" in
  build)
    podman_local build --layers=false -t "$IMAGE" \
      -f "$ROOT_DIR/api/runtime/amd/Containerfile" "$ROOT_DIR"
    ;;
  build-geosam2)
    BASE_ID="$(podman_local image inspect "$IMAGE" --format '{{.Id}}')"
    if [[ "$BASE_ID" != "c96b56796c753d749cb02b31b88da7567c3712861d214bfe26452d21bd7e6b8d" ]]; then
      printf 'Ticket 04 renderer image must be based on the accepted Ticket 02 runtime image; found %s.\n' "$BASE_ID" >&2
      exit 1
    fi
    podman_local build --layers=false -t "$GEOSAM2_IMAGE" \
      -f "$ROOT_DIR/api/runtime/amd/Containerfile.geosam2" "$ROOT_DIR"
    ;;
  status)
    podman_local images --digests
    ;;
  budget-check)
    # Read-only check that the pinned container can see host-wide VRAM counters.
    # This does not open /dev/kfd or start an inference workload.
    podman_local run --rm --network=none \
      --volume "$ROOT_DIR/api:/modly/api:ro" \
      --volume /sys:/sys:ro \
      --env PYTHONPATH=/modly/api \
      "$IMAGE" python -c '
import json
from pathlib import Path
from runtime.amd.gpu_budget import read_shared_vram_usage
records=[]
drm_class=Path("/sys/class/drm")
for card in sorted(drm_class.glob("card[0-9]*")):
    device=card/"device"
    total=device/"mem_info_vram_total"
    used=device/"mem_info_vram_used"
    if total.is_file() and used.is_file():
        records.append((int(total.read_text().strip()), int(used.read_text().strip())))
if not records:
    raise SystemExit("no host-wide DRM VRAM counters are visible")
target_total=max(total for total,_ in records)
print(json.dumps({"device_total_bytes":target_total,
                  "system_wide_used_bytes":read_shared_vram_usage(target_total),
                  "counter":"host DRM mem_info_vram_used"},sort_keys=True))
'
    ;;
  probe)
    podman_local run --rm --userns=host \
      --device /dev/kfd --device /dev/dri --group-add video --ipc=host \
      --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
      --volume "$ROOT_DIR/api:/modly/api:ro" \
      --volume "$ROOT_DIR/.modly-amd-runtime/results:/results:rw" \
      --env PYTHONPATH=/modly/api:/opt/rocm/lib \
      --env MODLY_AMD_RUNTIME_LOG=/results/amd-runtime-details.jsonl \
      "$IMAGE" python /modly/api/runtime/amd/probe_migraphx.py
    ;;
  memory)
    podman_local run --rm --userns=host \
      --device /dev/kfd --device /dev/dri --group-add video --ipc=host \
      --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
      --volume "$ROOT_DIR/api:/modly/api:ro" \
      --volume "$ROOT_DIR/.modly-amd-runtime/results:/results:rw" \
      --env PYTHONPATH=/modly/api:/opt/rocm/lib \
      --env MODLY_AMD_RUNTIME_LOG=/results/amd-runtime-details.jsonl \
    "$IMAGE" python /modly/api/runtime/amd/probe_sequential_memory.py
    ;;
  workflow-geosam2)
    if [[ $# -ne 5 ]]; then
      printf 'Usage: %s workflow-geosam2 WORKSPACE_DIR GEOMETRY_RELATIVE_PATH SIDECAR_RELATIVE_PATH RUN_ID\n' "$0" >&2
      exit 2
    fi
    WORKSPACE_DIR="$(realpath "$2")"
    GEOMETRY_RELATIVE_PATH="$3"
    SIDECAR_RELATIVE_PATH="$4"
    RUN_ID="$5"
    WORKFLOW_CONTAINER_NAME="modly-geosam2-$RUN_ID"
    WORKFLOW_CONTAINER_CIDFILE="$WORKSPACE_DIR/.${RUN_ID}.container.cid"
    if [[ -e "$WORKFLOW_CONTAINER_CIDFILE" || -L "$WORKFLOW_CONTAINER_CIDFILE" ]]; then
      printf 'Refusing to reuse GeoSAM2 container identity file: %s\n' "$WORKFLOW_CONTAINER_CIDFILE" >&2
      exit 2
    fi
    if [[ ! -d "$WORKSPACE_DIR" || ! -f "$WORKSPACE_DIR/$GEOMETRY_RELATIVE_PATH" || ! -f "$WORKSPACE_DIR/$SIDECAR_RELATIVE_PATH" ]]; then
      printf 'Workspace, geometry, and Structured Asset sidecar must exist before the workflow starts.\n' >&2
      exit 2
    fi
    if [[ ! "$RUN_ID" =~ ^[0-9a-fA-F-]{1,80}$ ]]; then
      printf 'Run ID must contain only hexadecimal characters and hyphens (maximum 80).\n' >&2
      exit 2
    fi
    if [[ "$GEOMETRY_RELATIVE_PATH" == /* || "$SIDECAR_RELATIVE_PATH" == /* || "$GEOMETRY_RELATIVE_PATH" =~ (^|/)\.\.(/|$) || "$SIDECAR_RELATIVE_PATH" =~ (^|/)\.\.(/|$) ]]; then
      printf 'Geometry and sidecar paths must be workspace-relative paths without traversal components.\n' >&2
      exit 2
    fi
    WORKFLOW_LOG_ROOT="${MODLY_AMD_WORKFLOW_LOG_DIR:-$ROOT_DIR/.modly-amd-runtime/results/workflow-geosam2}"
    mkdir -p "$WORKFLOW_LOG_ROOT"
    WORKFLOW_STDOUT_LOG="$WORKFLOW_LOG_ROOT/$RUN_ID.container.stdout.jsonl"
    WORKFLOW_CONTAINER_STDERR_LOG="$WORKFLOW_LOG_ROOT/$RUN_ID.container.stderr.log"
    WORKFLOW_INPUT_STDERR_LOG="$WORKFLOW_LOG_ROOT/$RUN_ID.input.stderr.log"
    WORKFLOW_FILTER_STDERR_LOG="$WORKFLOW_LOG_ROOT/$RUN_ID.filter.stderr.log"
    WORKFLOW_STATUS_LOG="$WORKFLOW_LOG_ROOT/$RUN_ID.pipeline-status.txt"
    for WORKFLOW_LOG in "$WORKFLOW_STDOUT_LOG" "$WORKFLOW_CONTAINER_STDERR_LOG" \
                        "$WORKFLOW_INPUT_STDERR_LOG" "$WORKFLOW_FILTER_STDERR_LOG" \
                        "$WORKFLOW_STATUS_LOG"; do
      if [[ -e "$WORKFLOW_LOG" ]]; then
        printf 'Refusing to overwrite workflow diagnostic log: %s\n' "$WORKFLOW_LOG" >&2
        exit 2
      fi
    done
    umask 077
    DIAGNOSTIC_ENV=()
    if [[ "${MODLY_GEOSAM2_DIAGNOSTICS:-}" == "1" ]]; then
      DIAGNOSTIC_ENV+=(--env MODLY_GEOSAM2_DIAGNOSTICS=1)
    fi
    PROMPT_SEED_LIFT_ENV=()
    if [[ "${MODLY_GEOSAM2_PROMPT_SEED_LIFT:-}" == "1" ]]; then
      PROMPT_SEED_LIFT_ENV+=(--env MODLY_GEOSAM2_PROMPT_SEED_LIFT=1)
    fi
    FINITE_RETRY_CANDIDATE_ENV=()
    if [[ "${MODLY_GEOSAM2_FINITE_RETRY_CANDIDATE:-}" == "1" ]]; then
      FINITE_RETRY_CANDIDATE_ENV+=(--env MODLY_GEOSAM2_FINITE_RETRY_CANDIDATE=1)
    fi
    PROPOSAL_LIFT_TRACE_ENV=()
    PROPOSAL_LIFT_TRACE_FLAG="${MODLY_GEOSAM2_PROPOSAL_LIFT_TRACE:-0}"
    if [[ "$PROPOSAL_LIFT_TRACE_FLAG" != "0" && "$PROPOSAL_LIFT_TRACE_FLAG" != "1" ]]; then
      printf 'MODLY_GEOSAM2_PROPOSAL_LIFT_TRACE must be 0 or 1.\n' >&2
      exit 2
    fi
    if [[ "$PROPOSAL_LIFT_TRACE_FLAG" == "1" ]]; then
      PROPOSAL_LIFT_TRACE_ENV+=(--env MODLY_GEOSAM2_PROPOSAL_LIFT_TRACE=1)
    fi
    CPU_OFFLOAD_ENV=()
    CPU_OFFLOAD_FLAG="${MODLY_GEOSAM2_CPU_OFFLOAD:-0}"
    if [[ "$CPU_OFFLOAD_FLAG" != "0" && "$CPU_OFFLOAD_FLAG" != "1" ]]; then
      printf 'MODLY_GEOSAM2_CPU_OFFLOAD must be 0 or 1.\n' >&2
      exit 2
    fi
    if [[ "$CPU_OFFLOAD_FLAG" == "1" ]]; then
      CPU_OFFLOAD_ENV+=(--env MODLY_GEOSAM2_CPU_OFFLOAD=1)
    fi
    BOX_REDUCTION_ENV=()
    BOX_REDUCTION_FLAG="${MODLY_GEOSAM2_BOUNDED_BOX_REDUCTION:-0}"
    if [[ "$BOX_REDUCTION_FLAG" != "0" && "$BOX_REDUCTION_FLAG" != "1" ]]; then
      printf 'MODLY_GEOSAM2_BOUNDED_BOX_REDUCTION must be 0 or 1.\n' >&2
      exit 2
    fi
    if [[ "$BOX_REDUCTION_FLAG" == "1" ]]; then
      BOX_REDUCTION_ENV+=(--env MODLY_GEOSAM2_BOUNDED_BOX_REDUCTION=1)
    fi
    RETAIN_CONTAINER_FLAG="${MODLY_AMD_WORKFLOW_RETAIN_CONTAINER:-0}"
    if [[ "$RETAIN_CONTAINER_FLAG" != "0" && "$RETAIN_CONTAINER_FLAG" != "1" ]]; then
      printf 'MODLY_AMD_WORKFLOW_RETAIN_CONTAINER must be 0 or 1.\n' >&2
      exit 2
    fi
    CONTAINER_REMOVE_ARGS=(--rm)
    if [[ "$RETAIN_CONTAINER_FLAG" == "1" ]]; then
      CONTAINER_REMOVE_ARGS=()
    fi
    set +e
    python -c 'import json,sys; print(json.dumps({"workspaceDir":"/workspace","input":{"filePath":sys.argv[1],"structuredAssetPath":sys.argv[2]},"params":{"run_id":sys.argv[3],"backend":"geosam2","seed":42}}))' \
      "$GEOMETRY_RELATIVE_PATH" "$SIDECAR_RELATIVE_PATH" "$RUN_ID" \
      2>"$WORKFLOW_INPUT_STDERR_LOG" \
      | podman_local run "${CONTAINER_REMOVE_ARGS[@]}" --interactive --name "$WORKFLOW_CONTAINER_NAME" \
          --cidfile "$WORKFLOW_CONTAINER_CIDFILE" --userns=host --network=none \
          --device /dev/kfd --device /dev/dri --group-add video --ipc=host \
          --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
          --volume /sys:/sys:ro \
          --volume "$ROOT_DIR/api:/modly/api:ro" \
          --volume "$ROOT_DIR/src:/modly/src:ro" \
          --volume "$ROOT_DIR/scripts:/modly/scripts:ro" \
          --volume "$ROOT_DIR/.modly-amd-runtime/source:/modly/.modly-amd-runtime/source:ro" \
          --volume "$ROOT_DIR/.modly-amd-runtime/models/geosam2:/modly/.modly-amd-runtime/models/geosam2:ro" \
          --volume "$ROOT_DIR/.modly-amd-runtime/package-cache/geosam2:/modly/.modly-amd-runtime/package-cache/geosam2:ro" \
          --volume "$ROOT_DIR/.modly-amd-runtime/renderers/blender-4.0.2:/modly/.modly-amd-runtime/renderers/blender-4.0.2:ro" \
          --volume "$ROOT_DIR/.modly-amd-runtime/api-test-venv/lib/python3.12/site-packages:/api-site:ro" \
          --volume "$WORKSPACE_DIR:/workspace:rw" \
          --env MODLY_API_DIR=/modly/api \
          --env PYTHONPATH=/modly/api:/modly/.modly-amd-runtime/package-cache/geosam2/site-packages:/api-site:/opt/rocm/lib \
          --env PYTHONDONTWRITEBYTECODE=1 \
          --env MODLY_AMD_RUNTIME_LOG=/workspace/amd-runtime-details.jsonl \
          "${DIAGNOSTIC_ENV[@]}" \
          "${PROMPT_SEED_LIFT_ENV[@]}" \
          "${FINITE_RETRY_CANDIDATE_ENV[@]}" \
          "${PROPOSAL_LIFT_TRACE_ENV[@]}" \
          "${CPU_OFFLOAD_ENV[@]}" \
          "${BOX_REDUCTION_ENV[@]}" \
          "$GEOSAM2_IMAGE" python -c '
import os, runpy, sys
api = "/modly/api"
sys.path[:] = [entry for entry in sys.path if os.path.realpath(entry or ".") != os.path.realpath(api)]
import typing_extensions
sys.path.insert(0, api)
script = "/modly/src/areas/workflows/nodes/reference-part-segmentation/processor.py"
sys.argv = [script]
runpy.run_path(script, run_name="__main__")
' 2>"$WORKFLOW_CONTAINER_STDERR_LOG" \
      | tee "$WORKFLOW_STDOUT_LOG" \
      | python "$ROOT_DIR/scripts/modly_workflow_event_filter.py" 2>"$WORKFLOW_FILTER_STDERR_LOG"
    WORKFLOW_PIPE_STATUS=("${PIPESTATUS[@]}")
    set -e
    printf 'input_json=%s\ncontainer=%s\nstdout_capture=%s\nevent_filter=%s\n' \
      "${WORKFLOW_PIPE_STATUS[0]:-125}" "${WORKFLOW_PIPE_STATUS[1]:-125}" \
      "${WORKFLOW_PIPE_STATUS[2]:-125}" "${WORKFLOW_PIPE_STATUS[3]:-125}" \
      > "$WORKFLOW_STATUS_LOG"
    printf 'GeoSAM2 workflow pipeline statuses: input=%s container=%s log=%s filter=%s; logs: %s\n' \
      "${WORKFLOW_PIPE_STATUS[0]:-125}" "${WORKFLOW_PIPE_STATUS[1]:-125}" \
      "${WORKFLOW_PIPE_STATUS[2]:-125}" "${WORKFLOW_PIPE_STATUS[3]:-125}" \
      "$WORKFLOW_LOG_ROOT" >&2
    for ((WORKFLOW_STATUS_INDEX=${#WORKFLOW_PIPE_STATUS[@]}-1; WORKFLOW_STATUS_INDEX >= 0; WORKFLOW_STATUS_INDEX--)); do
      if (( WORKFLOW_PIPE_STATUS[WORKFLOW_STATUS_INDEX] != 0 )); then
        exit "${WORKFLOW_PIPE_STATUS[WORKFLOW_STATUS_INDEX]}"
      fi
    done
    ;;
  diagnose-geosam2-qv)
    if [[ $# -ne 5 ]]; then
      printf 'Usage: %s diagnose-geosam2-qv WORKSPACE_DIR RENDER_DIR_RELATIVE_PATH FACE_MAP_RELATIVE_PATH OUTPUT_RELATIVE_PATH\n' "$0" >&2
      exit 2
    fi
    WORKSPACE_DIR="$(realpath "$2")"
    RENDER_RELATIVE_PATH="$3"
    FACE_MAP_RELATIVE_PATH="$4"
    OUTPUT_RELATIVE_PATH="$5"
    for RELATIVE_PATH in "$RENDER_RELATIVE_PATH" "$FACE_MAP_RELATIVE_PATH" "$OUTPUT_RELATIVE_PATH"; do
      if [[ "$RELATIVE_PATH" == /* || "$RELATIVE_PATH" =~ (^|/)\.\.(/|$) || -z "$RELATIVE_PATH" ]]; then
        printf 'Render, face-map, and output paths must be workspace-relative without traversal components.\n' >&2
        exit 2
      fi
    done
    RENDER_DIR="$(realpath -e "$WORKSPACE_DIR/$RENDER_RELATIVE_PATH")"
    FACE_MAP_PATH="$(realpath -e "$WORKSPACE_DIR/$FACE_MAP_RELATIVE_PATH")"
    OUTPUT_PATH="$(realpath -m "$WORKSPACE_DIR/$OUTPUT_RELATIVE_PATH")"
    OUTPUT_PARENT="$(dirname "$OUTPUT_PATH")"
    case "$RENDER_DIR" in "$WORKSPACE_DIR"/*) ;; *) printf 'Render bundle escapes the workspace.\n' >&2; exit 2 ;; esac
    case "$FACE_MAP_PATH" in "$WORKSPACE_DIR"/*) ;; *) printf 'Face map escapes the workspace.\n' >&2; exit 2 ;; esac
    case "$OUTPUT_PATH" in "$WORKSPACE_DIR"/*) ;; *) printf 'Output path escapes the workspace.\n' >&2; exit 2 ;; esac
    if [[ ! -d "$RENDER_DIR" || ! -f "$RENDER_DIR/render_manifest.json" || ! -f "$FACE_MAP_PATH" || ! -d "$OUTPUT_PARENT" || -e "$OUTPUT_PATH" ]]; then
      printf 'Render bundle and face map must exist; output parent must exist and output file must be new.\n' >&2
      exit 2
    fi
    SOURCE_ROOT="$ROOT_DIR/.modly-amd-runtime/source/geosam2-b5de23c60ab487d407b623d394a1614f9714761c"
    CHECKPOINT="$ROOT_DIR/.modly-amd-runtime/models/geosam2/geosam2.pt"
    SOURCE_LOCK="$ROOT_DIR/api/runtime/adapters/parts/GEOSAM2_SOURCE_LOCK.json"
    MODEL_LOCK="$ROOT_DIR/api/runtime/adapters/parts/GEOSAM2_MODEL_LOCK.json"
    DEPENDENCY_LOCK="$ROOT_DIR/api/runtime/adapters/parts/GEOSAM2_DEPENDENCY_LOCK.json"
    # Podman reports `.Id` as bare hex; normalize it for evidence while using
    # the immutable bare ID as the run reference accepted by Podman.
    # shellcheck source=/dev/null
    source "$ROOT_DIR/scripts/geosam2-image-id.sh"
    GEOSAM2_IMAGE_ID_RAW="$(podman_local image inspect "$GEOSAM2_IMAGE" --format '{{.Id}}')"
    if ! GEOSAM2_IMAGE_ID="$(normalize_geosam2_image_id "$GEOSAM2_IMAGE_ID_RAW")"; then
      printf 'GeoSAM2 diagnostic image does not have an immutable local image ID.\n' >&2
      exit 2
    fi
    GEOSAM2_IMAGE_REFERENCE="$(geosam2_immutable_image_reference "$GEOSAM2_IMAGE_ID")"
    umask 077
    TEMP_OUTPUT="$(mktemp "$OUTPUT_PARENT/.geosam2-qv.XXXXXXXX")"
    cleanup_qv_temp() {
      if [[ -n "${TEMP_OUTPUT:-}" ]]; then rm -f -- "$TEMP_OUTPUT"; fi
    }
    trap cleanup_qv_temp EXIT
    trap 'cleanup_qv_temp; exit 1' INT TERM
    if ! podman_local run --rm --userns=host --network=none \
        --device /dev/kfd --device /dev/dri --group-add video --ipc=host \
        --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
        --volume "$ROOT_DIR/api:/modly/api:ro" \
        --volume "$SOURCE_ROOT:/modly/.modly-amd-runtime/source/geosam2-b5de23c60ab487d407b623d394a1614f9714761c:ro" \
        --volume "$(dirname "$CHECKPOINT"):/modly/.modly-amd-runtime/models/geosam2:ro" \
        --volume "$ROOT_DIR/.modly-amd-runtime/package-cache/geosam2:/modly/.modly-amd-runtime/package-cache/geosam2:ro" \
        --volume "$ROOT_DIR/.modly-amd-runtime/api-test-venv/lib/python3.12/site-packages:/api-site:ro" \
        --volume "$RENDER_DIR:/input/renders:ro" \
        --volume "$FACE_MAP_PATH:/input/face-correspondence.json:ro" \
        --env MODLY_API_DIR=/modly/api \
        --env MODLY_GEOSAM2_IMAGE_ID="$GEOSAM2_IMAGE_ID" \
        --env PYTHONPATH=/modly/api:/modly/.modly-amd-runtime/package-cache/geosam2/site-packages:/api-site:/opt/rocm/lib \
        --env PYTHONDONTWRITEBYTECODE=1 \
        --env MODLY_AMD_RUNTIME_LOG=/dev/null \
        --env MODLY_GEOSAM2_IMAGE_ID="$GEOSAM2_IMAGE_ID" \
        "$GEOSAM2_IMAGE_REFERENCE" python -c '
import os,runpy,sys
api="/modly/api"
sys.path[:]=[entry for entry in sys.path if os.path.realpath(entry or ".") != os.path.realpath(api)]
import typing_extensions
sys.path.insert(0,"/modly")
sys.path.insert(0,api)
script=sys.argv[1]
sys.argv=[script,*sys.argv[2:]]
runpy.run_path(script,run_name="__main__")
' \
        /modly/api/runtime/amd/geosam2_qv_crosscheck.py \
        --source-root /modly/.modly-amd-runtime/source/geosam2-b5de23c60ab487d407b623d394a1614f9714761c \
        --renders /input/renders \
        --face-map /input/face-correspondence.json \
        --checkpoint /modly/.modly-amd-runtime/models/geosam2/geosam2.pt \
        --source-lock /modly/api/runtime/adapters/parts/GEOSAM2_SOURCE_LOCK.json \
        --model-lock /modly/api/runtime/adapters/parts/GEOSAM2_MODEL_LOCK.json \
        --dependency-lock /modly/api/runtime/adapters/parts/GEOSAM2_DEPENDENCY_LOCK.json \
        > "$TEMP_OUTPUT"; then
      rm -f "$TEMP_OUTPUT"
      exit 1
    fi
    if ! python "$ROOT_DIR/scripts/validate_geosam2_qv_report.py" "$TEMP_OUTPUT" \
        --project-root "$ROOT_DIR" --renders "$RENDER_DIR" \
        --face-map "$FACE_MAP_PATH" --container-image-id "$GEOSAM2_IMAGE_ID"; then
      exit 1
    fi
    if ! ln -- "$TEMP_OUTPUT" "$OUTPUT_PATH"; then
      printf 'Could not publish diagnostic output without replacing an existing file.\n' >&2
      exit 1
    fi
    rm -- "$TEMP_OUTPUT"
    TEMP_OUTPUT=""
    trap - EXIT INT TERM
    cat "$OUTPUT_PATH"
    ;;
  trace-geosam2-proposals)
    if [[ $# -lt 5 || $# -gt 7 ]]; then
      printf 'Usage: %s trace-geosam2-proposals WORKSPACE_DIR RENDER_DIR_RELATIVE_PATH FACE_MAP_RELATIVE_PATH OUTPUT_DIR_RELATIVE_PATH [PROPOSAL_ONLY_REPEAT_COUNT] [PROPOSAL_ONLY_ABA:0|1]\n' "$0" >&2
      exit 2
    fi
    WORKSPACE_DIR="$(realpath "$2")"
    RENDER_RELATIVE_PATH="$3"
    FACE_MAP_RELATIVE_PATH="$4"
    OUTPUT_RELATIVE_PATH="$5"
    PROPOSAL_ONLY_REPEAT_COUNT="${6:-0}"
    PROPOSAL_ONLY_ABA="${7:-0}"
    if [[ ! "$PROPOSAL_ONLY_REPEAT_COUNT" =~ ^[0-9]+$ ]] || (( PROPOSAL_ONLY_REPEAT_COUNT > 20 )); then
      printf 'Proposal-only repeat count must be an integer from 0 to 20.\n' >&2
      exit 2
    fi
    if [[ "$PROPOSAL_ONLY_ABA" != 0 && "$PROPOSAL_ONLY_ABA" != 1 ]] || (( PROPOSAL_ONLY_ABA == 1 && PROPOSAL_ONLY_REPEAT_COUNT != 0 )); then
      printf 'A-B-A must be 0 or 1 and cannot be combined with proposal-only repeats.\n' >&2
      exit 2
    fi
    for RELATIVE_PATH in "$RENDER_RELATIVE_PATH" "$FACE_MAP_RELATIVE_PATH" "$OUTPUT_RELATIVE_PATH"; do
      if [[ "$RELATIVE_PATH" == /* || "$RELATIVE_PATH" =~ (^|/)\.\.(/|$) || -z "$RELATIVE_PATH" ]]; then
        printf 'Render, face-map, and output paths must be workspace-relative without traversal components.\n' >&2
        exit 2
      fi
    done
    RENDER_DIR="$(realpath -e "$WORKSPACE_DIR/$RENDER_RELATIVE_PATH")"
    FACE_MAP_PATH="$(realpath -e "$WORKSPACE_DIR/$FACE_MAP_RELATIVE_PATH")"
    OUTPUT_PATH="$(realpath -m "$WORKSPACE_DIR/$OUTPUT_RELATIVE_PATH")"
    OUTPUT_PARENT="$(dirname "$OUTPUT_PATH")"
    OUTPUT_NAME="$(basename "$OUTPUT_PATH")"
    case "$RENDER_DIR" in "$WORKSPACE_DIR"/*) ;; *) printf 'Render bundle escapes the workspace.\n' >&2; exit 2 ;; esac
    case "$FACE_MAP_PATH" in "$WORKSPACE_DIR"/*) ;; *) printf 'Face map escapes the workspace.\n' >&2; exit 2 ;; esac
    case "$OUTPUT_PATH" in "$WORKSPACE_DIR"/*) ;; *) printf 'Output directory escapes the workspace.\n' >&2; exit 2 ;; esac
    if [[ ! -d "$RENDER_DIR" || ! -f "$RENDER_DIR/render_manifest.json" || ! -f "$FACE_MAP_PATH" || ! -d "$OUTPUT_PARENT" || -e "$OUTPUT_PATH" ]]; then
      printf 'Render bundle and face map must exist; output parent must exist and output directory must be new.\n' >&2
      exit 2
    fi
    SOURCE_ROOT="$ROOT_DIR/.modly-amd-runtime/source/geosam2-b5de23c60ab487d407b623d394a1614f9714761c"
    CHECKPOINT="$ROOT_DIR/.modly-amd-runtime/models/geosam2/geosam2.pt"
    SOURCE_LOCK="$ROOT_DIR/api/runtime/adapters/parts/GEOSAM2_SOURCE_LOCK.json"
    MODEL_LOCK="$ROOT_DIR/api/runtime/adapters/parts/GEOSAM2_MODEL_LOCK.json"
    DEPENDENCY_LOCK="$ROOT_DIR/api/runtime/adapters/parts/GEOSAM2_DEPENDENCY_LOCK.json"
    # shellcheck source=/dev/null
    source "$ROOT_DIR/scripts/geosam2-image-id.sh"
    GEOSAM2_IMAGE_ID_RAW="$(podman_local image inspect "$GEOSAM2_IMAGE" --format '{{.Id}}')"
    if ! GEOSAM2_IMAGE_ID="$(normalize_geosam2_image_id "$GEOSAM2_IMAGE_ID_RAW")"; then
      printf 'GeoSAM2 diagnostic image does not have an immutable local image ID.\n' >&2
      exit 2
    fi
    GEOSAM2_IMAGE_REFERENCE="$(geosam2_immutable_image_reference "$GEOSAM2_IMAGE_ID")"
    PROBE_DIAGNOSTIC_ARGS=(--proposal-only-repeats "$PROPOSAL_ONLY_REPEAT_COUNT")
    if [[ "$PROPOSAL_ONLY_ABA" == 1 ]]; then
      PROBE_DIAGNOSTIC_ARGS=(--proposal-only-aba)
    fi
    if podman_local run --rm --userns=host --network=none \
        --device /dev/kfd --device /dev/dri --group-add video --ipc=host \
        --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
        --volume "$ROOT_DIR/api:/modly/api:ro" \
        --volume "$SOURCE_ROOT:/modly/.modly-amd-runtime/source/geosam2-b5de23c60ab487d407b623d394a1614f9714761c:ro" \
        --volume "$(dirname "$CHECKPOINT"):/modly/.modly-amd-runtime/models/geosam2:ro" \
        --volume "$ROOT_DIR/.modly-amd-runtime/package-cache/geosam2:/modly/.modly-amd-runtime/package-cache/geosam2:ro" \
        --volume "$ROOT_DIR/.modly-amd-runtime/api-test-venv/lib/python3.12/site-packages:/api-site:ro" \
        --volume "$RENDER_DIR:/input/renders:ro" \
        --volume "$FACE_MAP_PATH:/input/face-correspondence.json:ro" \
        --volume "$OUTPUT_PARENT:/output-parent:rw" \
        --env MODLY_API_DIR=/modly/api \
        --env PYTHONPATH=/modly/api:/modly/.modly-amd-runtime/package-cache/geosam2/site-packages:/api-site:/opt/rocm/lib \
        --env PYTHONDONTWRITEBYTECODE=1 \
        --env MODLY_AMD_RUNTIME_LOG=/dev/null \
        "$GEOSAM2_IMAGE_REFERENCE" python -c '
import os,runpy,sys
api="/modly/api"
sys.path[:]=[entry for entry in sys.path if os.path.realpath(entry or ".") != os.path.realpath(api)]
import typing_extensions
sys.path.insert(0,"/modly")
sys.path.insert(0,api)
sys.path.insert(0,os.path.join(api,"runtime/adapters/parts"))
script=sys.argv[1]
sys.argv=[script,*sys.argv[2:]]
runpy.run_module("runtime.adapters.parts.geosam2_probe",run_name="__main__")
' \
        runtime.adapters.parts.geosam2_probe \
          --source-root /modly/.modly-amd-runtime/source/geosam2-b5de23c60ab487d407b623d394a1614f9714761c \
          --renders /input/renders \
          --checkpoint /modly/.modly-amd-runtime/models/geosam2/geosam2.pt \
          --output "/output-parent/$OUTPUT_NAME" \
          --face-map /input/face-correspondence.json \
          --source-lock /modly/api/runtime/adapters/parts/GEOSAM2_SOURCE_LOCK.json \
          --dependency-lock /modly/api/runtime/adapters/parts/GEOSAM2_DEPENDENCY_LOCK.json \
          --model-lock /modly/api/runtime/adapters/parts/GEOSAM2_MODEL_LOCK.json \
          --seed-policy all_rendered_views --run-count 1 --trace-proposals \
          "${PROBE_DIAGNOSTIC_ARGS[@]}"; then
      printf 'Proposal filter trace completed. Artifacts: %s\n' "$OUTPUT_PATH"
    else
      TRACE_EXIT_CODE=$?
      printf 'GeoSAM2 probe failed with exit %s; inspect any preserved proposal-filter trace under %s\n' "$TRACE_EXIT_CODE" "$OUTPUT_PATH" >&2
      exit "$TRACE_EXIT_CODE"
    fi
    ;;
  trace-geosam2-lifecycle)
    if [[ $# -lt 3 || $# -gt 4 ]]; then
      printf 'Usage: %s trace-geosam2-lifecycle WORKSPACE_DIR OUTPUT_DIR_RELATIVE_PATH [plain|image-boundaries|encoder-stages|encoder-first-conv]\n' "$0" >&2
      exit 2
    fi
    WORKSPACE_DIR="$(realpath "$2")"
    OUTPUT_RELATIVE_PATH="$3"
    LIFECYCLE_MODE="${4:-plain}"
    if [[ "$LIFECYCLE_MODE" != plain && "$LIFECYCLE_MODE" != image-boundaries && "$LIFECYCLE_MODE" != encoder-stages && "$LIFECYCLE_MODE" != encoder-first-conv ]]; then
      printf 'Lifecycle mode must be plain, image-boundaries, encoder-stages, or encoder-first-conv.\n' >&2
      exit 2
    fi
    if [[ "$OUTPUT_RELATIVE_PATH" == /* || "$OUTPUT_RELATIVE_PATH" =~ (^|/)\.\.(/|$) || -z "$OUTPUT_RELATIVE_PATH" ]]; then
      printf 'Output path must be workspace-relative without traversal components.\n' >&2
      exit 2
    fi
    F8_RUN_RELATIVE_PATH='.modly-amd-runtime/results/ticket04-variation-panel/flamingo/StructuredAssets/runs/f8c6a77e-7090-4713-bb7d-3ddcedadeb1b'
    RENDER_DIR="$(realpath -e "$WORKSPACE_DIR/$F8_RUN_RELATIVE_PATH/geosam2-render")"
    FACE_MAP_PATH="$(realpath -e "$WORKSPACE_DIR/$F8_RUN_RELATIVE_PATH/geosam2-input/face-correspondence.json")"
    OUTPUT_PATH="$(realpath -m "$WORKSPACE_DIR/$OUTPUT_RELATIVE_PATH")"
    OUTPUT_PARENT="$(dirname "$OUTPUT_PATH")"
    OUTPUT_NAME="$(basename "$OUTPUT_PATH")"
    case "$RENDER_DIR" in "$WORKSPACE_DIR"/*) ;; *) printf 'f8 render bundle escapes the workspace.\n' >&2; exit 2 ;; esac
    case "$FACE_MAP_PATH" in "$WORKSPACE_DIR"/*) ;; *) printf 'f8 face map escapes the workspace.\n' >&2; exit 2 ;; esac
    case "$OUTPUT_PATH" in "$WORKSPACE_DIR"/*) ;; *) printf 'Output path escapes the workspace.\n' >&2; exit 2 ;; esac
    if [[ ! -d "$RENDER_DIR" || ! -f "$RENDER_DIR/render_manifest.json" || ! -f "$FACE_MAP_PATH" || ! -d "$OUTPUT_PARENT" || -e "$OUTPUT_PATH" ]]; then
      printf 'Locked f8 render and face map must exist; output parent must exist and output directory must be new.\n' >&2
      exit 2
    fi
    SOURCE_ROOT="$ROOT_DIR/.modly-amd-runtime/source/geosam2-b5de23c60ab487d407b623d394a1614f9714761c"
    CHECKPOINT="$ROOT_DIR/.modly-amd-runtime/models/geosam2/geosam2.pt"
    SOURCE_LOCK="$ROOT_DIR/api/runtime/adapters/parts/GEOSAM2_SOURCE_LOCK.json"
    MODEL_LOCK="$ROOT_DIR/api/runtime/adapters/parts/GEOSAM2_MODEL_LOCK.json"
    DEPENDENCY_LOCK="$ROOT_DIR/api/runtime/adapters/parts/GEOSAM2_DEPENDENCY_LOCK.json"
    DIAGNOSTIC_LOCK="$ROOT_DIR/api/runtime/adapters/parts/GEOSAM2_LIFECYCLE_DIAGNOSTICS_LOCK.v1.json"
    EXPECTED_DIAGNOSTIC_LOCK_SHA256='b22098e3288229c231fecc0c6eab4bb8bdcc4fe65f6f62a932f3043934929297'
    LIFECYCLE_PROBE_MODULE='runtime.adapters.parts.geosam2_lifecycle_probe'
    BOUNDARY_LOCK_ARGS=()
    if [[ "$(sha256sum "$DIAGNOSTIC_LOCK" | cut -d ' ' -f 1)" != "$EXPECTED_DIAGNOSTIC_LOCK_SHA256" ]]; then
      printf 'Lifecycle diagnostic lock failed pinned SHA-256 verification.\n' >&2
      exit 2
    fi
    if [[ "$LIFECYCLE_MODE" == image-boundaries || "$LIFECYCLE_MODE" == encoder-stages || "$LIFECYCLE_MODE" == encoder-first-conv ]]; then
      BOUNDARY_LOCK="$ROOT_DIR/api/runtime/adapters/parts/GEOSAM2_IMAGE_PATH_BOUNDARY_LOCK.v1.json"
      EXPECTED_BOUNDARY_LOCK_SHA256='969128a913f36c57a8830dcb6ff079d493057a7fe8b434e6d20832fccbf0d756'
      if [[ "$(sha256sum "$BOUNDARY_LOCK" | cut -d ' ' -f 1)" != "$EXPECTED_BOUNDARY_LOCK_SHA256" ]]; then
        printf 'Image-path boundary lock failed pinned SHA-256 verification.\n' >&2
        exit 2
      fi
      LIFECYCLE_PROBE_MODULE='runtime.adapters.parts.geosam2_image_path_boundary_probe'
      BOUNDARY_LOCK_ARGS+=(--boundary-lock /modly/api/runtime/adapters/parts/GEOSAM2_IMAGE_PATH_BOUNDARY_LOCK.v1.json)
      BOUNDARY_LOCK_ARGS+=(--expected-boundary-lock-sha256 "$EXPECTED_BOUNDARY_LOCK_SHA256")
    fi
    STAGE_LOCK_ARGS=()
    if [[ "$LIFECYCLE_MODE" == encoder-stages || "$LIFECYCLE_MODE" == encoder-first-conv ]]; then
      STAGE_LOCK="$ROOT_DIR/api/runtime/adapters/parts/GEOSAM2_IMAGE_ENCODER_STAGE_LOCK.v1.json"
      EXPECTED_STAGE_LOCK_SHA256='d64d334d86a82793467b1645fe522a2ce71a76f163cfe390ef87a01c3203170b'
      EXPECTED_STAGE_RUNNER_SHA256='6e1878761f6f65dedd6d6da141a317828a15597e755e7d45d5cbace2625d6347'
      ACTUAL_STAGE_RUNNER_SHA256="$(sha256sum "$ROOT_DIR/api/runtime/adapters/parts/geosam2_image_encoder_stage_probe.py" | cut -d ' ' -f 1)"
      if [[ "$(sha256sum "$STAGE_LOCK" | cut -d ' ' -f 1)" != "$EXPECTED_STAGE_LOCK_SHA256" || "$ACTUAL_STAGE_RUNNER_SHA256" != "$EXPECTED_STAGE_RUNNER_SHA256" ]]; then
        printf 'Image-encoder stage lock failed pinned SHA-256 verification.\n' >&2
        exit 2
      fi
      LIFECYCLE_PROBE_MODULE='runtime.adapters.parts.geosam2_image_encoder_stage_probe'
      STAGE_LOCK_ARGS+=(--stage-lock /modly/api/runtime/adapters/parts/GEOSAM2_IMAGE_ENCODER_STAGE_LOCK.v1.json)
      STAGE_LOCK_ARGS+=(--expected-stage-lock-sha256 "$EXPECTED_STAGE_LOCK_SHA256")
    fi
    FIRST_CONV_LOCK_ARGS=()
    if [[ "$LIFECYCLE_MODE" == encoder-first-conv ]]; then
      FIRST_CONV_LOCK="$ROOT_DIR/api/runtime/adapters/parts/GEOSAM2_FIRST_CONV_LOCK.v1.json"
      EXPECTED_FIRST_CONV_LOCK_SHA256='a91fcbe6bdefc99a2f0268ef27fcfa9d8ee4779450c682c56ac2af3a526ada1c'
      EXPECTED_FIRST_CONV_PROBE_SHA256='b81ed031345977f9628361ab7b8eecbf000644ad1ae78dd979199c5d6c98b9ac'
      ACTUAL_FIRST_CONV_PROBE_SHA256="$(sha256sum "$ROOT_DIR/api/runtime/adapters/parts/geosam2_first_conv_probe.py" | cut -d ' ' -f 1)"
      if [[ "$(sha256sum "$FIRST_CONV_LOCK" | cut -d ' ' -f 1)" != "$EXPECTED_FIRST_CONV_LOCK_SHA256" || "$ACTUAL_FIRST_CONV_PROBE_SHA256" != "$EXPECTED_FIRST_CONV_PROBE_SHA256" ]]; then
        printf 'First-convolution diagnostic lock/runner identity check failed.\n' >&2
        exit 2
      fi
      LIFECYCLE_PROBE_MODULE='runtime.adapters.parts.geosam2_first_conv_probe'
      FIRST_CONV_LOCK_ARGS+=(--first-conv-lock /modly/api/runtime/adapters/parts/GEOSAM2_FIRST_CONV_LOCK.v1.json)
      FIRST_CONV_LOCK_ARGS+=(--expected-first-conv-lock-sha256 "$EXPECTED_FIRST_CONV_LOCK_SHA256")
    fi
    # shellcheck source=/dev/null
    source "$ROOT_DIR/scripts/geosam2-image-id.sh"
    GEOSAM2_IMAGE_ID_RAW="$(podman_local image inspect "$GEOSAM2_IMAGE" --format '{{.Id}}')"
    if ! GEOSAM2_IMAGE_ID="$(normalize_geosam2_image_id "$GEOSAM2_IMAGE_ID_RAW")"; then
      printf 'GeoSAM2 lifecycle diagnostic image does not have an immutable local image ID.\n' >&2
      exit 2
    fi
    GEOSAM2_IMAGE_REFERENCE="$(geosam2_immutable_image_reference "$GEOSAM2_IMAGE_ID")"
    if podman_local run --rm --userns=host --network=none \
        --device /dev/kfd --device /dev/dri --group-add video --ipc=host \
        --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
        --volume "$ROOT_DIR/api:/modly/api:ro" \
        --volume "$SOURCE_ROOT:/modly/.modly-amd-runtime/source/geosam2-b5de23c60ab487d407b623d394a1614f9714761c:ro" \
        --volume "$(dirname "$CHECKPOINT"):/modly/.modly-amd-runtime/models/geosam2:ro" \
        --volume "$ROOT_DIR/.modly-amd-runtime/package-cache/geosam2:/modly/.modly-amd-runtime/package-cache/geosam2:ro" \
        --volume "$ROOT_DIR/.modly-amd-runtime/api-test-venv/lib/python3.12/site-packages:/api-site:ro" \
        --volume "$RENDER_DIR:/input/renders:ro" \
        --volume "$FACE_MAP_PATH:/input/face-correspondence.json:ro" \
        --volume "$OUTPUT_PARENT:/output-parent:rw" \
        --env MODLY_API_DIR=/modly/api \
        --env MODLY_LIFECYCLE_PROBE_MODULE="$LIFECYCLE_PROBE_MODULE" \
        --env PYTHONPATH=/modly/api:/modly/.modly-amd-runtime/package-cache/geosam2/site-packages:/api-site:/opt/rocm/lib \
        --env PYTHONDONTWRITEBYTECODE=1 \
        --env MODLY_AMD_RUNTIME_LOG=/dev/null \
        "$GEOSAM2_IMAGE_REFERENCE" python -c '
import os,runpy,sys
api="/modly/api"
sys.path[:]=[entry for entry in sys.path if os.path.realpath(entry or ".") != os.path.realpath(api)]
import typing_extensions
sys.path.insert(0,"/modly")
sys.path.insert(0,api)
runpy.run_module(os.environ["MODLY_LIFECYCLE_PROBE_MODULE"],run_name="__main__")
' \
        --source-root /modly/.modly-amd-runtime/source/geosam2-b5de23c60ab487d407b623d394a1614f9714761c \
        --renders /input/renders \
        --face-map /input/face-correspondence.json \
        --checkpoint /modly/.modly-amd-runtime/models/geosam2/geosam2.pt \
        --source-lock /modly/api/runtime/adapters/parts/GEOSAM2_SOURCE_LOCK.json \
        --model-lock /modly/api/runtime/adapters/parts/GEOSAM2_MODEL_LOCK.json \
        --dependency-lock /modly/api/runtime/adapters/parts/GEOSAM2_DEPENDENCY_LOCK.json \
        --diagnostic-lock /modly/api/runtime/adapters/parts/GEOSAM2_LIFECYCLE_DIAGNOSTICS_LOCK.v1.json \
        --expected-diagnostic-lock-sha256 "$EXPECTED_DIAGNOSTIC_LOCK_SHA256" \
        "${BOUNDARY_LOCK_ARGS[@]}" \
        "${STAGE_LOCK_ARGS[@]}" \
        "${FIRST_CONV_LOCK_ARGS[@]}" \
        --output "/output-parent/$OUTPUT_NAME"; then
      if [[ "$LIFECYCLE_MODE" == encoder-first-conv ]]; then
        printf 'GeoSAM2 first-convolution diagnostic completed. Artifact: %s/lifecycle-diagnostic.json\n' "$OUTPUT_PATH"
      elif [[ "$LIFECYCLE_MODE" == encoder-stages ]]; then
        printf 'GeoSAM2 image-encoder stage diagnostic completed. Artifact: %s/lifecycle-diagnostic.json\n' "$OUTPUT_PATH"
      elif [[ "$LIFECYCLE_MODE" == image-boundaries ]]; then
        printf 'GeoSAM2 image-path boundary diagnostic completed. Artifact: %s/lifecycle-diagnostic.json\n' "$OUTPUT_PATH"
      else
        printf 'GeoSAM2 lifecycle diagnostic completed. Artifact: %s/lifecycle-diagnostic.json\n' "$OUTPUT_PATH"
      fi
    else
      TRACE_EXIT_CODE=$?
      printf 'GeoSAM2 lifecycle diagnostic failed with exit %s; inspect the preserved artifact under %s\n' "$TRACE_EXIT_CODE" "$OUTPUT_PATH" >&2
      exit "$TRACE_EXIT_CODE"
    fi
    ;;
  trace-geosam2-finite-retry)
    if [[ $# -lt 3 || $# -gt 4 ]]; then
      printf 'Usage: %s trace-geosam2-finite-retry WORKSPACE_DIR OUTPUT_DIR_RELATIVE_PATH [VIEW_INDEX_0_OR_2_OR_012_OR_PROD012]\n' "$0" >&2
      exit 2
    fi
    WORKSPACE_DIR="$(realpath "$2")"
    OUTPUT_RELATIVE_PATH="$3"
    VIEW_INDEX="${4:-0}"
    if [[ "$VIEW_INDEX" == 0 ]]; then
      PROBE_MODULE='runtime.adapters.parts.geosam2_finite_retry_probe'
      PROBE_LOCK_BASENAME='GEOSAM2_FINITE_RETRY_PROBE_LOCK.v1.json'
      EXPECTED_PROBE_LOCK_SHA256='1c37c04294236b3315ddc99da136077d4fc604a5054a9479ac80955d9be8f30a'
    elif [[ "$VIEW_INDEX" == 2 ]]; then
      PROBE_MODULE='runtime.adapters.parts.geosam2_finite_retry_view2_probe'
      PROBE_LOCK_BASENAME='GEOSAM2_FINITE_RETRY_VIEW2_PROBE_LOCK.v1.json'
      EXPECTED_PROBE_LOCK_SHA256='28ad68baccfe6cfede4459fd7ba032cf5f7eb654d197c90d5d1a6907e7cdcf3c'
    elif [[ "$VIEW_INDEX" == 012 ]]; then
      PROBE_MODULE='runtime.adapters.parts.geosam2_finite_retry_sequence_probe'
      PROBE_LOCK_BASENAME='GEOSAM2_FINITE_RETRY_SEQUENCE_PROBE_LOCK.v1.json'
      EXPECTED_PROBE_LOCK_SHA256='e223e2eaa1ea07d950c9da6d18e6989701fd269257bfee0f15350ad4646fe7d5'
    elif [[ "$VIEW_INDEX" == prod012 ]]; then
      PROBE_MODULE='runtime.adapters.parts.geosam2_finite_retry_production_context_probe'
      PROBE_LOCK_BASENAME='GEOSAM2_FINITE_RETRY_PRODUCTION_CONTEXT_LOCK.v1.json'
      EXPECTED_PROBE_LOCK_SHA256='5460be379e514bbae9bfaa6460726c80b3419bfee5e84d441a3602118b34ae8a'
    else
      printf 'Only locked Flamingo views 0, 2, the ordered sequence 0-1-2, or production-context 0-1-2 are supported.\n' >&2
      exit 2
    fi
    if [[ "$OUTPUT_RELATIVE_PATH" == /* || "$OUTPUT_RELATIVE_PATH" =~ (^|/)\.\.(/|$) || -z "$OUTPUT_RELATIVE_PATH" ]]; then
      printf 'Output path must be workspace-relative without traversal components.\n' >&2
      exit 2
    fi
    F8_RUN_RELATIVE_PATH='.modly-amd-runtime/results/ticket04-variation-panel/flamingo/StructuredAssets/runs/f8c6a77e-7090-4713-bb7d-3ddcedadeb1b'
    RENDER_DIR="$(realpath -e "$WORKSPACE_DIR/$F8_RUN_RELATIVE_PATH/geosam2-render")"
    FACE_MAP_PATH="$(realpath -e "$WORKSPACE_DIR/$F8_RUN_RELATIVE_PATH/geosam2-input/face-correspondence.json")"
    OUTPUT_PATH="$(realpath -m "$WORKSPACE_DIR/$OUTPUT_RELATIVE_PATH")"
    OUTPUT_PARENT="$(dirname "$OUTPUT_PATH")"
    OUTPUT_NAME="$(basename "$OUTPUT_PATH")"
    case "$RENDER_DIR" in "$WORKSPACE_DIR"/*) ;; *) printf 'f8 render bundle escapes the workspace.\n' >&2; exit 2 ;; esac
    case "$FACE_MAP_PATH" in "$WORKSPACE_DIR"/*) ;; *) printf 'f8 face map escapes the workspace.\n' >&2; exit 2 ;; esac
    case "$OUTPUT_PATH" in "$WORKSPACE_DIR"/*) ;; *) printf 'Output path escapes the workspace.\n' >&2; exit 2 ;; esac
    if [[ ! -d "$RENDER_DIR" || ! -f "$RENDER_DIR/render_manifest.json" || ! -f "$FACE_MAP_PATH" || ! -d "$OUTPUT_PARENT" || -e "$OUTPUT_PATH" ]]; then
      printf 'Locked f8 render and face map must exist; output parent must exist and output directory must be new.\n' >&2
      exit 2
    fi
    SOURCE_ROOT="$ROOT_DIR/.modly-amd-runtime/source/geosam2-b5de23c60ab487d407b623d394a1614f9714761c"
    CHECKPOINT="$ROOT_DIR/.modly-amd-runtime/models/geosam2/geosam2.pt"
    SOURCE_LOCK="$ROOT_DIR/api/runtime/adapters/parts/GEOSAM2_SOURCE_LOCK.json"
    MODEL_LOCK="$ROOT_DIR/api/runtime/adapters/parts/GEOSAM2_MODEL_LOCK.json"
    DEPENDENCY_LOCK="$ROOT_DIR/api/runtime/adapters/parts/GEOSAM2_DEPENDENCY_LOCK.json"
    PROBE_LOCK="$ROOT_DIR/api/runtime/adapters/parts/$PROBE_LOCK_BASENAME"
    if [[ "$(sha256sum "$PROBE_LOCK" | cut -d ' ' -f 1)" != "$EXPECTED_PROBE_LOCK_SHA256" ]]; then
      printf 'Finite-retry diagnostic lock failed pinned SHA-256 verification.\n' >&2
      exit 2
    fi
    # shellcheck source=/dev/null
    source "$ROOT_DIR/scripts/geosam2-image-id.sh"
    GEOSAM2_IMAGE_ID_RAW="$(podman_local image inspect "$GEOSAM2_IMAGE" --format '{{.Id}}')"
    if ! GEOSAM2_IMAGE_ID="$(normalize_geosam2_image_id "$GEOSAM2_IMAGE_ID_RAW")"; then
      printf 'GeoSAM2 finite-retry diagnostic image does not have an immutable local image ID.\n' >&2
      exit 2
    fi
    GEOSAM2_IMAGE_REFERENCE="$(geosam2_immutable_image_reference "$GEOSAM2_IMAGE_ID")"
    if podman_local run --rm --userns=host --network=none \
        --device /dev/kfd --device /dev/dri --group-add video --ipc=host \
        --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
        --volume "$ROOT_DIR/api:/modly/api:ro" \
        --volume "$SOURCE_ROOT:/modly/.modly-amd-runtime/source/geosam2-b5de23c60ab487d407b623d394a1614f9714761c:ro" \
        --volume "$(dirname "$CHECKPOINT"):/modly/.modly-amd-runtime/models/geosam2:ro" \
        --volume "$ROOT_DIR/.modly-amd-runtime/package-cache/geosam2:/modly/.modly-amd-runtime/package-cache/geosam2:ro" \
        --volume "$ROOT_DIR/.modly-amd-runtime/api-test-venv/lib/python3.12/site-packages:/api-site:ro" \
        --volume "$RENDER_DIR:/input/renders:ro" \
        --volume "$FACE_MAP_PATH:/input/face-correspondence.json:ro" \
        --volume "$OUTPUT_PARENT:/output-parent:rw" \
        --env MODLY_API_DIR=/modly/api \
        --env MODLY_GEOSAM2_IMAGE_ID="$GEOSAM2_IMAGE_ID" \
        --env MODLY_GEOSAM2_PROBE_MODULE="$PROBE_MODULE" \
        --env PYTHONPATH=/modly/api:/modly/.modly-amd-runtime/package-cache/geosam2/site-packages:/api-site:/opt/rocm/lib \
        --env PYTHONDONTWRITEBYTECODE=1 \
        --env MODLY_AMD_RUNTIME_LOG=/dev/null \
        "$GEOSAM2_IMAGE_REFERENCE" python -c '
import os,runpy,sys
api="/modly/api"
sys.path[:]=[entry for entry in sys.path if os.path.realpath(entry or ".") != os.path.realpath(api)]
import typing_extensions
sys.path.insert(0,"/modly")
sys.path.insert(0,api)
runpy.run_module(os.environ["MODLY_GEOSAM2_PROBE_MODULE"],run_name="__main__")
' \
        --source-root /modly/.modly-amd-runtime/source/geosam2-b5de23c60ab487d407b623d394a1614f9714761c \
        --renders /input/renders \
        --face-map /input/face-correspondence.json \
        --checkpoint /modly/.modly-amd-runtime/models/geosam2/geosam2.pt \
        --source-lock /modly/api/runtime/adapters/parts/GEOSAM2_SOURCE_LOCK.json \
        --model-lock /modly/api/runtime/adapters/parts/GEOSAM2_MODEL_LOCK.json \
        --dependency-lock /modly/api/runtime/adapters/parts/GEOSAM2_DEPENDENCY_LOCK.json \
        --probe-lock "/modly/api/runtime/adapters/parts/$PROBE_LOCK_BASENAME" \
        --expected-probe-lock-sha256 "$EXPECTED_PROBE_LOCK_SHA256" \
        --output "/output-parent/$OUTPUT_NAME"; then
      :
    else
      TRACE_EXIT_CODE=$?
      printf 'GeoSAM2 finite-retry diagnostic failed with exit %s; inspect the preserved artifact under %s\n' "$TRACE_EXIT_CODE" "$OUTPUT_PATH" >&2
      exit "$TRACE_EXIT_CODE"
    fi
    if [[ "$VIEW_INDEX" == prod012 ]]; then
      printf 'GeoSAM2 production-context diagnostic completed. Artifact: %s/production-context-probe.json\n' "$OUTPUT_PATH"
    else
      printf 'GeoSAM2 finite-retry diagnostic completed. Artifact: %s/finite-retry-probe.json\n' "$OUTPUT_PATH"
    fi
    ;;
  *)
    printf 'Usage: %s {build|build-geosam2|status|budget-check|probe|memory|workflow-geosam2|diagnose-geosam2-qv|trace-geosam2-proposals|trace-geosam2-lifecycle|trace-geosam2-finite-retry}\n' "$0" >&2
    exit 2
    ;;
esac
