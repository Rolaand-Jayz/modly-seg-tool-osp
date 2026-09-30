# Native Decider batch probe build

This probe is an optional native-only executable for the Modly Decider GGUF
adapter. It uses the exact pinned llama.cpp checkout and the project's existing
ROCm 7.14 image. It does not install or update ROCm.

## Input contract

Invoke:

`decider-synthetic-logits-probe --model TEXT_GGUF --mmproj MMPROJ_GGUF --input-jsonl REQUESTS.jsonl`

Each non-empty JSONL row contains `part_id`, `topology_revision`, `prompt`,
`view_index` (0 through 3), and one `image` object with `path`, `artifact_id`,
`digest` (the exact encoded file's `sha256:<hex>`), and `kind`. Each part must
have four consecutive rows ordered by `view_index`, with consistent part,
topology, and exact prompt bytes. Paths point to original encoded PNG, JPEG,
or supported WebP crop bytes. The pinned llama.cpp mtmd helper decodes those
bytes directly; the probe does not resize or re-encode them. The prompt uses
the Decider `Answer: (` answer slot and may omit a media marker; if present,
there may be at most one. If the prompt has no marker, one mtmd marker is
prepended. The prompt must end
exactly at `Answer: (`; this strict suffix ensures that the final evaluated
text position is the upstream Decider answer slot. The harness uses
`mtmd_helper_eval_chunks(..., logits_last=true)` and reads logits at the
final evaluated position. All rows, identities, paths, content digests, and
image decoding are checked before any stdout is emitted.

The header includes the llama.cpp commit/probe build identity and the pinned
ROCm runtime label. Device selection resolves the HIP backend registry
explicitly and fails instead of falling back to another generic GPU backend.

The output is JSONL: one `modly.decider.slot-scores.v2` header with backend,
device, exact prompt digest, exact model/projector digests, and the fixed
`equal_weight_mean_of_four_view_probabilities` aggregation identity; then one
score record per input row. Each score record includes `view_index`, `view_id`,
one exact evidence reference, raw restricted next-token logits for unique
one-token A-J choices, and the restricted-choice softmax values. Those
probabilities are uncalibrated and are not confidence claims. Modly's adapter
combines the four probability vectors by arithmetic mean and emits one
part-level decision; it does not average logits.

The batch mode loads the text model, context, and projector once and clears
llama.cpp memory before each one-image row. The legacy single-query CLI remains supported.

The pinned source warning `find_slot: non-consecutive token position` is
emitted by `src/llama-memory-recurrent.cpp` when a cache slot's final position
differs from its previous position plus the number of new tokens. Its
relationship to Decider's multimodal image positions and the effect on
recurrent-state correctness remain unresolved. Deterministic repeats do not
establish slot alignment, so the warning is preserved in stderr evidence.

## Build in the pinned runtime

The source tree must be the audited llama.cpp commit
`9575389609d6f8437de0b205561a4824d217c409`. The project runtime image is
`localhost/modly-amd-migraphx:ticket02` (ROCm 7.14.60850). Build directory,
compiler cache, temporary files, and output executable must remain under the
project's `.modly-amd-runtime/` work-drive directory.

From the repository root, with the pinned image and source checkout already
present locally:

```sh
RUNTIME=.modly-amd-runtime
SRC="$RUNTIME/source/llama.cpp-decider-9575389"
BUILD="$RUNTIME/build/llama-decider-9575389-rocm714-hip"
IMAGE=localhost/modly-amd-migraphx:ticket02

# Run this compile inside the existing image, mounting the work-drive tree at /mnt/workdrive.
# The pinned source vendors both headers and the SHA-256 C implementation.
podman --root "$PWD/$RUNTIME/storage" --runroot "$PWD/$RUNTIME/run" run --rm \
  -v "$PWD:/mnt/workdrive:rw" -w /mnt/workdrive "$IMAGE" bash -lc '
  set -e
  export TMPDIR=/mnt/workdrive/.modly-amd-runtime/tmp
  export XDG_CACHE_HOME=/mnt/workdrive/.modly-amd-runtime/cache
  SRC=/mnt/workdrive/.modly-amd-runtime/source/llama.cpp-decider-9575389
  BUILD=/mnt/workdrive/.modly-amd-runtime/build/llama-decider-9575389-rocm714-hip
  ROCM=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib
  c++ -O3 -DNDEBUG -std=c++17 \
    -I"$SRC/vendor" -I"$SRC/vendor/hash" -I"$SRC/vendor/hash/sha256" \
    -I"$SRC/include" -I"$SRC/ggml/include" -I"$SRC/tools/mtmd" \
    api/runtime/adapters/parts/decider_gguf/synthetic_logits_probe.cpp \
    "$SRC/vendor/hash/sha256/sha256.c" \
    "$BUILD/bin/libmtmd.so.0.4.1" "$BUILD/bin/libllama.so.0.4.1" \
    "$BUILD/bin/libggml.so.0.25.0" "$BUILD/bin/libggml-base.so.0.25.0" \
    "$BUILD/bin/libggml-hip.so.0.25.0" "$BUILD/bin/libggml-cpu.so.0.25.0" \
    -L"$ROCM" -Wl,-rpath,"$BUILD/bin:$ROCM" -lamdhip64 -lhipblas -lrocblas \
    -o "$BUILD/bin/decider-synthetic-logits-probe"
  '
```

The compiler invocation must run in the pinned runtime container because its
HIP compiler/runtime and shared libraries are part of the build identity. The
existing image is expected to have `/mnt/workdrive` mounted read/write. Keep
`TMPDIR`, `XDG_CACHE_HOME`, and any CMake compiler cache below
`.modly-amd-runtime/`; do not let container tooling fall back to the boot drive.
