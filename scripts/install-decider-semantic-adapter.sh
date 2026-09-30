#!/usr/bin/env bash
set -euo pipefail

# Install only the Decider semantic adapter as an immutable, code-only overlay.
# Model weights and credentials are deliberately excluded from this package.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PACKAGE_DIR="$ROOT_DIR/api"
TMPDIR="$ROOT_DIR/.modly-amd-runtime/tmp"
mkdir -p "$TMPDIR"
export TMPDIR
PACKAGE_SOURCE="$PACKAGE_DIR/runtime/adapters/parts/decider_2b_local.py"
ASSET_LOCK="$PACKAGE_DIR/runtime/adapters/parts/DECIDER_2B_VISION_GGUF_ASSET_LOCK.json"
HARNESS_LOCK="$PACKAGE_DIR/runtime/adapters/parts/DECIDER_2B_VISION_GGUF_HARNESS_LOCK.json"
for required in "$PACKAGE_DIR/pyproject.toml" "$PACKAGE_SOURCE" "$ASSET_LOCK" "$HARNESS_LOCK"; do
  [[ -f "$required" ]] || { printf 'Required Decider adapter package file is missing: %s\n' "$required" >&2; exit 1; }
done

PACKAGE_DIGEST="$(sha256sum "$PACKAGE_DIR/pyproject.toml" "$PACKAGE_SOURCE" "$ASSET_LOCK" "$HARNESS_LOCK" | sha256sum | cut -d ' ' -f 1)"
OVERLAY_ROOT="$ROOT_DIR/.modly-amd-runtime/package-cache/decider/$PACKAGE_DIGEST"
SITE_PACKAGES="$OVERLAY_ROOT/site-packages"
verify_overlay() {
  PYTHONPATH="$1${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -c '
import importlib.metadata
points = list(importlib.metadata.entry_points(group="modly.semantic_adapters"))
assert len(points) == 1 and points[0].name == "mindchain.decider-2b-vision.gguf.v1", points
assert points[0].module == "runtime.adapters.parts.decider_2b_local"
dist = points[0].dist
assert dist is not None
for name in ("DECIDER_2B_VISION_GGUF_ASSET_LOCK.json", "DECIDER_2B_VISION_GGUF_HARNESS_LOCK.json"):
    dist.locate_file("runtime/adapters/parts/" + name).resolve(strict=True)
'
}
if [[ -d "$SITE_PACKAGES" ]]; then
  verify_overlay "$SITE_PACKAGES" || { printf 'Existing Decider adapter overlay failed verification: %s\n' "$SITE_PACKAGES" >&2; exit 1; }
  printf '%s\n' "$SITE_PACKAGES"
  exit 0
fi

mkdir -p "$OVERLAY_ROOT"
TEMP_ROOT="$(mktemp -d "$OVERLAY_ROOT/.install.XXXXXX")"
cleanup() { rm -rf -- "$TEMP_ROOT"; }
trap cleanup EXIT
mkdir -p "$TEMP_ROOT/site-packages"
PIP_CACHE_DIR="$ROOT_DIR/.modly-amd-runtime/pip-cache" PIP_NO_CACHE_DIR=1 \
  "$PYTHON_BIN" -m pip install --disable-pip-version-check --no-deps --no-build-isolation \
  --target "$TEMP_ROOT/site-packages" "$PACKAGE_DIR"
verify_overlay "$TEMP_ROOT/site-packages"
mv -- "$TEMP_ROOT/site-packages" "$SITE_PACKAGES"
printf '%s\n' "$SITE_PACKAGES"
