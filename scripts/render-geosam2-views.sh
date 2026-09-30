#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BLENDER_DIR="$ROOT_DIR/.modly-amd-runtime/renderers/blender-4.0.2/blender-4.0.2-linux-x64"
BLENDER="$BLENDER_DIR/blender"
SOURCE_ROOT="$ROOT_DIR/.modly-amd-runtime/source/geosam2-b5de23c60ab487d407b623d394a1614f9714761c"
PYTHON_OVERLAY="$ROOT_DIR/.modly-amd-runtime/renderers/blender-4.0.2/python-overlay"
USER_ROOT="$ROOT_DIR/.modly-amd-runtime/renderers/blender-user"

if [[ $# -ne 3 ]]; then
  printf 'Usage: %s <canonical-inference-mesh.glb> glb <new-output-directory>\n' "$0" >&2
  exit 2
fi
if [[ "$2" != "glb" ]]; then
  printf 'GeoSAM2 renderer accepts only the canonical GLB inference input.\n' >&2
  exit 2
fi
if [[ ! -x "$BLENDER" || ! -f "$SOURCE_ROOT/geosam2_render.py" ]]; then
  printf 'Pinned project-local Blender or GeoSAM2 source is missing.\n' >&2
  exit 2
fi
if [[ -e "$3" ]]; then
  printf 'Output directory already exists; select a fresh run directory.\n' >&2
  exit 2
fi

export MODLY_GEOSAM2_SOURCE="$SOURCE_ROOT"
export MODLY_GEOSAM2_SOURCE_LOCK="$ROOT_DIR/api/runtime/adapters/parts/GEOSAM2_SOURCE_LOCK.json"
export MODLY_GEOSAM2_RENDER_PYTHON_OVERLAY="$PYTHON_OVERLAY"
export BLENDER_USER_CONFIG="$USER_ROOT/config"
export BLENDER_USER_SCRIPTS="$USER_ROOT/scripts"
export BLENDER_USER_DATAFILES="$USER_ROOT/datafiles"
mkdir -p "$BLENDER_USER_CONFIG" "$BLENDER_USER_SCRIPTS" "$BLENDER_USER_DATAFILES"

exec "$BLENDER" --background --factory-startup -P "$ROOT_DIR/api/runtime/adapters/parts/geosam2_render_entry.py" -- "$1" "$2" "$3"
