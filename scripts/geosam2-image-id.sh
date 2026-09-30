#!/usr/bin/env bash

normalize_geosam2_image_id() {
  local raw_id="${1#sha256:}"
  if [[ ! "$raw_id" =~ ^[0-9a-f]{64}$ ]]; then
    return 1
  fi
  printf 'sha256:%s' "$raw_id"
}

geosam2_immutable_image_reference() {
  local normalized_id="$1"
  if [[ ! "$normalized_id" =~ ^sha256:[0-9a-f]{64}$ ]]; then
    return 1
  fi
  printf '%s' "${normalized_id#sha256:}"
}
