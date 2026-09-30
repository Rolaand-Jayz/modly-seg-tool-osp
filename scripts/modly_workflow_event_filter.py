"""Validate and forward Modly workflow JSONL with pinned GeoSAM2 log lines."""

from __future__ import annotations

import json
import re
import sys
from typing import TextIO


_KNOWN_UPSTREAM_LOG_PATTERNS = (
    re.compile(r"Found [0-9]+ objects"),
    re.compile(r"Using device: cuda"),
    re.compile(r"Compiling all components for VOS setting\. First time may be very slow\."),
    re.compile(r"Removing [0-9]+ small components"),
    re.compile(r"Split -?[0-9]+(?:\.[0-9]+)? component\(s\) into unique labels"),
    re.compile(
        r"Exported labelled mesh to /workspace/StructuredAssets/runs/"
        r"[0-9a-fA-F-]{1,80}/geosam2-inference/upstream-output/"
        r"segmentation_postprocessed_autoView[0-9]{2}_fromPrompt00_pa0\.02\.glb"
    ),
    # HIP attention kernels on the project ROCm image can write these exact
    # descriptor lines to stdout before otherwise valid workflow JSONL. Keep
    # the grammar narrow so unrelated plain-text output still fails closed.
    re.compile(r"a_grid_desc_m_ak_container_\{[0-9]+, [0-9]+\}"),
    re.compile(r"b_grid_desc_n_bk_container_\{[0-9]+, [0-9]+\}"),
    re.compile(
        r"e_grid_desc_mblock_mperblock_nblock_nperblock_container_"
        r"\{[0-9]+, [0-9]+\}"
    ),
)


def _is_known_upstream_log(text: str) -> bool:
    return text == "Smoothing labels" or any(
        pattern.fullmatch(text) is not None for pattern in _KNOWN_UPSTREAM_LOG_PATTERNS
    )


def filter_workflow_events(source: TextIO, destination: TextIO) -> int:
    """Forward output and return failure for errors or unknown plain-text lines."""
    completed = False
    failed = False
    for line in source:
        destination.write(line)
        destination.flush()
        # Remove only the record terminator. Whitespace is part of a plain log
        # line and must match the pinned upstream output exactly.
        text = line[:-1] if line.endswith("\n") else line
        try:
            event = json.loads(text)
        except json.JSONDecodeError:
            if not _is_known_upstream_log(text):
                failed = True
            continue
        if not isinstance(event, dict):
            failed = True
            continue
        event_type = event.get("type")
        if event_type == "progress":
            percent = event.get("percent")
            label = event.get("label")
            # These fields and bounds are emitted by the pinned workflow
            # processor's validation and completion progress messages.
            valid = (
                isinstance(percent, int)
                and not isinstance(percent, bool)
                and 0 <= percent <= 100
                and isinstance(label, str)
                and bool(label)
            )
            failed = failed or not valid
        elif event_type == "done":
            failed = failed or not isinstance(event.get("result"), dict)
            completed = completed or isinstance(event.get("result"), dict)
        elif event_type == "error":
            valid = (
                isinstance(event.get("code"), str)
                and bool(event.get("code"))
                and event.get("stage_id") == "reference-part-segmentation"
                and isinstance(event.get("message"), str)
                and bool(event.get("message"))
            )
            failed = True  # The processor's error event always means failure.
            failed = failed or not valid
        else:
            # Ready/loaded/cancelled/unloaded are runner protocol events, not
            # valid output from this one-shot workflow processor.
            failed = True
    return 1 if failed or not completed else 0


def main() -> int:
    return filter_workflow_events(sys.stdin, sys.stdout)


if __name__ == "__main__":
    raise SystemExit(main())
