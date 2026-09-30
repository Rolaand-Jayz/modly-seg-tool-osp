"""Diagnostic run-to-run comparison for registered GeoSAM2 outputs.

This report is descriptive only. It never reads fixture truth or assigns a
quality score. `None` means the evidence needed for a measure was unavailable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

SENTINELS = {-1, 999}


def _load_run(manifest_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.resolve(strict=True)
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    def artifact(key: str) -> np.ndarray | None:
        item = manifest.get(key)
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            return None
        path = (root / item["path"]).resolve(strict=False)
        if root not in path.parents:
            raise ValueError(f"artifact path escapes run directory: {key}")
        if not path.is_file():
            return None
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = str(item.get("sha256", "")).removeprefix("sha256:")
        if expected and digest != expected:
            raise ValueError(f"artifact digest mismatch: {key}")
        return np.load(path, allow_pickle=False).reshape(-1)

    arrays = {
        "raw": artifact("upstream_label_artifact"),
        "completed": artifact("completed_label_artifact"),
        "canonical": artifact("label_artifact"),
        "fill_mask": artifact("unassigned_fill_mask_artifact"),
    }
    declared = [int(x["upstream_label_id"]) for x in manifest.get("regions", [])
                if isinstance(x, dict) and isinstance(x.get("upstream_label_id"), int)]
    face_count = manifest.get("face_count")
    row: dict[str, Any] = {
        "run_id": manifest.get("run_id"),
        "manifest": str(manifest_path),
        "face_count": face_count if isinstance(face_count, int) else None,
        "topology_revision": manifest.get("topology_revision"),
        "confidence_state": manifest.get("confidence", {}).get("state", "unknown"),
        "overlap": {"state": "unknown", "reason": "exclusive face-label arrays cannot reveal source-region overlap"},
        "missing_regions": None,
        "arrays": {},
    }
    raw = arrays["raw"]
    if raw is not None:
        regular = set(int(v) for v in np.unique(raw) if int(v) not in SENTINELS)
        row["missing_regions"] = sorted(set(declared) - regular)
    for name, values in arrays.items():
        if values is None:
            row["arrays"][name] = {"state": "unavailable"}
            continue
        sent = np.isin(values, list(SENTINELS)) if name in {"raw", "completed", "canonical"} else np.zeros(values.size, dtype=bool)
        valid = ~sent if name in {"raw", "completed", "canonical"} else np.ones(values.size, dtype=bool)
        labels = values[valid]
        row["arrays"][name] = {
            "state": "available",
            "item_count": int(values.size),
            "coverage_count": int(valid.sum()),
            "coverage_fraction": float(valid.mean()) if values.size else None,
            "sentinel_counts": {str(s): int(np.count_nonzero(values == s)) for s in sorted(SENTINELS)} if name in {"raw", "completed", "canonical"} else None,
            "region_count": int(np.unique(labels).size) if name in {"raw", "completed", "canonical"} else None,
            "fill_count": int(np.count_nonzero(values)) if name == "fill_mask" else None,
        }
    return {"_arrays": arrays, **row}


def _ari(a: np.ndarray, b: np.ndarray) -> float | None:
    """Adjusted Rand index, treating sentinel IDs as explicit categories."""
    if a.size != b.size or not a.size:
        return None
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    table = np.zeros((int(ai.max()) + 1, int(bi.max()) + 1), dtype=np.int64)
    np.add.at(table, (ai, bi), 1)
    choose2 = lambda x: x * (x - 1) // 2
    total = int(choose2(np.int64(a.size)))
    if total == 0:
        return 1.0 if np.array_equal(ai, bi) else 0.0
    sum_cells = int(choose2(table).sum())
    sum_rows = int(choose2(table.sum(axis=1)).sum())
    sum_cols = int(choose2(table.sum(axis=0)).sum())
    expected = sum_rows * sum_cols / total
    maximum = (sum_rows + sum_cols) / 2
    return 1.0 if maximum == expected else float((sum_cells - expected) / (maximum - expected))


def _maximum_matched_agreement(a: np.ndarray, b: np.ndarray) -> float | None:
    """Face agreement after maximum-overlap one-to-one region matching."""
    if a.size != b.size or not a.size:
        return None
    av, ai = np.unique(a, return_inverse=True)
    bv, bi = np.unique(b, return_inverse=True)
    n, m = len(av), len(bv)
    weights = np.zeros((n, m), dtype=np.int64)
    np.add.at(weights, (ai, bi), 1)
    # Rectangular Hungarian algorithm, padded with zero-weight dummy regions.
    size = max(n, m)
    cost = np.zeros((size + 1, size + 1), dtype=np.int64)
    cost[1:n+1, 1:m+1] = -weights
    u = np.zeros(size + 1, dtype=np.int64); v = np.zeros(size + 1, dtype=np.int64)
    p = np.zeros(size + 1, dtype=np.int64); way = np.zeros(size + 1, dtype=np.int64)
    for i in range(1, size + 1):
        p[0] = i; j0 = 0; minv = np.full(size + 1, np.iinfo(np.int64).max, dtype=np.int64); used = np.zeros(size + 1, dtype=bool)
        while True:
            used[j0] = True; i0 = p[j0]; delta = np.iinfo(np.int64).max; j1 = 0
            for j in range(1, size + 1):
                if not used[j]:
                    cur = cost[i0, j] - u[i0] - v[j]
                    if cur < minv[j]: minv[j] = cur; way[j] = j0
                    if minv[j] < delta: delta = minv[j]; j1 = j
            for j in range(size + 1):
                if used[j]: u[p[j]] += delta; v[j] -= delta
                else: minv[j] -= delta
            j0 = j1
            if p[j0] == 0: break
        while True:
            j1 = way[j0]; p[j0] = p[j1]; j0 = j1
            if j0 == 0: break
    matches = 0
    for j in range(1, size + 1):
        if 1 <= p[j] <= n and j <= m:
            matches += int(weights[p[j]-1, j-1])
    return matches / int(a.size)


def compare_manifests(paths: list[str | Path]) -> dict[str, Any]:
    if len(paths) < 2:
        raise ValueError("provide at least two run manifests")
    runs = [_load_run(Path(p)) for p in paths]
    summaries = [{k: v for k, v in run.items() if k != "_arrays"} for run in runs]
    pairs = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            left, right = runs[i], runs[j]
            pair: dict[str, Any] = {"left": i, "right": j, "metrics": {}}
            same_topology = (left["topology_revision"] is not None and left["topology_revision"] == right["topology_revision"])
            pair["topology_matches"] = same_topology
            for name in ("raw", "completed", "canonical"):
                a, b = left["_arrays"][name], right["_arrays"][name]
                metric: dict[str, Any] = {"state": "unknown"}
                if a is not None and b is not None and same_topology and a.size == b.size:
                    common_assigned = ~np.isin(a, list(SENTINELS)) & ~np.isin(b, list(SENTINELS))
                    aa, bb = a[common_assigned], b[common_assigned]
                    metric = {"state": "measured" if aa.size else "unknown",
                              "compared_face_count": int(aa.size),
                              "excluded_faces_with_sentinel_in_either_run": int(a.size - aa.size),
                              "face_agreement": float(np.mean(aa == bb)) if aa.size else None,
                              "maximum_overlap_matched_agreement": _maximum_matched_agreement(aa, bb),
                              "adjusted_rand_index": _ari(aa, bb),
                              "left_region_count": int(np.unique(aa).size) if aa.size else None,
                              "right_region_count": int(np.unique(bb).size) if bb.size else None}
                elif not same_topology:
                    metric["reason"] = "topology revisions do not match or are unavailable"
                else:
                    metric["reason"] = "label artifact unavailable or face counts differ"
                pair["metrics"][name] = metric
            a, b = left["_arrays"]["fill_mask"], right["_arrays"]["fill_mask"]
            pair["fill_mask"] = ({"state": "measured", "differing_faces": int(np.count_nonzero(a != b)),
                                  "item_count": int(a.size)} if a is not None and b is not None and same_topology and a.size == b.size
                                 else {"state": "unknown", "reason": "fill mask unavailable or topology/face counts differ"})
            pairs.append(pair)
    return {"schema": "modly.ticket04-run-variation-diagnostic/1", "diagnostic_only": True,
            "quality_score": None, "truth_accessed": False, "runs": summaries, "pairs": pairs}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare registered GeoSAM2 runs; diagnostic only, no truth or quality score.")
    parser.add_argument("manifests", nargs="+", help="paths to geosam2-inference/segmentation-manifest.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = compare_manifests(args.manifests)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
