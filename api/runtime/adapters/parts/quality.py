"""Ground-truth metrics for unordered, face-level part segmentation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from .regions import CandidateMask, PartSegmentationError


@dataclass(frozen=True)
class SegmentationScore:
    macro_iou: float
    face_coverage: float
    overlap_faces: int
    per_truth_iou: tuple[float, ...]


def _best_assignment(weights: list[list[float]]) -> list[tuple[int, int]]:
    """Maximum-weight rectangular assignment using the O(n^3) Hungarian method."""
    rows = len(weights)
    columns = len(weights[0]) if rows else 0
    if not rows or not columns:
        return []
    size = max(rows, columns)
    # Missing rows/columns represent unmatched parts with zero IoU.
    cost = [[-(weights[row][column] if row < rows and column < columns else 0.0)
             for column in range(size)] for row in range(size)]
    u = [0.0] * (size + 1)
    v = [0.0] * (size + 1)
    matched_row = [0] * (size + 1)
    predecessor = [0] * (size + 1)
    for row in range(1, size + 1):
        matched_row[0] = row
        current_column = 0
        minimum = [float("inf")] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[current_column] = True
            active_row = matched_row[current_column]
            delta = float("inf")
            next_column = 0
            for column in range(1, size + 1):
                if used[column]:
                    continue
                reduced = cost[active_row - 1][column - 1] - u[active_row] - v[column]
                if reduced < minimum[column]:
                    minimum[column] = reduced
                    predecessor[column] = current_column
                if minimum[column] < delta:
                    delta = minimum[column]
                    next_column = column
            for column in range(size + 1):
                if used[column]:
                    u[matched_row[column]] += delta
                    v[column] -= delta
                else:
                    minimum[column] -= delta
            current_column = next_column
            if matched_row[current_column] == 0:
                break
        while True:
            next_column = predecessor[current_column]
            matched_row[current_column] = matched_row[next_column]
            current_column = next_column
            if current_column == 0:
                break
    assignment = [(-1, -1)] * rows
    for column in range(1, size + 1):
        row = matched_row[column] - 1
        if 0 <= row < rows and column - 1 < columns:
            assignment[row] = (row, column - 1)
    return [item for item in assignment if item[1] >= 0]


def face_level_macro_iou(
    predicted: Sequence[CandidateMask],
    truth_labels: Sequence[int],
) -> SegmentationScore:
    """Compute macro IoU after optimal permutation matching of predicted IDs."""
    if not truth_labels:
        raise PartSegmentationError("INVALID_FIXTURE_TRUTH", "quality scoring needs at least one truth-labeled face")
    if any(type(label) is not int or label < 0 for label in truth_labels):
        raise PartSegmentationError("INVALID_FIXTURE_TRUTH", "truth labels must be nonnegative integer part IDs")
    pred_counts: Counter[int] = Counter()
    overlap: set[int] = set()
    for mask in predicted:
        for face_id in mask.face_ids:
            if type(face_id) is not int or not 0 <= face_id < len(truth_labels):
                raise PartSegmentationError("PART_FACE_OUT_OF_RANGE", "predicted mask references a face outside the known-truth fixture")
            if face_id in pred_counts:
                overlap.add(face_id)
            pred_counts[face_id] += 1
    truth_ids = sorted(set(truth_labels))
    pred_ids = list(range(len(predicted)))
    intersections = [[0 for _ in pred_ids] for _ in truth_ids]
    unions = [[0 for _ in pred_ids] for _ in truth_ids]
    for truth_row, truth_id in enumerate(truth_ids):
        truth_faces = {index for index, label in enumerate(truth_labels) if label == truth_id}
        for pred_col, candidate in enumerate(predicted):
            pred_faces = set(candidate.face_ids)
            intersections[truth_row][pred_col] = len(truth_faces & pred_faces)
            unions[truth_row][pred_col] = len(truth_faces | pred_faces)
    matching = _best_assignment([
        [intersections[r][c] / unions[r][c] if unions[r][c] else 0.0 for c in pred_ids]
        for r in range(len(truth_ids))
    ])
    score_by_truth_row = {
        row: intersections[row][column] / unions[row][column] if unions[row][column] else 0.0
        for row, column in matching
    }
    ious = [score_by_truth_row.get(row, 0.0) for row in range(len(truth_ids))]
    coverage = sum(1 for face_id in range(len(truth_labels)) if pred_counts[face_id] == 1) / len(truth_labels)
    return SegmentationScore(
        macro_iou=sum(ious) / len(truth_ids),
        face_coverage=coverage,
        overlap_faces=len(overlap),
        per_truth_iou=tuple(ious),
    )
