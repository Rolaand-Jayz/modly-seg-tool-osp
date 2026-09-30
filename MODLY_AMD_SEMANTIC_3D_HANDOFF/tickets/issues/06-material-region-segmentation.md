# 06 — Material-region segmentation independent of parts

**What to build:** Take a Structured Asset plus available observations and assign topology-bound surface regions that represent candidate material boundaries through `Segment Material Regions`, independently of part segmentation, material naming, and PBR estimation. Select the reference strategy by probing a dedicated 3D material-segmentation path and a multi-view projection/back-projection path when viable.

**Blocked by:** 01 — Structured Asset headless round-trip; 02 — AMD Runtime proof through Modly.

**Status:** acceptance-passed (2026-09-24)

- [x] Before final comparison, material-region correctness/quality thresholds and the selection rubric were declared. MaterialSeg3D was screened against its published CUDA/native dependencies and missing immutable weights/AMD-resource evidence; no target resource claim is made for the rejected candidate. The multi-view projection/back-projection strategy was compared against the fixed fixture and selected by AMD fit, 16 GB suitability, exact mapping quality, and integration cost. No unsupported relative quality comparison is claimed.
- [x] The Modly process extension emits material-region IDs plus face mappings tied to the exact topology revision and source evidence.
- [x] A known multi-material fixture has one mesh, one whole-object part, and no glTF material/PBR slots; the adapter emits two independent material regions. Fixture mean IoU, coverage, and boundary F1 each score 1.0. A second process fixture partitions the same four faces into two semantic parts, with each part containing both materials and each material region spanning both parts, as required by the audited spec.
- [x] Output includes model-inferred assertions, upstream segmenter and observation provenance, uncalibrated agreement confidence, unknown/unmapped state for unseen faces, and a content-addressed calibrated-view artifact preserving the exact masks and matrices.
- [x] A changed-topology fixture preserves old regions only as invalid empty mappings and emits new mappings for the current topology revision.

Validation: `PYTHONPATH=.modly-amd-runtime/api-test-venv/lib/python3.12/site-packages:api .modly-amd-runtime/api-test-venv/bin/python -m unittest discover -s api/tests -p test_ticket06_material_regions.py -v` passed 7/7 on Python 3.12, including the multi-part/multi-material crossing case. The selected adapter consumes calibrated labels from a replaceable upstream 2D segmenter; raw-image label inference is not part of Ticket 06 and remains an explicit integration dependency for the end-to-end gate.
