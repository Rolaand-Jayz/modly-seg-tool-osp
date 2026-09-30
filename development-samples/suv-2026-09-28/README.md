# User-supplied SUV development sample

The source is the exact PNG supplied in the conversation on 2026-09-28. It is a
1536 × 1024 image with SHA-256
`f5092fde7f8a18a2f0660fea6ac157c3826a7fdc485874e74ea91694aa8381af`.
It is a development observation, not a frozen acceptance fixture.

Modly's `reference-geometry` process extension generated the original mesh
through the pinned Hunyuan3D-2 Mini/Turbo adapter on the RX 7900 GRE. It used
seed 1337, balanced detail, and explicit PyTorch ROCm with model CPU offload.
`result.json` records the full adapter result and provenance. The original GLB
is `GeneratedGeometry/6d11c55c-1ef1-4550-b8f1-ffda20dc111a.glb`, with
1,108,620 triangles, 554,053 vertices, and SHA-256
`a9519460b1367523a9c340f452537f9a5cc8115bda48faa1a8e3745d908a975a`.
Its Structured Asset is under `StructuredAssets/runs/6d11c55c-1ef1-4550-b8f1-ffda20dc111a/`.
This is shape geometry only; the source image's colors and materials are not
embedded as PBR properties.

## Selected lower-detail mesh

`lod-optimized/suv-75000-faces.glb` is the selected development mesh. Blender
4.0.2 applied quadric edge-collapse to a copy of the original, then used
smooth normals with 45-degree sharp-edge preservation. The exported GLB has
75,000 triangles and 40,184 vertices. That is 93.23% fewer triangles and
92.75% fewer vertices than the original. It is 1,415,328 bytes, 92.91% smaller
than the original GLB. Its SHA-256 is
`39eae91adc8dc42f5ab896579d666aa00c806b80ce1d9a4f0fdadcaea5dbed68`.

The `lod-optimized/` views show the three-quarter front, side, and rear. The
wheels, arches, windows, bumper, spare tire, roof rack, cargo, and exhaust
remain visually recognizable. The 50,000-triangle candidate was visibly more
faceted. The 100,000-triangle candidate was smoother but provided a smaller
visible gain for this segmentation development case. Candidate files remain
under `lod-candidates/`; the first flat-shaded 75,000-triangle candidate is
retained for comparison and is not selected.

The selected mesh has a new topology revision,
`sha256:f65ea5a3000e4b366a8628a7360c7164f8b6fa4524b4186d5ba1ce1e0ae0a057`.
Its validated Structured Asset is
`StructuredAssets/runs/2c07f3e3-c6a5-4820-adbd-d4533a5f7d67/8c8c5d82-d525-43aa-9eb5-b9f460bb34a0.structured-asset.json`.
That sidecar keeps the source image and original generated GLB as lineage,
records the decimation method and script/Blender digests, and carries no face
mapping from the original topology.

## Segmentation state

The first GeoSAM2 development run used the original 1,108,620-triangle mesh.
It reached twelve renders and a face correspondence file. The user requested
that all processes stop before inference finished. The run exited with status
130; logs are under `segmentation-logs/` and partial artifacts are under
`StructuredAssets/runs/00177a90-724d-4116-af3c-f9e3192abfe4/`. They do not
establish part segmentation quality. Those original-topology artifacts cannot
be applied to the selected 75,000-triangle mesh. A future segmentation run
must render and map the selected mesh afresh.

Two setup errors before the successful geometry generation are preserved as
`attempt-01-failed.json` and `attempt-02-failed.json`. The successful run used
the project-managed AMD image and dependencies; no user custom installation
was used.
