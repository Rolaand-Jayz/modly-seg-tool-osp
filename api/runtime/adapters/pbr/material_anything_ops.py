"""Source-defined tensor operations used by Material Anything's mesh path.

This module ports only Kaolin's ``index_vertices_by_faces`` gather from the
pinned Material Anything call site. It deliberately does not rasterize UVs or
claim parity for Kaolin's rasterizer.
"""
from __future__ import annotations

import torch


MATERIAL_ANYTHING_COMMIT = "be3d6b32a195f968540abc2ee106dc02d4b07479"
KAOLIN_COMMIT = "06ffb7d955ca26b608c60a9e862327c56b226921"


def index_vertices_by_faces(vertices_features: torch.Tensor,
                            faces: torch.Tensor) -> torch.Tensor:
    """Return Kaolin-compatible per-face vertex features.

    The pinned Kaolin operator accepts batched floating point features with
    shape ``(B, V, K)`` and one unbatched ``torch.long`` face-index tensor
    ``(F, N)``. It returns ``(B, F, N, K)``. Material Anything supplies
    triangles (``N == 3``), but the underlying source operator permits any
    face width, so this function retains that behavior.

    This implementation uses PyTorch gather with the same index expansion as
    Kaolin v0.18.0. It does not normalize face order or merge duplicated
    vertices, so seams and per-face row alignment are preserved.
    """
    if not isinstance(vertices_features, torch.Tensor):
        raise TypeError("vertices_features must be a torch.Tensor")
    if not isinstance(faces, torch.Tensor):
        raise TypeError("faces must be a torch.Tensor")
    if vertices_features.ndim != 3:
        raise ValueError("vertices_features must have shape (batch, vertices, channels)")
    if faces.ndim != 2:
        raise ValueError("faces must have shape (faces, vertices_per_face)")
    if not vertices_features.is_floating_point():
        raise TypeError("vertices_features must have a floating point dtype")
    if faces.dtype != torch.long:
        raise TypeError("faces must have dtype torch.long")
    if vertices_features.device != faces.device:
        raise ValueError("vertices_features and faces must be on the same device")
    if vertices_features.shape[0] == 0:
        raise ValueError("vertices_features batch dimension must be non-empty")
    if vertices_features.shape[1] == 0:
        raise ValueError("vertices_features vertex dimension must be non-empty")
    if faces.shape[1] == 0:
        raise ValueError("faces vertex-per-face dimension must be non-empty")
    if faces.numel() and (torch.any(faces < 0) or torch.any(faces >= vertices_features.shape[1])):
        raise ValueError("faces contains an index outside the vertex-feature array")

    expanded_features = vertices_features.unsqueeze(2).expand(
        -1, -1, faces.shape[1], -1,
    )
    gather_indices = faces[None, ..., None].expand(
        vertices_features.shape[0], -1, -1, vertices_features.shape[-1],
    )
    return torch.gather(expanded_features, dim=1, index=gather_indices)


def gather_material_anything_face_inputs(
    verts_uvs: torch.Tensor,
    faces_uvs: torch.Tensor,
    verts_xyz: torch.Tensor,
    faces: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather UV and XYZ triangle corners while preserving original face rows.

    Inputs mirror the pinned caller's unbatched PyTorch3D mesh arrays. The
    resulting UV and XYZ face tensors each have a batch dimension of one,
    matching the source's ``unsqueeze(0)`` use of ``index_vertices_by_faces``.
    Separate index arrays are required because UV seams duplicate UV vertices
    while geometric faces keep their original vertex indices.
    """
    if not isinstance(faces_uvs, torch.Tensor) or not isinstance(faces, torch.Tensor):
        raise TypeError("faces_uvs and faces must be torch.Tensor values")
    if not isinstance(verts_uvs, torch.Tensor) or not isinstance(verts_xyz, torch.Tensor):
        raise TypeError("verts_uvs and verts_xyz must be torch.Tensor values")
    if faces_uvs.ndim != 2 or faces.ndim != 2:
        raise ValueError("faces_uvs and faces must each have shape (faces, 3)")
    if faces_uvs.shape[1] != 3 or faces.shape[1] != 3:
        raise ValueError("Material Anything inputs must contain triangle faces")
    if faces_uvs.shape[0] != faces.shape[0]:
        raise ValueError("UV and geometry face rows must align one-for-one")
    if verts_uvs.ndim != 2 or verts_xyz.ndim != 2:
        raise ValueError("verts_uvs and verts_xyz must be unbatched vertex-feature arrays")
    if verts_uvs.shape[1] != 2 or verts_xyz.shape[1] != 3:
        raise ValueError("UV vertices need 2 channels and geometry vertices need 3")
    if verts_uvs.device != faces_uvs.device or verts_xyz.device != faces.device:
        raise ValueError("each face-index tensor must share a device with its vertex features")

    uv_corners = index_vertices_by_faces(verts_uvs.unsqueeze(0), faces_uvs)
    xyz_corners = index_vertices_by_faces(verts_xyz.unsqueeze(0), faces)
    return uv_corners, xyz_corners
