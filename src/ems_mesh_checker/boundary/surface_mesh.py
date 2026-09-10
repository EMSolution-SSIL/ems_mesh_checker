"""Create meshio surface meshes while preserving source metadata."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np

from .result import BoundaryFace


def build_surface_mesh(mesh: Any, faces: Iterable[BoundaryFace]) -> Any:
    """Create a compact ``meshio.Mesh`` from selected boundary faces or edges."""

    import meshio

    ordered_faces = sorted(faces, key=lambda face: face.record.corner_key)
    source_indices: list[int] = []
    local_indices: dict[int, int] = {}
    grouped_faces: dict[str, list[BoundaryFace]] = defaultdict(list)
    for face in ordered_faces:
        cell_type = _surface_cell_type(face.oriented_nodes)
        grouped_faces[cell_type].append(face)
        for point_index in face.oriented_nodes:
            if point_index not in local_indices:
                local_indices[point_index] = len(source_indices)
                source_indices.append(point_index)

    points = np.asarray(mesh.points)
    point_dimension = points.shape[1] if points.ndim == 2 else 3
    compact_points = points[source_indices] if source_indices else np.empty((0, point_dimension))
    point_data = {
        "source_point_index": np.asarray(source_indices, dtype=int),
        "source_node_id": _source_node_ids(mesh, source_indices),
    }

    cells: list[tuple[str, np.ndarray]] = []
    metadata: dict[str, list[np.ndarray]] = defaultdict(list)
    pairs = sorted({face.property_pair for face in ordered_faces if face.property_pair is not None})
    pair_ids = {pair: index for index, pair in enumerate(pairs)}
    for cell_type in ("line", "triangle", "quad"):
        block_faces = grouped_faces.get(cell_type, [])
        if not block_faces:
            continue
        cells.append(
            (
                cell_type,
                np.asarray(
                    [[local_indices[index] for index in face.oriented_nodes] for face in block_faces],
                    dtype=int,
                ),
            )
        )
        metadata["owner_element_id"].append(_owner_metadata(block_faces, "element_id"))
        metadata["owner_property_id"].append(_owner_metadata(block_faces, "property_id"))
        metadata["neighbor_element_id"].append(_neighbor_metadata(block_faces, "element_id"))
        metadata["neighbor_property_id"].append(_neighbor_metadata(block_faces, "property_id"))
        metadata["property_pair_id"].append(
            np.asarray([pair_ids.get(face.property_pair, -1) for face in block_faces], dtype=int)
        )
        metadata["source_local_face_id"].append(
            np.asarray([face.owner.local_face_index for face in block_faces], dtype=int)
        )
        metadata["source_cell_block_index"].append(
            np.asarray([face.owner.cell_block_index for face in block_faces], dtype=int)
        )
        metadata["source_local_element_index"].append(
            np.asarray([face.owner.local_element_index for face in block_faces], dtype=int)
        )
    return meshio.Mesh(points=compact_points, cells=cells, point_data=point_data, cell_data=dict(metadata))


def _surface_cell_type(nodes: tuple[int, ...]) -> str:
    cell_types = {2: "line", 3: "triangle", 4: "quad"}
    try:
        return cell_types[len(nodes)]
    except KeyError as error:
        raise ValueError(f"unsupported surface entity with {len(nodes)} nodes") from error


def _source_node_ids(mesh: Any, source_indices: list[int]) -> np.ndarray:
    ids = getattr(mesh, "point_data", {}).get("id")
    if ids is None or len(ids) != len(mesh.points):
        return np.asarray(source_indices, dtype=int)
    return np.asarray(ids, dtype=int)[source_indices]


def _owner_metadata(faces: list[BoundaryFace], name: str) -> np.ndarray:
    return np.asarray([getattr(face.owner, name) if getattr(face.owner, name) is not None else -1 for face in faces])


def _neighbor_metadata(faces: list[BoundaryFace], name: str) -> np.ndarray:
    return np.asarray(
        [
            getattr(face.neighbor, name)
            if face.neighbor is not None and getattr(face.neighbor, name) is not None
            else -1
            for face in faces
        ]
    )
