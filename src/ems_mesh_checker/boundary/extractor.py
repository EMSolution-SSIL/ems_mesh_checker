"""Classify topology records into external and property-interface boundaries."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

import numpy as np

from ems_mesh_checker.topology.face import FaceOwner, FaceRecord
from ems_mesh_checker.topology.face_table import FaceTable

from .result import BoundaryFace, BoundaryResult, PropertyPair, QuadTriInterface


class BoundaryExtractionError(RuntimeError):
    """Raised when strict boundary extraction encounters a diagnostic warning."""


class BoundaryExtractor:
    """Extract external surfaces and interfaces from a ``meshio.Mesh``."""

    def __init__(
        self,
        mesh: Any,
        *,
        strict: bool = False,
        suppress_quad_tri_interfaces: bool = False,
    ) -> None:
        self.mesh = mesh
        self.strict = strict
        self.suppress_quad_tri_interfaces = suppress_quad_tri_interfaces

    def extract(self) -> BoundaryResult:
        """Classify all supported top-dimensional boundary entities."""

        face_table = FaceTable.from_mesh(self.mesh)
        external_faces: list[BoundaryFace] = []
        internal_faces: list[FaceRecord] = []
        property_interfaces: dict[PropertyPair, list[BoundaryFace]] = defaultdict(list)
        nonmanifold_faces: list[FaceRecord] = []
        unclassified_faces: list[FaceRecord] = []
        warnings: list[str] = []
        missing_property_reported = False
        quad_tri_interfaces = (
            _find_quad_tri_interfaces(face_table)
            if self.suppress_quad_tri_interfaces
            else ()
        )
        suppressed_keys = {
            record.corner_key
            for interface in quad_tri_interfaces
            for record in (interface.quad_face, *interface.triangle_faces)
        }

        for record in face_table.iter_records():
            if record.corner_key in suppressed_keys:
                continue
            if record.owner_count == 1:
                owner = record.owners[0]
                if owner.property_id is None:
                    if not missing_property_reported:
                        warnings.append(_warning("missing_property_id", record))
                        missing_property_reported = True
                external_faces.append(_boundary_face(self.mesh, record, owner, None))
                continue

            if record.owner_count >= 3:
                nonmanifold_faces.append(record)
                warnings.append(_warning("nonmanifold_face", record))
                continue

            first, second = record.owners
            if first.property_id is None or second.property_id is None:
                unclassified_faces.append(record)
                if not missing_property_reported:
                    warnings.append(_warning("missing_property_id", record))
                    missing_property_reported = True
                continue
            if first.property_id == second.property_id:
                internal_faces.append(record)
                continue

            owner, neighbor = sorted(
                (first, second), key=lambda candidate: (candidate.property_id, candidate.element_id or -1)
            )
            face = _boundary_face(self.mesh, record, owner, neighbor)
            assert face.property_pair is not None
            property_interfaces[face.property_pair].append(face)

        patches = {
            pair: _connected_patches(faces)
            for pair, faces in property_interfaces.items()
        }
        result = BoundaryResult.create(
            mesh=self.mesh,
            face_table=face_table,
            external_faces=external_faces,
            internal_faces=internal_faces,
            property_interfaces=property_interfaces,
            interface_patches=patches,
            nonmanifold_faces=nonmanifold_faces,
            unclassified_faces=unclassified_faces,
            suppressed_quad_tri_interfaces=list(quad_tri_interfaces),
            warnings=warnings,
        )
        if self.strict and result.warnings:
            raise BoundaryExtractionError("; ".join(result.warnings))
        return result


def _boundary_face(
    mesh: Any,
    record: FaceRecord,
    owner: FaceOwner,
    neighbor: FaceOwner | None,
) -> BoundaryFace:
    return BoundaryFace(
        record=record,
        owner=owner,
        neighbor=neighbor,
        oriented_nodes=_orient_outward(mesh, owner),
    )


def _find_quad_tri_interfaces(face_table: FaceTable) -> tuple[QuadTriInterface, ...]:
    """Find suppressible same-Property hexa-tetra coupling surfaces.

    Only single-owner faces are considered. The two triangles must use exactly
    the quad's four corner nodes and share one of the quad's two diagonals. This
    avoids suppressing an arbitrary pair of triangles that merely shares a
    perimeter edge and happens to reference the same four nodes. All three face
    owners must have the same non-missing Property ID: a quad/triangle coupling
    between different Properties is a real Property boundary and must remain.
    """

    triangle_records: dict[tuple[int, ...], FaceRecord] = {}
    quad_records: list[FaceRecord] = []
    for record in face_table.iter_records():
        if record.owner_count != 1:
            continue
        owner = record.owners[0]
        if owner.cell_type == "tetra" and len(record.corner_key) == 3:
            triangle_records[record.corner_key] = record
        elif owner.cell_type == "hexahedron" and len(record.corner_key) == 4:
            quad_records.append(record)

    interfaces: list[QuadTriInterface] = []
    for quad_record in quad_records:
        quad_owner = quad_record.owners[0]
        q0, q1, q2, q3 = quad_owner.oriented_nodes
        diagonals = {tuple(sorted((q0, q2))), tuple(sorted((q1, q3)))}
        candidates = [
            triangle_records[tuple(sorted(nodes))]
            for nodes in _three_node_subsets(quad_record.corner_key)
            if tuple(sorted(nodes)) in triangle_records
        ]
        pairs: list[tuple[FaceRecord, FaceRecord]] = []
        for first_index, first in enumerate(candidates):
            for second in candidates[first_index + 1 :]:
                first_nodes = set(first.corner_key)
                second_nodes = set(second.corner_key)
                if first_nodes | second_nodes != set(quad_record.corner_key):
                    continue
                shared_edge = tuple(sorted(first_nodes & second_nodes))
                if shared_edge in diagonals:
                    pairs.append((first, second))
        if len(pairs) == 1 and _has_one_property(quad_record, *pairs[0]):
            interfaces.append(
                QuadTriInterface(
                    quad_face=quad_record,
                    triangle_faces=tuple(sorted(pairs[0], key=lambda record: record.corner_key)),
                )
            )
    return tuple(sorted(interfaces, key=lambda interface: interface.quad_face.corner_key))


def _has_one_property(*records: FaceRecord) -> bool:
    property_ids = {record.owners[0].property_id for record in records}
    return None not in property_ids and len(property_ids) == 1


def _three_node_subsets(nodes: tuple[int, ...]):
    for omitted_index in range(4):
        yield tuple(node for index, node in enumerate(nodes) if index != omitted_index)


def _orient_outward(mesh: Any, owner: FaceOwner) -> tuple[int, ...]:
    """Orient a 3-D face away from its owner centroid; keep 2-D edges unchanged."""

    nodes = owner.oriented_nodes
    if len(nodes) < 3:
        return nodes

    points = np.asarray(mesh.points)
    face_points = points[np.asarray(nodes, dtype=int)]
    normal = np.cross(face_points[1] - face_points[0], face_points[2] - face_points[0])
    if np.linalg.norm(normal) == 0.0:
        return nodes

    element_nodes = mesh.cells[owner.cell_block_index].data[owner.local_element_index]
    element_centroid = np.mean(points[np.asarray(element_nodes, dtype=int)], axis=0)
    face_centroid = np.mean(face_points, axis=0)
    if np.dot(normal, face_centroid - element_centroid) < 0.0:
        return (nodes[0], *reversed(nodes[1:]))
    return nodes


def _connected_patches(faces: list[BoundaryFace]) -> tuple[tuple[BoundaryFace, ...], ...]:
    """Partition one property-pair interface into stable edge-connected patches."""

    ordered_faces = sorted(faces, key=lambda face: face.record.corner_key)
    adjacency: list[set[int]] = [set() for _ in ordered_faces]
    entity_owners: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for face_index, face in enumerate(ordered_faces):
        for entity in _adjacency_entities(face.oriented_nodes):
            entity_owners[entity].append(face_index)

    for face_indices in entity_owners.values():
        for face_index in face_indices:
            adjacency[face_index].update(other for other in face_indices if other != face_index)

    visited: set[int] = set()
    patches: list[tuple[BoundaryFace, ...]] = []
    for start in range(len(ordered_faces)):
        if start in visited:
            continue
        queue: deque[int] = deque([start])
        visited.add(start)
        indices: list[int] = []
        while queue:
            current = queue.popleft()
            indices.append(current)
            for next_index in sorted(adjacency[current]):
                if next_index not in visited:
                    visited.add(next_index)
                    queue.append(next_index)
        patches.append(tuple(ordered_faces[index] for index in sorted(indices)))
    return tuple(patches)


def _adjacency_entities(nodes: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    if len(nodes) == 2:
        return tuple((node,) for node in nodes)
    return tuple(
        tuple(sorted((nodes[index], nodes[(index + 1) % len(nodes)])))
        for index in range(len(nodes))
    )


def _warning(code: str, record: FaceRecord) -> str:
    return f"{code}: corner_key={record.corner_key}"
