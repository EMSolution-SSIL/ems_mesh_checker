"""Unmodified free faces and auditable connectivity candidates."""

from __future__ import annotations

from collections import defaultdict
from itertools import product
from typing import Any

import numpy as np

from ems_mesh_checker.topology.face_table import FaceTable, TopologyError

from .report import CheckFinding, CheckReport, CheckResult


def _coordinate_groups(points: np.ndarray, indices: set[int], tolerance: float) -> tuple[dict[int, int], list[list[int]]]:
    """Group coincident/nearby indices with a spatial hash, not an all-pairs search."""

    parent = {index: index for index in indices}

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def join(first: int, second: int) -> None:
        a, b = root(first), root(second)
        if a != b:
            parent[max(a, b)] = min(a, b)

    if tolerance == 0.0:
        buckets: dict[tuple[float, ...], int] = {}
        for index in sorted(indices):
            coordinate = points[index]
            if not np.isfinite(coordinate).all():
                continue
            key = tuple(float(value) for value in coordinate)
            previous = buckets.setdefault(key, index)
            join(index, previous)
    else:
        buckets2: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for index in sorted(indices):
            coordinate = points[index]
            if not np.isfinite(coordinate).all():
                continue
            key = tuple(int(value) for value in np.floor(coordinate / tolerance))
            for offset in product((-1, 0, 1), repeat=points.shape[1]):
                neighbor = tuple(key[axis] + offset[axis] for axis in range(len(key)))
                for other in buckets2.get(neighbor, ()):
                    if np.linalg.norm(coordinate - points[other]) <= tolerance:
                        join(index, other)
            buckets2[key].append(index)

    groups: dict[int, list[int]] = defaultdict(list)
    for index in sorted(indices):
        groups[root(index)].append(index)
    return {index: root(index) for index in indices}, [group for group in groups.values() if len(group) > 1]


def _patches(faces: list[Any], dimension: int, points: np.ndarray) -> tuple[list[dict[str, Any]], list[int]]:
    parent = list(range(len(faces)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    edge_owners: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for index, record in enumerate(faces):
        nodes = record.owners[0].oriented_nodes
        if dimension == 3:
            entities = [tuple(sorted((start, end))) for start, end in zip(nodes, nodes[1:] + nodes[:1])]
        else:
            entities = [(node,) for node in nodes]
        for entity in entities:
            edge_owners[entity].append(index)
    for owners in edge_owners.values():
        for other in owners[1:]:
            parent[root(other)] = root(owners[0])

    grouped: dict[int, list[int]] = defaultdict(list)
    for index in range(len(faces)):
        grouped[root(index)].append(index)
    boundary_counts: dict[int, int] = defaultdict(int)
    for owners in edge_owners.values():
        if len(owners) != 2:
            boundary_counts[root(owners[0])] += 1
    patches: list[dict[str, Any]] = []
    face_patch_indices = [0] * len(faces)
    for patch_index, (component_root, members) in enumerate(grouped.items(), start=1):
        boundary_entities = boundary_counts[component_root]
        node_indices = sorted({
            node for face_index in members for node in faces[face_index].owners[0].oriented_nodes
        })
        coordinates = points[node_indices]
        bounds = None if not np.isfinite(coordinates).all() else {
            "minimum": coordinates.min(axis=0).tolist(),
            "maximum": coordinates.max(axis=0).tolist(),
        }
        if bounds is not None:
            bounds["size"] = (np.asarray(bounds["maximum"]) - bounds["minimum"]).tolist()
        for face_index in members:
            face_patch_indices[face_index] = patch_index
        patches.append({
            "patch_index": patch_index,
            "face_count": len(members),
            "closed": boundary_entities == 0,
            "boundary_entity_count": boundary_entities,
            "first_face_index": members[0],
            "node_count": len(node_indices),
            "bounds": bounds,
        })
    return patches, face_patch_indices


def _enclosed_shell_candidates(patches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Use strict bounding-box containment as a clue, never a hole verdict."""
    closed = [patch for patch in patches if patch["closed"] and patch["bounds"] is not None]
    candidates: list[dict[str, Any]] = []
    for patch in closed:
        minimum = np.asarray(patch["bounds"]["minimum"])
        maximum = np.asarray(patch["bounds"]["maximum"])
        enclosing = []
        for outer in closed:
            if outer is patch:
                continue
            outer_minimum = np.asarray(outer["bounds"]["minimum"])
            outer_maximum = np.asarray(outer["bounds"]["maximum"])
            if np.all(outer_minimum < minimum) and np.all(maximum < outer_maximum):
                enclosing.append(outer)
        if enclosing:
            nearest = min(enclosing, key=lambda item: np.prod(item["bounds"]["size"]))
            candidates.append({
                "patch_index": patch["patch_index"],
                "enclosing_patch_index": nearest["patch_index"],
                "first_face_index": patch["first_face_index"],
                "face_count": patch["face_count"],
                "bounds": patch["bounds"],
            })
    return candidates


def add_connectivity_checks(
    mesh: Any,
    report: CheckReport,
    *,
    point_ids: np.ndarray | None,
    coordinate_tolerance: float = 0.0,
) -> None:
    """Add Phase 2 results; free faces alone are not classified as defects."""

    if coordinate_tolerance < 0 or not np.isfinite(coordinate_tolerance):
        raise ValueError("coordinate_tolerance must be finite and non-negative")
    structural_errors = {
        "INVALID_CONNECTIVITY", "INVALID_NODE_REFERENCE", "INVALID_CELL_METADATA",
        "REPEATED_ELEMENT_NODE", "DUPLICATE_ELEMENT",
    }
    if any(item.code in structural_errors for item in report.findings):
        reason = "invalid connectivity or cell metadata prevents a reliable face table"
        report.checks["connectivity"] = CheckResult("NOT_RUN", reason)
        report.checks["free_faces"] = CheckResult("NOT_RUN", reason)
        return
    try:
        table = FaceTable.from_mesh(mesh)
    except TopologyError as error:
        report.checks["connectivity"] = CheckResult("NOT_RUN", str(error))
        report.checks["free_faces"] = CheckResult("NOT_RUN", str(error))
        return

    points = np.asarray(mesh.points, dtype=float)
    records = list(table.iter_records())
    free_faces = [record for record in records if record.owner_count == 1]
    nonmanifold = [record for record in records if record.owner_count > 2]
    indices = {index for record in records for index in record.corner_key}
    representatives, coordinate_groups = _coordinate_groups(points, indices, coordinate_tolerance)
    new_findings: list[CheckFinding] = []
    for group in coordinate_groups:
        new_findings.append(CheckFinding(
            "COINCIDENT_NODE_CANDIDATE", "WARNING",
            f"{len(group)} distinct point indices share coordinates within tolerance {coordinate_tolerance:g}",
            node_ids=tuple(int(point_ids[index]) for index in group) if point_ids is not None else (),
            local_node_index=group[0],
        ))
    for record in nonmanifold:
        new_findings.append(CheckFinding(
            "NONMANIFOLD_FACE", "WARNING",
            f"boundary entity has {record.owner_count} owning elements",
            node_ids=tuple(int(point_ids[index]) for index in record.corner_key) if point_ids is not None else (),
            element_id=record.owners[0].element_id,
            related_element_ids=tuple(owner.element_id for owner in record.owners[1:] if owner.element_id is not None),
            property_id=record.owners[0].property_id,
        ))

    geometric_faces: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for index, record in enumerate(free_faces):
        if all(np.isfinite(points[node]).all() for node in record.corner_key):
            geometric_faces[tuple(sorted(representatives[node] for node in record.corner_key))].append(index)
    coincident_groups = []
    for members in geometric_faces.values():
        if len(members) < 2:
            continue
        owners = [free_faces[index].owners[0] for index in members]
        coincident_groups.append(members)
        new_findings.append(CheckFinding(
            "COINCIDENT_UNSHARED_FACES", "WARNING",
            f"{len(members)} free faces coincide geometrically but do not share node indices",
            element_id=owners[0].element_id,
            related_element_ids=tuple(owner.element_id for owner in owners[1:] if owner.element_id is not None),
            property_id=owners[0].property_id,
            cell_block_index=owners[0].cell_block_index,
            local_element_index=owners[0].local_element_index,
        ))

    patches, face_patch_indices = _patches(free_faces, table.source_dimension, points)
    candidates = _enclosed_shell_candidates(patches) if table.source_dimension == 3 else []
    for candidate in candidates:
        owner = free_faces[candidate["first_face_index"]].owners[0]
        bounds = candidate["bounds"]
        center = tuple((low + high) / 2 for low, high in zip(bounds["minimum"], bounds["maximum"]))
        new_findings.append(CheckFinding(
            "ENCLOSED_FREE_FACE_SHELL_CANDIDATE", "INFO",
            "closed free-face shell lies inside another shell's bounding box; inspect whether it is an intended cavity",
            element_id=owner.element_id, property_id=owner.property_id,
            cell_block_index=owner.cell_block_index,
            local_element_index=owner.local_element_index,
            coordinates=center,
        ))
    report.raw_free_faces = [
        {
            "face_index": index,
            "patch_index": face_patch_indices[index],
            "point_indices": list(record.owners[0].oriented_nodes),
            "node_ids": [int(point_ids[node]) for node in record.owners[0].oriented_nodes] if point_ids is not None else None,
            "element_id": record.owners[0].element_id,
            "property_id": record.owners[0].property_id,
            "cell_block_index": record.owners[0].cell_block_index,
            "local_element_index": record.owners[0].local_element_index,
            "local_face_index": record.owners[0].local_face_index,
        }
        for index, record in enumerate(free_faces)
    ]
    report.topology = {
        "source_dimension": table.source_dimension,
        "source_element_count": table.num_source_elements,
        "face_count": len(records),
        "free_face_count": len(free_faces),
        "nonmanifold_face_count": len(nonmanifold),
        "coincident_node_group_count": len(coordinate_groups),
        "coincident_unshared_face_group_count": len(coincident_groups),
        "coordinate_tolerance": coordinate_tolerance,
        "free_face_patches": patches,
        "enclosed_free_face_shell_candidates": candidates,
        "notes": "Bounding-box-contained closed shells are review candidates, not proof of cavities or holes.",
    }
    report.findings.extend(new_findings)
    has_unsupported = report.checks["supported_elements"].status == "UNSUPPORTED"
    status = "UNSUPPORTED" if has_unsupported else "FINDINGS" if new_findings else "PASS"
    reason = "unsupported elements are omitted from the face table" if has_unsupported else None
    report.checks["connectivity"] = CheckResult(status, reason, len(new_findings))
    report.checks["free_faces"] = CheckResult(
        "UNSUPPORTED" if has_unsupported else "PASS", reason,
    )
