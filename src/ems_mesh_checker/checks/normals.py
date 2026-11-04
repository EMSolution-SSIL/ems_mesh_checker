"""Orientation checks for explicit triangle/quad elements."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from ems_mesh_checker.topology.element_topology import get_element_topology
from ems_mesh_checker.topology.face_table import FaceTable, TopologyError

from .report import CheckFinding, CheckReport, CheckResult


@dataclass(frozen=True)
class _Surface:
    block_index: int
    local_index: int
    nodes: tuple[int, ...]
    element_id: int | None
    property_id: int | None


def _normal(points: np.ndarray, nodes: tuple[int, ...]) -> np.ndarray:
    xyz = np.zeros((len(nodes), 3), dtype=float)
    xyz[:, : points.shape[1]] = points[list(nodes)]
    normal = np.zeros(3)
    for index in range(1, len(nodes) - 1):
        normal += np.cross(xyz[index] - xyz[0], xyz[index + 1] - xyz[0])
    return normal


def _cell_values(mesh: Any, name: str, block_index: int) -> np.ndarray | None:
    blocks = getattr(mesh, "cell_data", {}).get(name)
    if blocks is None or len(blocks) != len(mesh.cells):
        return None
    array = np.asarray(blocks[block_index])
    if array.ndim != 1 or len(array) != len(mesh.cells[block_index].data) or not np.issubdtype(array.dtype, np.integer):
        return None
    return array


def add_normal_checks(
    mesh: Any,
    report: CheckReport,
    *,
    point_ids: np.ndarray | None,
    surface_owner_properties: dict[int, int] | None = None,
    surface_normal_direction: str = "outward",
    surface_normal_directions: dict[int, str] | None = None,
) -> None:
    """Compare input face order, never the extractor's normalized output order."""

    structural_errors = {
        "INVALID_CONNECTIVITY", "INVALID_NODE_REFERENCE", "INVALID_CELL_METADATA",
        "REPEATED_ELEMENT_NODE", "DUPLICATE_ELEMENT",
    }
    if any(item.code in structural_errors for item in report.findings):
        report.checks["normal_orientation"] = CheckResult(
            "NOT_RUN", "invalid connectivity prevents normal comparison",
        )
        return
    points = np.asarray(mesh.points, dtype=float)
    surfaces: list[_Surface] = []
    has_volume = False
    for block_index, block in enumerate(mesh.cells):
        topology = get_element_topology(str(block.type))
        if topology is None:
            continue
        if topology.dimension == 3 and len(block.data):
            has_volume = True
        if topology.dimension != 2:
            continue
        ids = _cell_values(mesh, "element_id", block_index)
        properties = _cell_values(mesh, "property_id", block_index)
        for local_index, row in enumerate(block.data):
            surfaces.append(_Surface(
                block_index, local_index, tuple(int(value) for value in row),
                None if ids is None else int(ids[local_index]),
                None if properties is None else int(properties[local_index]),
            ))
    if not surfaces:
        report.checks["normal_orientation"] = CheckResult(
            "NOT_RUN", "no explicit triangle or quad elements",
        )
        return

    findings: list[CheckFinding] = []
    edges: dict[tuple[int | None, int, int], list[tuple[int, int]]] = defaultdict(list)
    for face_index, surface in enumerate(surfaces):
        for start, end in zip(surface.nodes, surface.nodes[1:] + surface.nodes[:1]):
            key = (surface.property_id, min(start, end), max(start, end))
            edges[key].append((face_index, 1 if start < end else -1))
        if not np.isfinite(points[list(surface.nodes)]).all() or np.linalg.norm(_normal(points, surface.nodes)) == 0:
            findings.append(CheckFinding(
                "NORMAL_UNDEFINED", "WARNING", "surface normal cannot be determined",
                element_id=surface.element_id, property_id=surface.property_id,
                cell_block_index=surface.block_index, local_element_index=surface.local_index,
            ))
    for (_, start, end), owners in edges.items():
        if len(owners) == 2 and owners[0][1] == owners[1][1]:
            first, second = surfaces[owners[0][0]], surfaces[owners[1][0]]
            findings.append(CheckFinding(
                "SURFACE_ORIENTATION_CONFLICT", "WARNING",
                "adjacent surface elements traverse their shared edge in the same direction",
                node_ids=(int(point_ids[start]), int(point_ids[end])) if point_ids is not None else (),
                element_id=first.element_id,
                related_element_ids=(second.element_id,) if second.element_id is not None else (),
                property_id=first.property_id,
                cell_block_index=first.block_index, local_element_index=first.local_index,
            ))

    if has_volume:
        try:
            table = FaceTable.from_mesh(mesh, dimension=3)
        except TopologyError as error:
            report.checks["normal_orientation"] = CheckResult("NOT_RUN", str(error))
            return
        for surface in surfaces:
            nodes = surface.nodes
            if not np.isfinite(points[list(nodes)]).all():
                continue
            normal = _normal(points, nodes)
            if np.linalg.norm(normal) == 0:
                continue
            record = table.get(nodes)
            if record is None:
                findings.append(CheckFinding(
                    "SURFACE_WITHOUT_VOLUME_FACE", "WARNING",
                    "explicit surface has no exact matching volume face",
                    element_id=surface.element_id, property_id=surface.property_id,
                    cell_block_index=surface.block_index, local_element_index=surface.local_index,
                ))
                continue
            owners = record.owners
            selected_property = (surface_owner_properties or {}).get(surface.property_id)
            if len(owners) == 1:
                owner = owners[0]
                if selected_property is not None and owner.property_id != selected_property:
                    findings.append(CheckFinding(
                        "SURFACE_OWNER_MISMATCH", "WARNING",
                        "surface does not border its configured volume Property",
                        element_id=surface.element_id, property_id=surface.property_id,
                        cell_block_index=surface.block_index, local_element_index=surface.local_index,
                    ))
                    continue
            elif selected_property is None or sum(o.property_id == selected_property for o in owners) != 1:
                findings.append(CheckFinding(
                    "NORMAL_OWNER_AMBIGUOUS", "WARNING",
                    "more than one volume owns this face; specify the owning volume Property",
                    element_id=surface.element_id, property_id=surface.property_id,
                    cell_block_index=surface.block_index, local_element_index=surface.local_index,
                ))
                continue
            else:
                owner = next(o for o in owners if o.property_id == selected_property)
            volume_nodes = mesh.cells[owner.cell_block_index].data[owner.local_element_index]
            if not np.isfinite(points[list(volume_nodes)]).all():
                continue
            outward = _normal(points, owner.oriented_nodes)
            face_center = np.mean(points[list(owner.oriented_nodes)], axis=0)
            element_center = np.mean(points[list(volume_nodes)], axis=0)
            direction = np.zeros(3)
            direction[: points.shape[1]] = face_center - element_center
            if np.dot(outward, direction) < 0:
                outward = -outward
            if np.linalg.norm(outward) == 0 or np.dot(outward, direction) == 0:
                findings.append(CheckFinding(
                    "NORMAL_REFERENCE_UNDEFINED", "WARNING",
                    "volume face cannot establish a reliable outward direction",
                    element_id=surface.element_id, property_id=surface.property_id,
                    cell_block_index=surface.block_index, local_element_index=surface.local_index,
                ))
            else:
                expected_direction = (surface_normal_directions or {}).get(
                    surface.property_id, surface_normal_direction,
                )
                alignment = np.dot(normal, outward)
                if not ((expected_direction == "outward" and alignment < 0)
                        or (expected_direction == "inward" and alignment > 0)):
                    continue
                findings.append(CheckFinding(
                    "SURFACE_INWARD_NORMAL" if expected_direction == "outward" else "SURFACE_OUTWARD_NORMAL",
                    "WARNING",
                    "explicit surface normal points into the selected volume; expected outward"
                    if expected_direction == "outward" else
                    "explicit surface normal points out of the selected volume; expected inward",
                    element_id=surface.element_id,
                    related_element_ids=(owner.element_id,) if owner.element_id is not None else (),
                    property_id=surface.property_id,
                    cell_block_index=surface.block_index, local_element_index=surface.local_index,
                ))

    report.findings.extend(findings)
    unsupported = report.checks["supported_elements"].status == "UNSUPPORTED"
    reason = None
    if unsupported:
        reason = "unsupported element types were not checked"
    elif has_volume:
        reason = f"expected {surface_normal_direction} relative to selected volume"
        if surface_normal_directions:
            reason += f"; Property overrides: {dict(sorted(surface_normal_directions.items()))}"
    report.checks["normal_orientation"] = CheckResult(
        "UNSUPPORTED" if unsupported else "FINDINGS" if findings else "PASS",
        reason,
        len(findings),
    )
