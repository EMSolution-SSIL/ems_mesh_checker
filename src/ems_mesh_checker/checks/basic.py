"""Phase 0/1 checks: readable input, provenance, and dimensions.

No geometry-quality result is inferred from these inexpensive structural checks.
A separately built private backend owns area, volume, and Jacobian decisions.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from ems_mesh_checker.topology.element_topology import ElementTopology, get_element_topology

from .connectivity import add_connectivity_checks
from .design_dimensions import add_design_dimension_checks
from .geometry import add_geometry_checks
from .normals import add_normal_checks
from .report import CheckFinding, CheckReport, CheckResult, MeshCheckError


@dataclass(frozen=True)
class _Cell:
    block_index: int
    local_index: int
    topology: ElementTopology
    nodes: tuple[int, ...]
    element_id: int | None
    property_id: int | None


def _ids(mesh: Any, name: str, count: int, report: CheckReport) -> np.ndarray | None:
    raw = getattr(mesh, "point_data", {}).get(name)
    if raw is None:
        report.findings.append(CheckFinding(f"MISSING_{name.upper()}", "WARNING", f"point_data[{name!r}] is missing"))
        return None
    array = np.asarray(raw)
    if array.ndim != 1 or len(array) != count or not np.issubdtype(array.dtype, np.integer):
        report.findings.append(CheckFinding("INVALID_POINT_IDS", "ERROR", f"point_data[{name!r}] must contain {count} integer IDs"))
        return None
    seen: dict[int, int] = {}
    for index, value in enumerate(array):
        identifier = int(value)
        if identifier in seen:
            report.findings.append(CheckFinding(
                "DUPLICATE_NODE_ID", "ERROR", f"original node ID {identifier} is repeated",
                node_ids=(identifier,), local_node_index=index,
            ))
        else:
            seen[identifier] = index
    return array


def _cell_ids(mesh: Any, name: str, report: CheckReport) -> list[np.ndarray | None]:
    raw = getattr(mesh, "cell_data", {}).get(name)
    if raw is None:
        report.findings.append(CheckFinding(
            f"MISSING_{name.upper()}", "WARNING", f"cell_data[{name!r}] is missing",
        ))
        return [None] * len(mesh.cells)
    if len(raw) != len(mesh.cells):
        report.findings.append(CheckFinding(
            "INVALID_CELL_METADATA", "ERROR",
            f"cell_data[{name!r}] has {len(raw)} blocks; expected {len(mesh.cells)}",
        ))
        return [None] * len(mesh.cells)
    blocks: list[np.ndarray | None] = []
    for block_index, (block, values) in enumerate(zip(mesh.cells, raw, strict=True)):
        array = np.asarray(values)
        if array.ndim != 1 or len(array) != len(block.data) or not np.issubdtype(array.dtype, np.integer):
            report.findings.append(CheckFinding(
                "INVALID_CELL_METADATA", "ERROR",
                f"cell_data[{name!r}][{block_index}] must contain one integer per element",
                cell_block_index=block_index,
            ))
            blocks.append(None)
        else:
            blocks.append(array)
    return blocks


def _node_id(point_ids: np.ndarray | None, index: int) -> int | None:
    return None if point_ids is None else int(point_ids[index])


def _bounds(points: np.ndarray, indices: set[int], point_ids: np.ndarray | None) -> dict[str, Any] | None:
    valid = np.array(sorted(index for index in indices if np.isfinite(points[index]).all()), dtype=int)
    if not len(valid):
        return None
    selected = points[valid]
    minimum_positions = np.argmin(selected, axis=0)
    maximum_positions = np.argmax(selected, axis=0)
    minima = selected[minimum_positions, np.arange(points.shape[1])]
    maxima = selected[maximum_positions, np.arange(points.shape[1])]
    return {
        "node_count": len(valid),
        "minimum": minima.tolist(),
        "maximum": maxima.tolist(),
        "size": (maxima - minima).tolist(),
        "minimum_node_ids": [_node_id(point_ids, int(valid[index])) for index in minimum_positions],
        "maximum_node_ids": [_node_id(point_ids, int(valid[index])) for index in maximum_positions],
        "minimum_point_indices": valid[minimum_positions].tolist(),
        "maximum_point_indices": valid[maximum_positions].tolist(),
    }


def _components(cells: list[_Cell], points: np.ndarray, point_ids: np.ndarray | None) -> list[dict[str, Any]]:
    groups: dict[tuple[int, int], list[_Cell]] = defaultdict(list)
    for cell in cells:
        if cell.property_id is not None:
            groups[(cell.property_id, cell.topology.dimension)].append(cell)

    output: list[dict[str, Any]] = []
    for (property_id, dimension), members in sorted(groups.items()):
        parent = list(range(len(members)))

        def root(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        entity_owner: dict[tuple[int, ...], int] = {}
        for index, cell in enumerate(members):
            for entity in cell.topology.boundary_entities:
                key = tuple(sorted(cell.nodes[local] for local in entity))
                other = entity_owner.setdefault(key, index)
                parent[root(index)] = root(other)

        grouped: dict[int, list[_Cell]] = defaultdict(list)
        for index, cell in enumerate(members):
            grouped[root(index)].append(cell)
        for component_index, component_cells in enumerate(grouped.values(), start=1):
            indices = {node for cell in component_cells for node in cell.nodes}
            first = component_cells[0]
            output.append({
                "property_id": property_id,
                "dimension": dimension,
                "component_index": component_index,
                "element_count": len(component_cells),
                "first_element_id": first.element_id,
                "first_cell_block_index": first.block_index,
                "first_local_element_index": first.local_index,
                "bounds": _bounds(points, indices, point_ids),
            })
    return output


def check_mesh(
    mesh: Any,
    *,
    source: str | None = None,
    coordinate_unit: str = "model coordinate unit",
    coordinate_tolerance: float = 0.0,
    surface_owner_properties: dict[int, int] | None = None,
    surface_normal_direction: str = "outward",
    surface_normal_directions: dict[int, str] | None = None,
    geometry_backend: str | Any = "auto",
    geometry_relative_tolerance: float = 1e-10,
    dimension_spec: dict[str, Any] | None = None,
) -> CheckReport:
    """Inspect a mesh without changing it; geometry checks need a private backend."""

    if not hasattr(mesh, "points") or not hasattr(mesh, "cells"):
        raise MeshCheckError("mesh must provide points and cells")
    try:
        points = np.asarray(mesh.points, dtype=float)
    except (TypeError, ValueError) as error:
        raise MeshCheckError("points must contain numeric coordinates") from error
    if points.ndim != 2 or points.shape[1] not in (2, 3):
        raise MeshCheckError("points must have shape (n, 2) or (n, 3)")
    if not coordinate_unit:
        raise MeshCheckError("coordinate_unit must not be empty")
    if surface_normal_direction not in {"outward", "inward"}:
        raise MeshCheckError("surface_normal_direction must be outward or inward")
    if any(direction not in {"outward", "inward"} for direction in (surface_normal_directions or {}).values()):
        raise MeshCheckError("surface_normal_directions values must be outward or inward")

    cell_types: dict[str, int] = defaultdict(int)
    for block in mesh.cells:
        cell_types[str(block.type)] += len(block.data)
    report = CheckReport(
        source=source,
        num_points=len(points),
        num_cells=sum(cell_types.values()),
        cell_types=dict(cell_types),
        coordinate_unit=coordinate_unit,
    )
    point_ids = _ids(mesh, "id", len(points), report)
    element_ids = _cell_ids(mesh, "element_id", report)
    property_ids = _cell_ids(mesh, "property_id", report)

    for index in np.flatnonzero(~np.isfinite(points).all(axis=1)):
        report.findings.append(CheckFinding(
            "NONFINITE_COORDINATE", "ERROR", "node has a non-finite coordinate",
            node_ids=() if point_ids is None else (int(point_ids[index]),),
            local_node_index=int(index),
        ))

    used_nodes: set[int] = set()
    property_nodes: dict[int, set[int]] = defaultdict(set)
    supported_cells: list[_Cell] = []
    seen_elements: dict[int, tuple[int, int]] = {}
    seen_connectivity: dict[tuple[str, tuple[int, ...]], dict[int | None, int | None]] = {}
    unsupported_count = 0
    for block_index, block in enumerate(mesh.cells):
        topology = get_element_topology(str(block.type))
        connectivity = np.asarray(block.data)
        if topology is None and str(block.type) == "line":
            report.findings.append(CheckFinding(
                "AUXILIARY_CELL_TYPE", "INFO",
                "line elements are retained for input accounting but are not area/volume cells",
                cell_block_index=block_index,
            ))
        elif topology is None:
            unsupported_count += len(block.data)
            report.findings.append(CheckFinding(
                "UNSUPPORTED_CELL_TYPE", "WARNING",
                f"{block.type} elements are not checked for topology or dimensions by component",
                cell_block_index=block_index,
            ))
        if connectivity.ndim != 2 or not np.issubdtype(connectivity.dtype, np.integer):
            report.findings.append(CheckFinding(
                "INVALID_CONNECTIVITY", "ERROR",
                "connectivity must be a two-dimensional integer array",
                cell_block_index=block_index,
            ))
            continue
        if topology is not None and connectivity.shape[1] != topology.node_count:
            report.findings.append(CheckFinding(
                "INVALID_CONNECTIVITY", "ERROR",
                f"{block.type} needs {topology.node_count} nodes per element; got {connectivity.shape[1]}",
                cell_block_index=block_index,
            ))
            continue
        for local_index, row in enumerate(connectivity):
            element_id = None if element_ids[block_index] is None else int(element_ids[block_index][local_index])
            property_id = None if property_ids[block_index] is None else int(property_ids[block_index][local_index])
            if element_id is not None:
                previous = seen_elements.setdefault(element_id, (block_index, local_index))
                if previous != (block_index, local_index):
                    report.findings.append(CheckFinding(
                        "DUPLICATE_ELEMENT_ID", "ERROR", f"original element ID {element_id} is repeated",
                        element_id=element_id, property_id=property_id,
                        cell_block_index=block_index, local_element_index=local_index,
                    ))
            if np.any(row < 0) or np.any(row >= len(points)):
                report.findings.append(CheckFinding(
                    "INVALID_NODE_REFERENCE", "ERROR", "element references a point index outside the mesh",
                    element_id=element_id, property_id=property_id,
                    cell_block_index=block_index, local_element_index=local_index,
                ))
                continue
            nodes = tuple(int(node) for node in row)
            used_nodes.update(nodes)
            if property_id is not None:
                property_nodes[property_id].update(nodes)
            if topology is None:
                continue
            cell = _Cell(block_index, local_index, topology, nodes, element_id, property_id)
            supported_cells.append(cell)
            if len(set(nodes)) != len(nodes):
                report.findings.append(CheckFinding(
                    "REPEATED_ELEMENT_NODE", "ERROR", "element repeats a point index",
                    node_ids=tuple(int(point_ids[node]) for node in nodes) if point_ids is not None else (),
                    element_id=element_id, property_id=property_id,
                    cell_block_index=block_index, local_element_index=local_index,
                ))
            key = (topology.cell_type, tuple(sorted(nodes)))
            previous_properties = seen_connectivity.setdefault(key, {})
            if previous_properties:
                is_cross_property_surface = (
                    topology.dimension == 2 and property_id is not None
                    and None not in previous_properties and property_id not in previous_properties
                )
                report.findings.append(CheckFinding(
                    "OVERLAPPING_SURFACE_PROPERTIES" if is_cross_property_surface else "DUPLICATE_ELEMENT",
                    "INFO" if is_cross_property_surface else "ERROR",
                    "surface elements in different Properties use the same nodes; verify the overlay is intentional"
                    if is_cross_property_surface else "another element uses the same nodes and type",
                    element_id=element_id, property_id=property_id,
                    related_element_ids=tuple(
                        previous_id for previous_id in previous_properties.values()
                        if previous_id is not None
                    ),
                    cell_block_index=block_index, local_element_index=local_index,
                ))
            previous_properties.setdefault(property_id, element_id)
            if np.isfinite(points[list(nodes)]).all():
                edges: set[tuple[int, int]] = set()
                for entity in topology.boundary_entities:
                    ring = [nodes[local] for local in entity]
                    for start, end in zip(ring, ring[1:] + ring[:1]):
                        edges.add(tuple(sorted((start, end))))
                for start, end in sorted(edges):
                    if np.array_equal(points[start], points[end]):
                        report.findings.append(CheckFinding(
                            "ZERO_LENGTH_EDGE", "ERROR", "element has a zero-length edge",
                            node_ids=(int(point_ids[start]), int(point_ids[end])) if point_ids is not None else (),
                            element_id=element_id, property_id=property_id,
                            cell_block_index=block_index, local_element_index=local_index,
                            measured_value=0.0,
                        ))
                        break

    for index in range(len(points)):
        if index not in used_nodes:
            report.findings.append(CheckFinding(
                "UNUSED_NODE", "WARNING", "node is not referenced by any valid element",
                node_ids=() if point_ids is None else (int(point_ids[index]),),
                local_node_index=index,
                coordinates=tuple(float(value) for value in points[index]) if np.isfinite(points[index]).all() else None,
            ))

    report.dimensions = {
        "model": _bounds(points, used_nodes, point_ids),
        "properties": [
            {"property_id": property_id, "bounds": _bounds(points, indices, point_ids)}
            for property_id, indices in sorted(property_nodes.items())
        ],
        "components": _components(supported_cells, points, point_ids),
    }
    report.checks["input_integrity"] = CheckResult(
        "FINDINGS" if report.findings else "PASS",
        finding_count=len(report.findings),
    )
    has_finite_bounds = report.dimensions["model"] is not None
    partial_bounds = any(item.code == "NONFINITE_COORDINATE" for item in report.findings)
    report.checks["dimensions"] = CheckResult(
        "FINDINGS" if has_finite_bounds and partial_bounds else "PASS" if has_finite_bounds else "NOT_RUN",
        reason="non-finite referenced nodes were excluded" if partial_bounds else None if has_finite_bounds else "no finite referenced nodes",
    )
    report.checks["property_dimensions"] = CheckResult(
        "PASS" if property_nodes else "NOT_RUN",
        reason=None if property_nodes else "no valid Property IDs on referenced elements",
    )
    report.checks["supported_elements"] = CheckResult(
        "UNSUPPORTED" if unsupported_count else "PASS",
        reason=f"{unsupported_count} unsupported elements" if unsupported_count else None,
    )
    add_design_dimension_checks(report, dimension_spec)
    add_geometry_checks(
        mesh, report, backend=geometry_backend,
        relative_tolerance=geometry_relative_tolerance,
    )
    add_connectivity_checks(
        mesh, report, point_ids=point_ids, coordinate_tolerance=coordinate_tolerance,
    )
    add_normal_checks(
        mesh, report, point_ids=point_ids,
        surface_owner_properties=surface_owner_properties,
        surface_normal_direction=surface_normal_direction,
        surface_normal_directions=surface_normal_directions,
    )
    return report
