"""Strict L-prism and coaxial-cylinder parameter correspondences.

Only recognized shapes with explicit feature coordinates are accepted. The
unmodified mesh remains the sole source for every case in a survey.
"""

from __future__ import annotations

from collections import defaultdict
import json
from typing import Any

import numpy as np

from ems_mesh_checker.boundary import BoundaryExtractor
from ems_mesh_checker.io_contract import inspect_mesh_contract
from ems_mesh_checker.topology.element_topology import get_element_topology

from .geometry_follow import (
    MorphResult, MorphingError, _apply_targets, _check_manifest_digest,
    _property_nodes, _seal_manifest, mesh_fingerprint,
)


def _property_component_count(mesh: Any, property_id: int) -> int:
    members: list[tuple[tuple[int, ...], ...]] = []
    dimensions: set[int] = set()
    for block, ids in zip(mesh.cells, mesh.cell_data["property_id"], strict=True):
        topology = get_element_topology(str(block.type))
        for row, pid in zip(block.data, ids, strict=True):
            if int(pid) != property_id:
                continue
            if topology is None:
                raise MorphingError(f"unsupported element type in Property {property_id}: {block.type}")
            dimensions.add(topology.dimension)
            members.append(tuple(tuple(sorted(int(row[local]) for local in face)) for face in topology.boundary_entities))
    if len(dimensions) != 1 or not members:
        raise MorphingError("target Property must contain one element dimension")
    parent = list(range(len(members)))
    owners: dict[tuple[int, ...], int] = {}
    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index
    for index, faces in enumerate(members):
        for face in faces:
            other = owners.setdefault(face, index)
            parent[root(index)] = root(other)
    return len({root(index) for index in range(len(members))})


def _property_boundary(mesh: Any, property_id: int):
    result = BoundaryExtractor(mesh).extract()
    if result.warnings or result.nonmanifold_faces or result.unclassified_faces:
        raise MorphingError("Property boundary is ambiguous or nonmanifold")
    faces = [face for face in result.external_faces if face.owner.property_id == property_id]
    for pair, patch in result.property_interfaces.items():
        if property_id in pair:
            faces.extend(patch)
    if not faces:
        raise MorphingError("target Property has no extracted boundary")
    return faces, result


def _base_manifest(
    mesh: Any, property_id: int, deformable_properties: tuple[int, ...], tolerance: float,
    *, symmetry_planes: tuple[tuple[int, float], ...] = (),
) -> dict[str, Any]:
    inspect_mesh_contract(mesh)
    if not np.isfinite(mesh.points).all() or not np.isfinite(tolerance) or tolerance <= 0:
        raise MorphingError("coordinates must be finite and tolerance positive")
    if not deformable_properties or property_id in deformable_properties:
        raise MorphingError("specify distinct surrounding deformable Properties")
    if _property_component_count(mesh, property_id) != 1:
        raise MorphingError("target Property has multiple disconnected components")
    target = _property_nodes(mesh, property_id)
    missing = [pid for pid in deformable_properties if not _property_nodes(mesh, pid)]
    if missing:
        raise MorphingError(f"deformable Properties do not exist: {missing}")
    faces, boundaries = _property_boundary(mesh, property_id)
    external = {int(node) for face in boundaries.external_faces for node in face.oriented_nodes}
    for node in target & external:
        if not any(abs(mesh.points[node, axis] - value) <= tolerance
                   for axis, value in symmetry_planes):
            raise MorphingError("target Property touches the fixed external boundary")
    for face in boundaries.external_faces:
        if face.owner.property_id != property_id:
            continue
        face_points = np.asarray(mesh.points)[list(face.oriented_nodes)]
        if not any(np.all(np.abs(face_points[:, axis] - value) <= tolerance)
                   for axis, value in symmetry_planes):
            raise MorphingError("target Property touches an external face outside the declared symmetry planes")
    allowed = {property_id, *deformable_properties}
    node_properties: dict[int, set[int]] = defaultdict(set)
    for block, ids in zip(mesh.cells, mesh.cell_data["property_id"], strict=True):
        for row, pid in zip(block.data, ids, strict=True):
            for node in row:
                node_properties[int(node)].add(int(pid))
    if any(node_properties[node] - allowed for node in target):
        raise MorphingError("target nodes touch a non-deformable Property")
    ids = np.asarray(mesh.point_data["id"], dtype=int)
    return {
        "schema_version": 1, "mesh_sha256": mesh_fingerprint(mesh),
        "property_id": int(property_id),
        "deformable_properties": sorted(set(map(int, deformable_properties))),
        "symmetry_planes": [[int(axis), float(value)] for axis, value in symmetry_planes],
        "tolerance": float(tolerance),
        "target_node_ids": sorted(int(ids[index]) for index in target),
        "boundary_node_ids": sorted({int(ids[node]) for face in faces for node in face.oriented_nodes}),
        "boundary_faces": [
            {"node_ids": [int(ids[node]) for node in face.oriented_nodes],
             "element_id": face.owner.element_id,
             "property_pair": face.property_pair}
            for face in faces
        ],
        "diagnostics": [],
    }


def _verify_manifest(mesh: Any, manifest: dict[str, Any], remake) -> None:
    if manifest.get("schema_version") != 1 or mesh_fingerprint(mesh) != manifest.get("mesh_sha256"):
        raise MorphingError("base mesh does not match the correspondence manifest")
    _check_manifest_digest(manifest)
    expected = remake()
    for key in ("shape", "target_node_ids", "boundary_node_ids", "boundary_faces", "minimum", "maximum"):
        if json.dumps(manifest.get(key), sort_keys=True) != json.dumps(expected.get(key), sort_keys=True):
            raise MorphingError(f"manifest correspondence field {key} does not match the base mesh")


def create_l_prism_manifest(
    mesh: Any, property_id: int, *, deformable_properties: tuple[int, ...],
    notch: tuple[float, float], tolerance: float = 1e-8,
    coordinate_unit: str = "model coordinate unit",
) -> dict[str, Any]:
    """Recognize a lower-left L cross-section in 2-D or its straight 3-D extrusion."""

    base = _base_manifest(mesh, property_id, deformable_properties, tolerance)
    points = np.asarray(mesh.points, dtype=float)
    target_indices = np.flatnonzero(np.isin(mesh.point_data["id"], base["target_node_ids"]))
    minimum = points[target_indices, :2].min(axis=0)
    maximum = points[target_indices, :2].max(axis=0)
    corner = np.asarray(notch, dtype=float)
    if corner.shape != (2,) or not np.isfinite(corner).all() or np.any(corner <= minimum + tolerance) or np.any(corner >= maximum - tolerance):
        raise MorphingError("L notch must be strictly inside the outer bounds")
    xy = points[target_indices, :2]
    if np.any((xy[:, 0] > corner[0] + tolerance) & (xy[:, 1] > corner[1] + tolerance)):
        raise MorphingError("target Property has nodes in the excluded L-notch quadrant")
    is_3d = points.shape[1] == 3 and np.ptp(points[target_indices, 2]) > tolerance
    # End caps of an extrusion may have interior XY nodes. Only side faces
    # constrain the recognized cross-section outline.
    side_indices: set[int] = set()
    id_to_index = {int(value): index for index, value in enumerate(mesh.point_data["id"])}
    for face in base["boundary_faces"]:
        indices = [id_to_index[int(identifier)] for identifier in face["node_ids"]]
        if is_3d and np.ptp(points[indices, 2]) <= tolerance:
            continue
        side_indices.update(indices)
    outline = points[sorted(side_indices), :2]
    x, y = outline[:, 0], outline[:, 1]
    on_outline = (
        np.isclose(x, minimum[0], atol=tolerance, rtol=0)
        | np.isclose(y, minimum[1], atol=tolerance, rtol=0)
        | (np.isclose(x, maximum[0], atol=tolerance, rtol=0) & (y <= corner[1] + tolerance))
        | (np.isclose(y, corner[1], atol=tolerance, rtol=0) & (x >= corner[0] - tolerance))
        | (np.isclose(x, corner[0], atol=tolerance, rtol=0) & (y >= corner[1] - tolerance))
        | (np.isclose(y, maximum[1], atol=tolerance, rtol=0) & (x <= corner[0] + tolerance))
    )
    if not len(outline) or not np.all(on_outline):
        raise MorphingError("boundary does not fit the six named L outline segments")
    for vertex in ((minimum[0], minimum[1]), (maximum[0], minimum[1]),
                   (maximum[0], corner[1]), (corner[0], corner[1]),
                   (corner[0], maximum[1]), (minimum[0], maximum[1])):
        if not np.any(np.linalg.norm(outline - vertex, axis=1) <= tolerance):
            raise MorphingError(f"L outline vertex {vertex} has no corresponding node")
    base.update({"shape": "l_prism_3d" if is_3d else "l_shape_2d",
                 "coordinate_system": "global_xy", "coordinate_unit": coordinate_unit,
                 "minimum": minimum.tolist(), "maximum": maximum.tolist(),
                 "notch": corner.tolist()})
    return _seal_manifest(base)


def apply_l_prism_case(
    mesh: Any, manifest: dict[str, Any], *,
    translation: tuple[float, float] = (0.0, 0.0),
    length_x: float | None = None, length_y: float | None = None,
    arm_width_x: float | None = None, arm_width_y: float | None = None,
    geometry_backend: str | Any = "required", interpolation: str = "weighted_laplace",
) -> MorphResult:
    if manifest.get("shape") not in {"l_shape_2d", "l_prism_3d"}:
        raise MorphingError("manifest is not an L shape")
    _verify_manifest(mesh, manifest, lambda: create_l_prism_manifest(
        mesh, int(manifest["property_id"]),
        deformable_properties=tuple(manifest["deformable_properties"]),
        notch=tuple(manifest["notch"]), tolerance=float(manifest["tolerance"]),
        coordinate_unit=str(manifest["coordinate_unit"]),
    ))
    minimum, maximum, notch = (np.asarray(manifest[name], dtype=float) for name in ("minimum", "maximum", "notch"))
    old_length = maximum - minimum
    old_width = notch - minimum
    new_length = np.asarray((old_length[0] if length_x is None else length_x,
                             old_length[1] if length_y is None else length_y), dtype=float)
    new_width = np.asarray((old_width[0] if arm_width_x is None else arm_width_x,
                            old_width[1] if arm_width_y is None else arm_width_y), dtype=float)
    shift = np.asarray(translation, dtype=float)
    if shift.shape != (2,) or not np.isfinite(shift).all() or not np.isfinite(new_length).all() or not np.isfinite(new_width).all():
        raise MorphingError("L edit parameters must be finite two-dimensional values")
    if np.any(new_width <= 0) or np.any(new_width >= new_length):
        raise MorphingError("L arms must have positive widths smaller than their lengths")
    points = np.asarray(mesh.points, dtype=float)
    lookup = {int(value): index for index, value in enumerate(mesh.point_data["id"])}
    moving = np.asarray([lookup[int(value)] for value in manifest["target_node_ids"]], dtype=int)
    target = points.copy()
    for coordinate in range(2):
        local = points[moving, coordinate] - minimum[coordinate]
        target[moving, coordinate] = minimum[coordinate] + shift[coordinate] + np.where(
            local <= old_width[coordinate],
            local * new_width[coordinate] / old_width[coordinate],
            new_width[coordinate] + (local - old_width[coordinate])
            * (new_length[coordinate] - new_width[coordinate])
            / (old_length[coordinate] - old_width[coordinate]),
        )
    return _apply_targets(
        mesh, manifest, moving, target, geometry_backend=geometry_backend,
        interpolation=interpolation,
        parameters={"translation": shift.tolist(), "length": new_length.tolist(), "arm_width": new_width.tolist()},
    )


def _axis_frame(point: tuple[float, float, float], direction: tuple[float, float, float]):
    origin = np.asarray(point, dtype=float)
    axis = np.asarray(direction, dtype=float)
    if origin.shape != (3,) or axis.shape != (3,) or not np.isfinite(origin).all() or not np.isfinite(axis).all():
        raise MorphingError("axis point and direction must contain three finite coordinates")
    length = float(np.linalg.norm(axis))
    if length == 0:
        raise MorphingError("cylinder axis direction must be nonzero")
    return origin, axis / length


def create_cylinder_manifest(
    mesh: Any, property_id: int, *, deformable_properties: tuple[int, ...],
    axis_point: tuple[float, float, float], axis_direction: tuple[float, float, float],
    tolerance: float = 1e-8, coordinate_unit: str = "model coordinate unit",
) -> dict[str, Any]:
    """Recognize a complete coaxial annular volume with unchanged end planes."""

    base = _base_manifest(mesh, property_id, deformable_properties, tolerance)
    points = np.asarray(mesh.points, dtype=float)
    if points.shape[1] != 3 or not any(get_element_topology(block.type).dimension == 3 for block in mesh.cells if get_element_topology(block.type)):
        raise MorphingError("cylinder workflow requires a 3-D volume mesh")
    origin, axis = _axis_frame(axis_point, axis_direction)
    target_indices = np.flatnonzero(np.isin(mesh.point_data["id"], base["target_node_ids"]))
    relative = points[target_indices] - origin
    axial = relative @ axis
    radial = np.linalg.norm(relative - axial[:, None] * axis, axis=1)
    inner, outer = float(radial.min()), float(radial.max())
    if inner <= tolerance or outer - inner <= 2 * tolerance:
        raise MorphingError("cylinder needs distinct positive inner and outer radii")
    lookup = {int(value): index for index, value in enumerate(mesh.point_data["id"])}
    side_indices: set[int] = set()
    ring_edges: dict[str, set[tuple[int, int]]] = {"inner": set(), "outer": set()}
    for face in base["boundary_faces"]:
        indices = [lookup[int(value)] for value in face["node_ids"]]
        face_axial = (points[indices] - origin) @ axis
        if np.ptp(face_axial) > tolerance:
            side_indices.update(indices)
            face_relative = points[indices] - origin
            face_radial = np.linalg.norm(
                face_relative - (face_relative @ axis)[:, None] * axis, axis=1,
            )
            for label, radius_value in (("inner", inner), ("outer", outer)):
                if not np.all(np.abs(face_radial - radius_value) <= tolerance):
                    continue
                for first, second in zip(indices, indices[1:] + indices[:1]):
                    first_axial = float((points[first] - origin) @ axis)
                    second_axial = float((points[second] - origin) @ axis)
                    if abs(first_axial - axial.min()) <= tolerance and abs(second_axial - axial.min()) <= tolerance:
                        ring_edges[label].add(tuple(sorted((first, second))))
    if not side_indices:
        raise MorphingError("cylinder has no recognizable axial side surface")
    side = points[sorted(side_indices)] - origin
    side_axial = side @ axis
    side_radial = np.linalg.norm(side - side_axial[:, None] * axis, axis=1)
    if np.any(np.minimum(np.abs(side_radial - inner), np.abs(side_radial - outer)) > tolerance):
        raise MorphingError("cylinder side nodes are not on two coaxial radii")
    if np.count_nonzero(np.abs(side_radial - inner) <= tolerance) < 3 or np.count_nonzero(np.abs(side_radial - outer) <= tolerance) < 3:
        raise MorphingError("cylinder inner and outer surfaces need at least three nodes each")
    for label, edges in ring_edges.items():
        adjacency: dict[int, set[int]] = defaultdict(set)
        for first, second in edges:
            adjacency[first].add(second)
            adjacency[second].add(first)
        if len(adjacency) < 3 or any(len(neighbours) != 2 for neighbours in adjacency.values()):
            raise MorphingError(f"cylinder {label} ring is open; partial sectors are unsupported")
        visited = {next(iter(adjacency))}
        pending = list(visited)
        while pending:
            for neighbour in adjacency[pending.pop()]:
                if neighbour not in visited:
                    visited.add(neighbour)
                    pending.append(neighbour)
        if len(visited) != len(adjacency):
            raise MorphingError(f"cylinder {label} ring has multiple components")
    base.update({"shape": "coaxial_cylinder_3d", "coordinate_unit": coordinate_unit,
                 "coordinate_system": "axis_point_direction",
                 "axis_point": origin.tolist(), "axis_direction": axis.tolist(),
                 "minimum": [inner, float(axial.min())],
                 "maximum": [outer, float(axial.max())],
                 "inner_radius": inner, "outer_radius": outer})
    return _seal_manifest(base)


def apply_cylinder_case(
    mesh: Any, manifest: dict[str, Any], *, inner_radius: float | None = None,
    outer_radius: float | None = None, geometry_backend: str | Any = "required",
    interpolation: str = "weighted_laplace",
) -> MorphResult:
    if manifest.get("shape") != "coaxial_cylinder_3d":
        raise MorphingError("manifest is not a coaxial cylinder")
    _verify_manifest(mesh, manifest, lambda: create_cylinder_manifest(
        mesh, int(manifest["property_id"]),
        deformable_properties=tuple(manifest["deformable_properties"]),
        axis_point=tuple(manifest["axis_point"]), axis_direction=tuple(manifest["axis_direction"]),
        tolerance=float(manifest["tolerance"]), coordinate_unit=str(manifest["coordinate_unit"]),
    ))
    old_inner, old_outer = float(manifest["inner_radius"]), float(manifest["outer_radius"])
    new_inner = old_inner if inner_radius is None else float(inner_radius)
    new_outer = old_outer if outer_radius is None else float(outer_radius)
    if not np.isfinite((new_inner, new_outer)).all() or new_inner <= 0 or new_outer <= new_inner:
        raise MorphingError("target radii must satisfy 0 < inner < outer")
    points = np.asarray(mesh.points, dtype=float)
    lookup = {int(value): index for index, value in enumerate(mesh.point_data["id"])}
    moving = np.asarray([lookup[int(value)] for value in manifest["target_node_ids"]], dtype=int)
    origin, axis = _axis_frame(tuple(manifest["axis_point"]), tuple(manifest["axis_direction"]))
    relative = points[moving] - origin
    axial = relative @ axis
    radial_vector = relative - axial[:, None] * axis
    radius = np.linalg.norm(radial_vector, axis=1)
    if np.any(radius <= 0):
        raise MorphingError("target Property contains a node on the cylinder axis")
    fraction = (radius - old_inner) / (old_outer - old_inner)
    target = points.copy()
    target[moving] = origin + axial[:, None] * axis + (
        (new_inner + fraction * (new_outer - new_inner)) / radius
    )[:, None] * radial_vector
    return _apply_targets(
        mesh, manifest, moving, target, geometry_backend=geometry_backend,
        interpolation=interpolation,
        parameters={"inner_radius": new_inner, "outer_radius": new_outer},
    )
