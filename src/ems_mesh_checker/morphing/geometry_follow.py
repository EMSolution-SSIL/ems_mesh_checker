"""Safe, reproducible 2-D rectangle edits on an existing Property mesh.

The manifest, rather than a simplified GEO/DXF curve, is the authority for
original node correspondence.  The independent ``ems_mesh_morpher`` package
propagates prescribed displacement into neighbouring Properties.  No morpher
implementation is copied into this MIT-licensed project.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from ems_mesh_checker.boundary import BoundaryExtractor
from ems_mesh_checker.checks import check_mesh
from ems_mesh_checker.io_contract import inspect_mesh_contract


class MorphingError(ValueError):
    """The edit cannot safely be mapped to the unchanged mesh topology."""


@dataclass(frozen=True)
class MorphResult:
    mesh: Any
    report: dict[str, Any]


def _seal_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    content = {key: value for key, value in manifest.items() if key not in {"manifest_sha256", "base_mesh_path"}}
    canonical = json.dumps(content, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    manifest["manifest_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return manifest


def _check_manifest_digest(manifest: dict[str, Any]) -> None:
    expected = manifest.get("manifest_sha256")
    if not isinstance(expected, str) or _seal_manifest(dict(manifest))["manifest_sha256"] != expected:
        raise MorphingError("correspondence manifest was edited or is incomplete")


def mesh_fingerprint(mesh: Any) -> str:
    """Hash coordinates, connectivity and original IDs in their input order."""

    inspect_mesh_contract(mesh)
    digest = hashlib.sha256()

    def add_array(value: Any, dtype: str) -> None:
        array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())

    add_array(mesh.points, "<f8")
    add_array(mesh.point_data["id"], "<i8")
    for index, block in enumerate(mesh.cells):
        digest.update(str(block.type).encode("ascii"))
        add_array(block.data, "<i8")
        add_array(mesh.cell_data["element_id"][index], "<i8")
        add_array(mesh.cell_data["property_id"][index], "<i8")
    return digest.hexdigest()


def _property_nodes(mesh: Any, property_id: int) -> set[int]:
    nodes: set[int] = set()
    for block, ids in zip(mesh.cells, mesh.cell_data["property_id"], strict=True):
        if np.any(np.asarray(ids) == property_id):
            nodes.update(int(value) for value in np.asarray(block.data)[np.asarray(ids) == property_id].ravel())
    return nodes


def _boundary_edges(mesh: Any, property_id: int) -> list[tuple[int, int]]:
    boundaries = BoundaryExtractor(mesh).extract()
    if boundaries.nonmanifold_faces or boundaries.unclassified_faces or boundaries.warnings:
        raise MorphingError("boundary extraction is ambiguous; inspect the mesh diagnostics first")
    faces = [face for face in boundaries.external_faces if face.owner.property_id == property_id]
    for pair, interface in boundaries.property_interfaces.items():
        if property_id in pair:
            faces.extend(interface)
    if any(len(face.oriented_nodes) != 2 for face in faces):
        raise MorphingError("the initial rectangle workflow supports only 2-D boundary edges")
    return sorted({tuple(sorted(map(int, face.oriented_nodes))) for face in faces})


def create_rectangle_manifest(
    mesh: Any,
    property_id: int,
    *,
    deformable_properties: tuple[int, ...],
    axis: str = "x",
    coordinate_unit: str = "model coordinate unit",
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Fit one axis-aligned rectangle and retain every original boundary ID."""

    inspect_mesh_contract(mesh)
    points = np.asarray(mesh.points, dtype=float)
    if any(block.type not in {"triangle", "quad"} for block in mesh.cells):
        raise MorphingError("rectangle Phase 1 accepts only linear triangle and quad elements")
    if points.shape[1] not in (2, 3) or (points.shape[1] == 3 and not np.allclose(points[:, 2], points[0, 2])):
        raise MorphingError("rectangle Phase 1 requires a planar 2-D mesh")
    if not np.isfinite(points).all() or not np.isfinite(tolerance) or tolerance <= 0:
        raise MorphingError("coordinates must be finite and tolerance positive")
    if axis not in {"x", "y"}:
        raise MorphingError("thickness axis must be x or y")
    if property_id in deformable_properties or not deformable_properties:
        raise MorphingError("specify at least one distinct surrounding deformable Property")
    target_nodes = _property_nodes(mesh, property_id)
    if not target_nodes:
        raise MorphingError(f"Property {property_id} has no elements")
    missing = sorted(pid for pid in deformable_properties if not _property_nodes(mesh, pid))
    if missing:
        raise MorphingError(f"deformable Properties do not exist: {missing}")
    edges = _boundary_edges(mesh, property_id)
    if len(edges) < 4:
        raise MorphingError("rectangle boundary needs at least four edges")
    incidence: dict[int, int] = {}
    for first, second in edges:
        incidence[first] = incidence.get(first, 0) + 1
        incidence[second] = incidence.get(second, 0) + 1
    if any(degree != 2 for degree in incidence.values()):
        raise MorphingError("boundary is open or branching")
    # A degree-two graph may still have several loops (holes or disconnected regions).
    adjacency: dict[int, set[int]] = {node: set() for node in incidence}
    for first, second in edges:
        adjacency[first].add(second)
        adjacency[second].add(first)
    visited = {next(iter(adjacency))}
    pending = list(visited)
    while pending:
        for neighbour in adjacency[pending.pop()]:
            if neighbour not in visited:
                visited.add(neighbour)
                pending.append(neighbour)
    if len(visited) != len(adjacency):
        raise MorphingError("Property has multiple boundary loops; select a single simple rectangle")
    boundary_indices = np.asarray(sorted(adjacency), dtype=int)
    boundary_xy = points[boundary_indices, :2]
    minimum = boundary_xy.min(axis=0)
    maximum = boundary_xy.max(axis=0)
    extent = maximum - minimum
    if np.any(extent <= 2 * tolerance):
        raise MorphingError("rectangle has a zero or unresolved extent")
    side_distance = np.minimum(np.abs(boundary_xy - minimum), np.abs(boundary_xy - maximum))
    if np.any(np.min(side_distance, axis=1) > tolerance):
        raise MorphingError("boundary contains nodes away from the four rectangle sides")
    side_names = ("x_min", "x_max", "y_min", "y_max")
    side_nodes: dict[str, list[int]] = {}
    original_ids = np.asarray(mesh.point_data["id"], dtype=int)
    for side, coordinate, value in (
        (side_names[0], 0, minimum[0]), (side_names[1], 0, maximum[0]),
        (side_names[2], 1, minimum[1]), (side_names[3], 1, maximum[1]),
    ):
        side_nodes[side] = sorted(int(original_ids[index]) for index in boundary_indices if abs(points[index, coordinate] - value) <= tolerance)
        if len(side_nodes[side]) < 2:
            raise MorphingError(f"rectangle side {side} has too few nodes")
    allowed = {property_id, *deformable_properties}
    node_properties: dict[int, set[int]] = {}
    for block, ids in zip(mesh.cells, mesh.cell_data["property_id"], strict=True):
        for row, pid in zip(block.data, ids, strict=True):
            for node in row:
                node_properties.setdefault(int(node), set()).add(int(pid))
    outside_shared = sorted(int(original_ids[node]) for node in target_nodes if node_properties[node] - allowed)
    if outside_shared:
        raise MorphingError(f"target nodes touch non-deformable Properties: {outside_shared[:12]}")
    outer_nodes: set[int] = set()
    boundaries = BoundaryExtractor(mesh).extract()
    for face in boundaries.external_faces:
        outer_nodes.update(map(int, face.oriented_nodes))
    outer_shared = sorted(int(original_ids[node]) for node in target_nodes & outer_nodes)
    if outer_shared:
        raise MorphingError(f"target touches the fixed external boundary: {outer_shared[:12]}")
    return _seal_manifest({
        "schema_version": 1,
        "shape": "rectangle_2d",
        "mesh_sha256": mesh_fingerprint(mesh),
        "coordinate_unit": coordinate_unit,
        "coordinate_system": "global_xy",
        "property_id": int(property_id),
        "deformable_properties": sorted(set(map(int, deformable_properties))),
        "thickness_axis": axis,
        "minimum": minimum.tolist(),
        "maximum": maximum.tolist(),
        "tolerance": float(tolerance),
        "boundary_node_ids": sorted(int(original_ids[index]) for index in boundary_indices),
        "side_node_ids": side_nodes,
        "boundary_edges": [[int(original_ids[a]), int(original_ids[b])] for a, b in edges],
        "target_node_ids": sorted(int(original_ids[index]) for index in target_nodes),
        "diagnostics": [],
    })


def _new_geometry_errors(before: Any, after: Any) -> list[dict[str, Any]]:
    severe = {"DEGENERATE_ELEMENT", "NEAR_DEGENERATE_ELEMENT", "INVERTED_VOLUME_ELEMENT", "MIXED_JACOBIAN_SIGNS", "FOLDED_SURFACE_ELEMENT"}
    def identity(item: Any) -> tuple[Any, ...]:
        return (item.code, item.cell_block_index, item.local_element_index)
    old = {identity(item) for item in before.findings if item.code in severe}
    return [
        {"code": item.code, "element_id": item.element_id, "property_id": item.property_id}
        for item in after.findings if item.code in severe and identity(item) not in old
    ]


def _surface_orientation_flips(before: Any, after: Any) -> list[int]:
    """Detect relative 2-D element flips even when a backend reports unsigned area."""

    flips: list[int] = []
    for block_index, (old_block, new_block) in enumerate(zip(before.cells, after.cells, strict=True)):
        if old_block.type not in {"triangle", "quad"}:
            continue
        nodes = np.asarray(old_block.data, dtype=int)
        old = np.asarray(before.points, dtype=float)[nodes, :2]
        new = np.asarray(after.points, dtype=float)[nodes, :2]
        def area(coordinates: np.ndarray) -> np.ndarray:
            x, y = coordinates[:, :, 0], coordinates[:, :, 1]
            return 0.5 * np.sum(x * np.roll(y, -1, axis=1) - y * np.roll(x, -1, axis=1), axis=1)
        old_area, new_area = area(old), area(new)
        old_scale = np.max(np.ptp(old, axis=1), axis=1)
        nearly_zero = 64 * np.finfo(float).eps * np.square(old_scale)
        ids = np.asarray(before.cell_data["element_id"][block_index], dtype=int)
        flips.extend(int(ids[index]) for index in np.flatnonzero(
            (np.abs(old_area) > nearly_zero) & (old_area * new_area <= 0)
        ))
    return flips


def apply_rectangle_case(
    mesh: Any,
    manifest: dict[str, Any],
    *,
    translation: tuple[float, float] = (0.0, 0.0),
    thickness: float | None = None,
    anchor: str = "center",
    geometry_backend: str | Any = "required",
    interpolation: str = "weighted_laplace",
) -> MorphResult:
    """Morph from the original mesh; never chain a second case from an output."""

    if manifest.get("schema_version") != 1 or manifest.get("shape") != "rectangle_2d":
        raise MorphingError("unsupported correspondence manifest")
    _check_manifest_digest(manifest)
    if mesh_fingerprint(mesh) != manifest.get("mesh_sha256"):
        raise MorphingError("base mesh does not match the correspondence manifest")
    expected = create_rectangle_manifest(
        mesh, int(manifest["property_id"]),
        deformable_properties=tuple(map(int, manifest["deformable_properties"])),
        axis=str(manifest["thickness_axis"]),
        coordinate_unit=str(manifest["coordinate_unit"]), tolerance=float(manifest["tolerance"]),
    )
    for field in ("minimum", "maximum", "boundary_node_ids", "boundary_edges", "side_node_ids", "target_node_ids"):
        if manifest.get(field) != expected[field]:
            raise MorphingError(f"manifest correspondence field {field} does not match the base mesh")
    if anchor not in {"min", "center", "max"}:
        raise MorphingError("anchor must be min, center, or max")
    translation_array = np.asarray(translation, dtype=float)
    if translation_array.shape != (2,) or not np.isfinite(translation_array).all():
        raise MorphingError("translation must contain two finite values")
    if interpolation not in {"weighted_laplace", "rbf"}:
        raise MorphingError("interpolation must be weighted_laplace or rbf")
    minimum = np.asarray(manifest["minimum"], dtype=float)
    maximum = np.asarray(manifest["maximum"], dtype=float)
    coordinate = 0 if manifest["thickness_axis"] == "x" else 1
    old_thickness = float(maximum[coordinate] - minimum[coordinate])
    new_thickness = old_thickness if thickness is None else float(thickness)
    if not np.isfinite(new_thickness) or new_thickness <= 0:
        raise MorphingError("target thickness must be positive and finite")
    target_min = minimum + translation_array
    target_max = maximum + translation_array
    change = new_thickness - old_thickness
    if anchor == "min":
        target_max[coordinate] += change
    elif anchor == "max":
        target_min[coordinate] -= change
    else:
        target_min[coordinate] -= change / 2
        target_max[coordinate] += change / 2
    points = np.asarray(mesh.points, dtype=float)
    node_ids = np.asarray(mesh.point_data["id"], dtype=int)
    index_of = {int(value): index for index, value in enumerate(node_ids)}
    moving = np.asarray([index_of[int(value)] for value in manifest["target_node_ids"]], dtype=int)
    target = points.copy()
    target[moving, :2] += translation_array
    fraction = (points[moving, coordinate] - minimum[coordinate]) / old_thickness
    target[moving, coordinate] = target_min[coordinate] + new_thickness * fraction
    return _apply_targets(
        mesh, manifest, moving, target, geometry_backend=geometry_backend,
        interpolation=interpolation,
        parameters={"translation": translation_array.tolist(), "thickness": new_thickness, "anchor": anchor},
    )


def _apply_targets(
    mesh: Any,
    manifest: dict[str, Any],
    moving: np.ndarray,
    target: np.ndarray,
    *,
    geometry_backend: str | Any,
    interpolation: str,
    parameters: dict[str, Any],
    relax_symmetry_exterior: bool = False,
) -> MorphResult:
    if interpolation == "analytic_radial" and manifest.get("shape") != "octant_shell_3d":
        raise MorphingError("analytic_radial propagation is available only for an octant box shell")
    points = np.asarray(mesh.points, dtype=float)
    prescribed = target - points
    allowed_properties = {int(manifest["property_id"]), *map(int, manifest["deformable_properties"])}
    allowed = set().union(*(_property_nodes(mesh, pid) for pid in allowed_properties))
    boundaries = BoundaryExtractor(mesh).extract()
    external = {int(node) for face in boundaries.external_faces for node in face.oriented_nodes}
    symmetry_planes = [(int(axis), float(value)) for axis, value in manifest.get("symmetry_planes", [])]
    non_symmetry_external: set[int] = set()
    if relax_symmetry_exterior:
        for face in boundaries.external_faces:
            face_nodes = list(map(int, face.oriented_nodes))
            if not any(np.all(np.abs(points[face_nodes, axis] - value) <= float(manifest["tolerance"]))
                       for axis, value in symmetry_planes):
                non_symmetry_external.update(face_nodes)
    for node in set(map(int, moving)) & external:
        matching = [(axis, value) for axis, value in symmetry_planes
                    if abs(points[node, axis] - value) <= float(manifest["tolerance"])]
        if not matching or any(abs(prescribed[node, axis]) > float(manifest["tolerance"])
                               for axis, _ in matching):
            raise MorphingError("moving nodes leave a fixed external or symmetry boundary")
    relaxable = (external & allowed) - set(map(int, moving)) - non_symmetry_external if relax_symmetry_exterior else set()
    fixed = (set(range(len(points))) - allowed) | (external - set(map(int, moving)) - relaxable)
    deformable = allowed - set(moving) - fixed
    try:
        from ems_mesh_morpher.config import MorphConfig, MorphingConfig, QualityConfig, RegionConfig, SelectionConfig
        from ems_mesh_morpher.mesh_motion import apply_mesh_motion
    except ImportError as error:
        raise MorphingError("install ems_mesh_morpher to propagate the prescribed geometry edit") from error
    regions = RegionConfig(
        moving=SelectionConfig(point_indices=tuple(sorted(map(int, moving)))),
        deformable=SelectionConfig(point_indices=tuple(sorted(deformable))),
        fixed=SelectionConfig(point_indices=tuple(sorted(fixed))),
        conflict_policy="error",
    )
    config = MorphConfig(
        mode="geometry_follow",
        regions=regions,
        morphing=MorphingConfig(type="none" if interpolation == "analytic_radial" else interpolation),
        quality=QualityConfig(enabled=False),
    )
    before = check_mesh(mesh, geometry_backend=geometry_backend)
    if before.checks["area_volume_degeneracy"].status not in {"PASS", "FINDINGS"} or before.geometry_quality.get("unsupported_elements"):
        raise MorphingError("the baseline geometry quality check did not cover all supported area/volume elements")
    if any(item.severity == "ERROR" for item in before.findings):
        raise MorphingError("the baseline mesh already has error-severity check findings")
    try:
        motion = apply_mesh_motion(mesh, config, prescribed_displacement=prescribed)
    except ValueError as error:
        raise MorphingError(f"mesh morpher rejected the case: {error}") from error
    if relaxable:
        for axis, value in symmetry_planes:
            nodes = [node for node in relaxable if abs(points[node, axis] - value) <= float(manifest["tolerance"])]
            motion.mesh.points[nodes, axis] = value
    after = check_mesh(motion.mesh, geometry_backend=geometry_backend)
    if after.checks["area_volume_degeneracy"].status not in {"PASS", "FINDINGS"} or after.geometry_quality.get("unsupported_elements"):
        raise MorphingError("the morphed geometry quality check did not complete")
    new_errors = _new_geometry_errors(before, after)
    flipped_ids = _surface_orientation_flips(mesh, motion.mesh)
    if flipped_ids:
        raise MorphingError(f"surface element orientation flipped: {flipped_ids[:12]}")
    if new_errors:
        raise MorphingError(f"new degenerate or inverted elements: {new_errors[:8]}")
    other_errors = [item.code for item in after.findings if item.severity == "ERROR"]
    if other_errors:
        raise MorphingError(f"morphed mesh has error-severity check findings: {other_errors[:8]}")
    target_error = float(np.max(np.linalg.norm(np.asarray(motion.mesh.points)[moving] - target[moving], axis=1)))
    if target_error > float(manifest["tolerance"]):
        raise MorphingError(f"target-boundary mismatch {target_error:g} exceeds tolerance")
    report = {
        "schema_version": 1,
        "status": "PASS",
        "base_mesh_sha256": manifest["mesh_sha256"],
        "property_id": manifest["property_id"],
        "parameters": parameters,
        "interpolation": interpolation,
        "moving_nodes": len(moving),
        "deformable_nodes": len(deformable),
        "fixed_nodes": len(fixed),
        "max_displacement": float(np.linalg.norm(np.asarray(motion.mesh.points) - points, axis=1).max()),
        "max_target_error": target_error,
        "new_geometry_defects": new_errors,
        "quality_before": before.geometry_quality,
        "quality_after": after.geometry_quality,
        "checks_before": {name: item.status for name, item in before.checks.items()},
        "checks_after": {name: item.status for name, item in after.checks.items()},
    }
    report.update(parameters)
    return MorphResult(motion.mesh, report)
