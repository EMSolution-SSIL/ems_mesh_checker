"""Small geometry-following edits for positive-octant 1/8 symmetry models.

The manifest records original node correspondence.  These mappings are only
accepted for a connected target or a complete octant box shell; they are
not a general CAD reconstruction from a mesh.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .geometry_follow import MorphResult, MorphingError, _apply_targets, _seal_manifest
from .parametric import _base_manifest, _verify_manifest


_OCTANT_PLANES = ((0, 0.0), (1, 0.0), (2, 0.0))


def _octant_base(
    mesh: Any, property_id: int, deformable_properties: tuple[int, ...],
    tolerance: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    points = np.asarray(mesh.points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise MorphingError("octant workflows require a 3-D volume mesh")
    base = _base_manifest(
        mesh, property_id, deformable_properties, tolerance,
        symmetry_planes=_OCTANT_PLANES,
    )
    id_to_index = {int(identifier): index for index, identifier in enumerate(mesh.point_data["id"])}
    indices = np.asarray([id_to_index[int(identifier)] for identifier in base["target_node_ids"]], dtype=int)
    target = points[indices]
    minimum, maximum = target.min(axis=0), target.max(axis=0)
    if np.any(np.abs(minimum) > tolerance) or np.any(maximum <= tolerance):
        raise MorphingError("target must occupy the positive octant and touch all three symmetry planes")
    base.update({
        "coordinate_system": "global_xyz_positive_octant",
        "minimum": minimum.tolist(), "maximum": maximum.tolist(),
    })
    return base, points, indices


def create_octant_scale_manifest(
    mesh: Any, property_id: int, *, deformable_properties: tuple[int, ...],
    tolerance: float = 1e-8, coordinate_unit: str = "model coordinate unit",
) -> dict[str, Any]:
    """Recognize one connected octant Property for scaling or axis extent edits."""

    base, _, _ = _octant_base(mesh, property_id, deformable_properties, tolerance)
    base.update({"shape": "octant_scale_3d", "coordinate_unit": coordinate_unit})
    return _seal_manifest(base)


def apply_octant_scale_case(
    mesh: Any, manifest: dict[str, Any], *, scale: float | None = None,
    displacement: tuple[float, float, float] | None = None,
    geometry_backend: str | Any = "required", interpolation: str = "weighted_laplace",
) -> MorphResult:
    if manifest.get("shape") != "octant_scale_3d":
        raise MorphingError("manifest is not an octant scale model")
    if (scale is None) == (displacement is None):
        raise MorphingError("specify exactly one of scale or displacement")
    _verify_manifest(mesh, manifest, lambda: create_octant_scale_manifest(
        mesh, int(manifest["property_id"]),
        deformable_properties=tuple(manifest["deformable_properties"]),
        tolerance=float(manifest["tolerance"]), coordinate_unit=str(manifest["coordinate_unit"]),
    ))
    points = np.asarray(mesh.points, dtype=float)
    id_to_index = {int(identifier): index for index, identifier in enumerate(mesh.point_data["id"])}
    moving = np.asarray([id_to_index[int(identifier)] for identifier in manifest["target_node_ids"]], dtype=int)
    target = points.copy()
    if scale is not None:
        try:
            scale = float(scale)
        except (TypeError, ValueError, OverflowError) as error:
            raise MorphingError("scale must be a finite positive number") from error
        if not np.isfinite(scale) or scale <= 0 or scale == 1:
            raise MorphingError("scale must be finite, positive, and different from 1")
        factors = np.full(3, scale)
        parameters = {"scale": scale}
    else:
        try:
            offsets = np.asarray(displacement, dtype=float)
        except (TypeError, ValueError, OverflowError) as error:
            raise MorphingError("displacement must contain three finite numbers") from error
        if offsets.shape != (3,) or not np.isfinite(offsets).all() or not np.any(offsets):
            raise MorphingError("displacement must contain three finite numbers and move at least one axis")
        old_maximum = np.asarray(manifest["maximum"], dtype=float)
        with np.errstate(over="ignore"):
            new_maximum = old_maximum + offsets
            factors = new_maximum / old_maximum
        if not np.isfinite(factors).all() or np.any(new_maximum <= 0):
            raise MorphingError("displacement must leave every target extent positive")
        parameters = {"displacement": offsets.tolist(), "new_maximum": new_maximum.tolist()}
    target[moving] = points[moving] * factors
    return _apply_targets(
        mesh, manifest, moving, target, geometry_backend=geometry_backend,
        interpolation=interpolation, parameters=parameters,
        relax_symmetry_exterior=True,
    )


def create_octant_shell_manifest(
    mesh: Any, property_id: int, *, inner_property: int, outer_property: int,
    deformable_properties: tuple[int, ...], tolerance: float = 1e-8,
    coordinate_unit: str = "model coordinate unit",
) -> dict[str, Any]:
    """Recognize a complete box shell between two max-coordinate radii."""

    if len({property_id, inner_property, outer_property}) != 3:
        raise MorphingError("shell, inner, and outer Properties must differ")
    if {inner_property, outer_property} - set(deformable_properties):
        raise MorphingError("both neighboring Properties must be deformable")
    all_properties = {int(pid) for values in mesh.cell_data["property_id"] for pid in values}
    if all_properties != {property_id, inner_property, outer_property}:
        raise MorphingError("analytic shell workflow requires exactly the shell and its two neighboring Properties")
    base, points, indices = _octant_base(mesh, property_id, deformable_properties, tolerance)
    if np.any(points < -tolerance):
        raise MorphingError("analytic shell workflow requires nonnegative octant coordinates")
    radius = np.max(points[indices], axis=1)
    inner, outer = float(radius.min()), float(radius.max())
    if inner <= tolerance or outer - inner <= 2 * tolerance:
        raise MorphingError("shell needs distinct positive inner and outer radii")
    id_to_index = {int(identifier): index for index, identifier in enumerate(mesh.point_data["id"])}
    side_counts = {inner_property: 0, outer_property: 0}
    for face in base["boundary_faces"]:
        pair = face["property_pair"]
        if pair is None:
            continue
        other = next((int(pid) for pid in pair if int(pid) != property_id), None)
        if other not in side_counts:
            raise MorphingError("shell touches an unexpected Property")
        face_points = points[[id_to_index[int(identifier)] for identifier in face["node_ids"]]]
        expected = inner if other == inner_property else outer
        if np.any(np.abs(np.max(face_points, axis=1) - expected) > tolerance):
            raise MorphingError("interface does not lie on the expected inner or outer box")
        side_counts[other] += 1
    if not all(side_counts.values()):
        raise MorphingError("shell must have both complete inner and outer interfaces")
    far_radius = float(np.max(points))
    if far_radius <= outer + tolerance:
        raise MorphingError("outer Property needs a distinct fixed far boundary")
    base.update({
        "shape": "octant_shell_3d", "coordinate_unit": coordinate_unit,
        "inner_property": int(inner_property), "outer_property": int(outer_property),
        "inner_radius": inner, "outer_radius": outer,
        "far_radius": far_radius,
    })
    return _seal_manifest(base)


def apply_octant_shell_case(
    mesh: Any, manifest: dict[str, Any], *, inner_radius: float | None = None,
    outer_radius: float | None = None, inner_displacement: float | None = None,
    outer_displacement: float | None = None, geometry_backend: str | Any = "required",
    interpolation: str = "analytic_radial",
) -> MorphResult:
    if manifest.get("shape") != "octant_shell_3d":
        raise MorphingError("manifest is not an octant box shell")
    _verify_manifest(mesh, manifest, lambda: create_octant_shell_manifest(
        mesh, int(manifest["property_id"]),
        inner_property=int(manifest["inner_property"]), outer_property=int(manifest["outer_property"]),
        deformable_properties=tuple(manifest["deformable_properties"]),
        tolerance=float(manifest["tolerance"]), coordinate_unit=str(manifest["coordinate_unit"]),
    ))
    old_inner, old_outer = float(manifest["inner_radius"]), float(manifest["outer_radius"])
    if inner_radius is not None and inner_displacement is not None:
        raise MorphingError("specify either inner_radius or inner_displacement, not both")
    if outer_radius is not None and outer_displacement is not None:
        raise MorphingError("specify either outer_radius or outer_displacement, not both")
    try:
        new_inner = old_inner + float(inner_displacement) if inner_displacement is not None else (
            old_inner if inner_radius is None else float(inner_radius)
        )
        new_outer = old_outer + float(outer_displacement) if outer_displacement is not None else (
            old_outer if outer_radius is None else float(outer_radius)
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise MorphingError("shell radii must be numeric") from error
    if not np.isfinite((new_inner, new_outer)).all() or not (0 < new_inner < new_outer):
        raise MorphingError("shell radii must satisfy 0 < inner < outer")
    if new_inner == old_inner and new_outer == old_outer:
        raise MorphingError("at least one shell face must move")
    if interpolation != "analytic_radial":
        raise MorphingError("octant box shell currently supports only analytic_radial propagation")
    points = np.asarray(mesh.points, dtype=float)
    far_radius = float(manifest["far_radius"])
    if new_outer >= far_radius:
        raise MorphingError("outer shell face must remain inside the fixed far boundary")
    old_radius = np.max(points, axis=1)
    new_radius = np.empty_like(old_radius)
    inside = old_radius <= old_inner
    shell = (old_radius > old_inner) & (old_radius <= old_outer)
    outside = old_radius > old_outer
    new_radius[inside] = old_radius[inside] * new_inner / old_inner
    new_radius[shell] = new_inner + (old_radius[shell] - old_inner) * (new_outer - new_inner) / (old_outer - old_inner)
    new_radius[outside] = new_outer + (old_radius[outside] - old_outer) * (far_radius - new_outer) / (far_radius - old_outer)
    target = points.copy()
    positive = old_radius > 0
    target[positive] = points[positive] * (new_radius[positive] / old_radius[positive])[:, None]
    moving_mask = np.linalg.norm(target - points, axis=1) > 1e-13 * far_radius
    target[~moving_mask] = points[~moving_mask]
    moving = np.flatnonzero(moving_mask)
    parameters = {"inner_radius": new_inner, "outer_radius": new_outer}
    if inner_displacement is not None:
        parameters["inner_displacement"] = float(inner_displacement)
    if outer_displacement is not None:
        parameters["outer_displacement"] = float(outer_displacement)
    return _apply_targets(
        mesh, manifest, moving, target, geometry_backend=geometry_backend,
        interpolation=interpolation,
        parameters=parameters,
    )
