"""Export validated Property solids from representative mesh geometry to STEP."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, isclose, radians, sin
from pathlib import Path
import tempfile

import numpy as np

from ems_mesh_checker.geometry.circular_geometry import CircularGeometryNormalizer
from ems_mesh_checker.geometry.curve_simplification import (
    CircularArcPrimitive,
    CurveSimplificationConfig,
    LinePrimitive,
    simplify_planar_loop,
)
from ems_mesh_checker.geometry.planar_regions import PlanarLoop, PlanarRegion, PlanarRegionResult
from ems_mesh_checker.geometry.ruled_surfaces import RuledSurfaceConfig, RuledSurfacePatch, reconstruct_ruled_surfaces
from ems_mesh_checker.export.representative_geo import (
    _detected_arcs,
    _model_extent,
    _physical_label,
    _ruled_cap_arcs,
    _topology_safe_primitives,
    _unique_arcs,
)
from ems_mesh_checker.export.volume_topology import VolumeTopologyConfig


class StepExportError(RuntimeError):
    """Representative geometry cannot be safely exported as a CAD solid."""


@dataclass(frozen=True)
class StepExportResult:
    path: Path
    property_label: str
    solid_count: int
    face_count: int
    unit: str


@dataclass(frozen=True)
class StepAssemblyResult:
    path: Path
    property_count: int
    solid_count: int


def write_planar_regions_step(
    result: PlanarRegionResult,
    path: str | Path,
    *,
    unit: str,
    curve_config: CurveSimplificationConfig | None = None,
    ruled_surface_config: RuledSurfaceConfig | None = None,
    volume_topology_config: VolumeTopologyConfig | None = None,
) -> StepExportResult:
    """Write one Property group as STEP, rejecting incomplete or invalid solids.

    The writer accepts the same validated planar regions and cylindrical-strip
    candidates as the GEO writer. No tessellated or silently repaired fallback is
    used: a missing face must not become a plausible but incorrect CAD solid.
    """

    try:
        import cadquery as cq
        from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing
    except ImportError as error:  # pragma: no cover - depends on installation
        raise StepExportError("CadQuery is required for STEP export") from error

    if result.feature_edges.source_kind != "property":
        raise StepExportError("STEP solid export requires a Property boundary")
    if not result.regions:
        raise StepExportError("no representative surfaces are available")
    if result.unsupported_regions:
        raise StepExportError(
            f"{len(result.unsupported_regions)} surface component(s) could not be reconstructed"
        )
    if volume_topology_config is not None and not volume_topology_config.enabled:
        raise StepExportError("volume reconstruction is disabled")
    unit = unit.upper()
    if unit not in {"MM", "CM", "M", "KM", "INCH", "FT", "MI", "UM", "NM"}:
        raise ValueError(f"unsupported STEP coordinate unit: {unit}")

    extent = _model_extent((result,))
    tolerance = max(1.0e-10, extent * 1.0e-8)
    arc_templates = _unique_arcs(_detected_arcs(result, curve_config), tolerance)
    normalizer = CircularGeometryNormalizer(
        arc_templates, point_tolerance=tolerance, curve_config=curve_config
    )
    ruled = reconstruct_ruled_surfaces(
        result,
        curve_config=curve_config,
        config=ruled_surface_config,
        arc_templates=arc_templates,
    )
    allowed_arcs = _ruled_cap_arcs(ruled)
    replaced_regions = set(ruled.replaced_region_indices)
    faces = []
    preserve_unmatched_arcs = True
    for region in result.regions:
        if region.region_index in replaced_regions:
            continue
        faces.append(
            _planar_face(
                cq,
                region,
                normalizer,
                allowed_arcs=allowed_arcs,
                preserve_unmatched_arcs=preserve_unmatched_arcs,
                tolerance=tolerance,
                curve_config=curve_config,
            )
        )
    for patch in ruled.patches:
        faces.append(_cylindrical_face(cq, result, patch, normalizer, tolerance))
    if not faces:
        raise StepExportError("no valid CAD faces were assembled")

    sewer = BRepBuilderAPI_Sewing(tolerance)
    for face in faces:
        sewer.Add(face.wrapped)
    sewer.Perform()
    sewn = cq.Shape.cast(sewer.SewedShape())
    shells = sewn.Shells()
    if not shells:
        raise StepExportError("CAD faces did not form a shell")
    if len(sewn.Faces()) != len(faces):
        raise StepExportError("CAD sewing changed the surface count")

    solids = []
    for index, shell in enumerate(shells):
        if not shell.Closed() or not shell.isValid():
            raise StepExportError(f"component {index} is an open or invalid CAD shell")
        solid = cq.Solid.makeSolid(shell)
        if not solid.isValid() or solid.Volume() <= 0.0:
            raise StepExportError(f"component {index} is not a valid positive-volume CAD solid")
        solids.append(solid)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    shape = solids[0] if len(solids) == 1 else cq.Compound.makeCompound(solids)
    with tempfile.NamedTemporaryFile(suffix=".step", dir=output.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        shape.exportStep(str(temporary_path), unit=unit)
        imported = cq.importers.importStep(str(temporary_path), unit=unit)
        imported_solids = [item for shape in imported.vals() for item in shape.Solids()]
        if len(imported_solids) != len(solids) or any(not solid.isValid() for solid in imported_solids):
            raise StepExportError("STEP read-back did not preserve the expected solids")
        temporary_path.replace(output)
    finally:
        temporary_path.unlink(missing_ok=True)
    return StepExportResult(output, _physical_label(result, 0), len(solids), len(faces), unit)


def write_step_assembly(
    property_steps: list[StepExportResult] | tuple[StepExportResult, ...],
    path: str | Path,
    *,
    unit: str,
) -> StepAssemblyResult:
    """Combine validated Property STEP solids as named assembly components.

    Components remain independent CAD solids; no Boolean union is attempted.
    The individual STEP files are retained for Property-specific inspection.
    """
    if not property_steps:
        raise StepExportError("no Property STEP solids are available for assembly")
    unit = unit.upper()
    if any(item.unit != unit for item in property_steps):
        raise StepExportError("Property STEP units do not match the assembly unit")
    try:
        import cadquery as cq
    except ImportError as error:  # pragma: no cover - depends on installation
        raise StepExportError("CadQuery is required for STEP export") from error

    output = Path(path)
    assembly = cq.Assembly(name=output.stem)
    expected_volumes = []
    for item in property_steps:
        solids = cq.importers.importStep(str(item.path), unit=unit).solids().vals()
        if len(solids) != item.solid_count or not all(solid.isValid() for solid in solids):
            raise StepExportError(f"Property STEP failed assembly read-back: {item.path}")
        for component_index, solid in enumerate(solids, start=1):
            assembly.add(solid, name=f"{item.property_label}_C{component_index}")
            expected_volumes.append(solid.Volume())

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".step", dir=output.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        assembly.export(str(temporary_path), exportType="STEP", unit=unit)
        actual_solids = cq.importers.importStep(str(temporary_path), unit=unit).solids().vals()
        if len(actual_solids) != len(expected_volumes) or not all(
            solid.isValid() for solid in actual_solids
        ):
            raise StepExportError("combined STEP did not preserve the expected solids")
        if not isclose(
            sum(solid.Volume() for solid in actual_solids),
            sum(expected_volumes),
            rel_tol=1.0e-7,
            abs_tol=1.0e-10,
        ):
            raise StepExportError("combined STEP changed the total solid volume")
        temporary_path.replace(output)
    finally:
        temporary_path.unlink(missing_ok=True)
    return StepAssemblyResult(output, len(property_steps), len(expected_volumes))


def _planar_face(cq, region: PlanarRegion, normalizer: CircularGeometryNormalizer, *,
                 allowed_arcs: tuple[CircularArcPrimitive, ...],
                 preserve_unmatched_arcs: bool, tolerance: float,
                 curve_config: CurveSimplificationConfig | None):
    loops = (region.outer_loop, *region.hole_loops)
    wires = [
        _wire(cq, loop, region.normal, normalizer, allowed_arcs=allowed_arcs,
              preserve_unmatched_arcs=preserve_unmatched_arcs, tolerance=tolerance,
              curve_config=curve_config)
        for loop in loops
    ]
    try:
        face = cq.Face.makeFromWires(wires[0], wires[1:])
    except Exception as error:
        raise StepExportError(f"region {region.region_index} could not form a CAD plane: {error}") from error
    if not face.isValid() or face.Area() <= 0.0:
        raise StepExportError(f"region {region.region_index} produced an invalid CAD face")
    normal = np.asarray(face.normalAt().toTuple(), dtype=float)
    if np.dot(normal, region.normal) < 0.0:
        face = cq.Face(face.wrapped.Reversed())
    return face


def _wire(cq, loop: PlanarLoop, normal: np.ndarray, normalizer: CircularGeometryNormalizer,
          *, allowed_arcs: tuple[CircularArcPrimitive, ...], preserve_unmatched_arcs: bool,
          tolerance: float, curve_config: CurveSimplificationConfig | None):
    simplified = simplify_planar_loop(loop, normal=normal, config=curve_config)
    primitives = _topology_safe_primitives(
        loop,
        simplified.primitives,
        allowed_arcs=allowed_arcs,
        preserve_unmatched_arcs_as_lines=preserve_unmatched_arcs,
        tolerance=tolerance,
    )
    edges = []
    for raw in primitives:
        primitive = normalizer.normalize_primitive(raw)
        if isinstance(primitive, LinePrimitive):
            edges.append(cq.Edge.makeLine(tuple(primitive.start), tuple(primitive.end)))
        else:
            midpoint = _arc_middle(primitive, loop, normal)
            edges.append(cq.Edge.makeThreePointArc(tuple(primitive.start), tuple(midpoint), tuple(primitive.end)))
    try:
        wire = cq.Wire.assembleEdges(edges)
    except Exception as error:
        raise StepExportError(f"representative edges could not form a CAD wire: {error}") from error
    if not wire.isValid() or not wire.Closed():
        raise StepExportError("representative CAD wire is open or invalid")
    return wire


def _arc_middle(arc: CircularArcPrimitive, loop: PlanarLoop, region_normal: np.ndarray) -> np.ndarray:
    points = np.asarray(loop.points, dtype=float)
    start_index = int(np.argmin(np.linalg.norm(points - arc.start, axis=1)))
    if arc.source_edge_count >= 2:
        middle_index = (start_index + max(1, arc.source_edge_count // 2)) % len(points)
        sample = points[middle_index] - arc.center
        radius = float(np.linalg.norm(sample))
        if radius > 0.0 and np.linalg.norm(points[middle_index] - arc.end) > 1.0e-12:
            return arc.center + float(arc.radius) * sample / radius
    axis = np.asarray(arc.plane_normal if arc.plane_normal is not None else region_normal, dtype=float)
    axis /= np.linalg.norm(axis)
    vector = np.asarray(arc.start - arc.center, dtype=float)
    angle = radians(float(arc.angle_degrees) / 2.0)
    return arc.center + vector * cos(angle) + np.cross(axis, vector) * sin(angle)


def _cylindrical_face(cq, result: PlanarRegionResult, patch: RuledSurfacePatch,
                      normalizer: CircularGeometryNormalizer, tolerance: float):
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCP.GeomAbs import GeomAbs_Cylinder
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt

    lower = normalizer.normalize_primitive(patch.primitives[0])
    upper_reverse = normalizer.normalize_primitive(patch.primitives[2])
    if not isinstance(lower, CircularArcPrimitive) or not isinstance(upper_reverse, CircularArcPrimitive):
        raise StepExportError("a cylindrical patch lacks circular cap edges")
    extrusion = np.asarray(patch.extrusion_vector, dtype=float)
    height = float(np.linalg.norm(extrusion))
    if height <= tolerance:
        raise StepExportError("a cylindrical patch has zero height")
    axis = extrusion / height
    first = np.asarray(lower.start - lower.center, dtype=float)
    last = np.asarray(lower.end - lower.center, dtype=float)
    first /= np.linalg.norm(first)
    last /= np.linalg.norm(last)
    signed_angle = atan2(float(np.dot(axis, np.cross(first, last))), float(np.dot(first, last)))
    sweep = radians(float(lower.angle_degrees))
    if abs(abs(signed_angle) - sweep) > 1.0e-3:
        raise StepExportError("a cylindrical patch has inconsistent arc direction")
    x_direction = first if signed_angle > 0.0 else last
    frame = gp_Ax2(gp_Pnt(*map(float, lower.center)), gp_Dir(*map(float, axis)),
                   gp_Dir(*map(float, x_direction)))
    wedge = cq.Shape.cast(BRepPrimAPI_MakeCylinder(frame, float(lower.radius), height, sweep).Shape())
    cylinder_faces = [face for face in wedge.Faces()
                      if BRepAdaptor_Surface(face.wrapped).GetType() == GeomAbs_Cylinder]
    if len(cylinder_faces) != 1:
        raise StepExportError("the CAD kernel did not construct one analytical cylinder face")
    face = cylinder_faces[0]
    source_normals = np.asarray(
        [result.regions[index].normal for index in patch.source_region_indices], dtype=float
    )
    target_normal = np.mean(source_normals, axis=0)
    if np.linalg.norm(target_normal) > 1.0e-12:
        if np.dot(np.asarray(face.normalAt().toTuple()), target_normal) < 0.0:
            face = cq.Face(face.wrapped.Reversed())
    if not face.isValid():
        raise StepExportError("the analytical cylinder face is invalid")
    return face
