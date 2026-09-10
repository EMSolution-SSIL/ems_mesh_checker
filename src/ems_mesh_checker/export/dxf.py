"""Simple DXF export for meshio boundary lines and faces using ezdxf."""

from __future__ import annotations

from math import cos, radians, sin
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ems_mesh_checker.geometry.curve_simplification import (
    CircularArcPrimitive,
    CurvePrimitive,
    CurveSimplificationConfig,
    LinePrimitive,
)
from ems_mesh_checker.geometry.planar_regions import PlanarRegionResult
from ems_mesh_checker.geometry.ruled_surfaces import RuledSurfaceConfig

from .representative_geo import collect_representative_curve_groups
from .volume_topology import VolumeTopologyConfig


class DXFExportError(RuntimeError):
    """Raised when DXF export cannot be completed."""


def write_dxf(surface_mesh: Any, path: str | Path, *, layer: str | None = None) -> Path:
    """Write a boundary ``meshio.Mesh`` as DXF `LINE` and `3DFACE` entities.

    When ``layer`` is omitted, layers are derived from owner/neighbor Property IDs:
    `EXTERNAL_P<id>` or `INTERFACE_P<low>_P<high>`.
    """

    ezdxf = _import_ezdxf()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Gmsh's DXF reader supports the compact legacy ASCII feature set more
    # consistently than newer AC10xx subclass-heavy variants. LINE, 3DFACE and
    # layers used here are all part of DXF R12.
    document = ezdxf.new("R12")
    _add_surface_mesh(document, surface_mesh, layer=layer)
    document.saveas(output)
    return output


def write_boundary_result_dxf(result: Any, path: str | Path) -> Path:
    """Write external and Property-interface boundaries of an extraction result.

    The external surface is emitted on `EXTERNAL`; each interface uses a dedicated
    `INTERFACE_P<low>_P<high>` layer. This avoids losing the Property-pair identity
    when a single DXF contains every boundary surface.
    """

    ezdxf = _import_ezdxf()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    document = ezdxf.new("R12")
    _add_surface_mesh(document, result.external_surface, layer="EXTERNAL")
    for property_a, property_b in result.property_interfaces:
        _add_surface_mesh(
            document,
            result.get_interface(property_a, property_b),
            layer=f"INTERFACE_P{property_a}_P{property_b}",
        )
    document.saveas(output)
    return output


def write_feature_edges_dxf(result: Any, path: str | Path, *, layer: str | None = None) -> Path:
    """Write one representative feature-edge result as 3-D DXF polylines.

    This is intentionally separate from :func:`write_dxf`: the latter writes
    every finite-element face, whereas this function writes the post-processed
    feature curves returned by the PyVista geometry layer.
    """

    return write_feature_edge_results_dxf((result,), path, layer=layer)


def write_feature_edge_results_dxf(
    results: Any,
    path: str | Path,
    *,
    layer: str | None = None,
) -> Path:
    """Write feature curves from one or more results as DXF ``POLYLINE`` entities."""

    ezdxf = _import_ezdxf()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    # POLYLINE is available in legacy DXF and supports non-planar feature
    # curves.  The representative DXF is for inspection/CAD exchange; Gmsh
    # validation continues to use the corresponding .geo output.
    document = ezdxf.new("R12")
    modelspace = document.modelspace()
    for result in results:
        entity_layer = layer or _feature_edge_layer(result)
        _ensure_layer(document, entity_layer)
        for curve in result.curves:
            vertices = [_as_xyz(point) for point in np.asarray(curve.points)]
            if curve.closed and vertices:
                vertices.append(vertices[0])
            if len(vertices) >= 2:
                modelspace.add_polyline3d(vertices, dxfattribs={"layer": entity_layer})
    document.saveas(output)
    return output


def write_planar_regions_dxf(
    results: PlanarRegionResult | Iterable[PlanarRegionResult],
    path: str | Path,
    *,
    curve_config: CurveSimplificationConfig | None = None,
    ruled_surface_config: RuledSurfaceConfig | None = None,
    volume_topology_config: VolumeTopologyConfig | None = None,
) -> Path:
    """Write representative geometry as shared DXF ``LINE`` and ``ARC`` entities.

    This curve-only export uses the same conservative arc recognition and
    cylindrical-strip reconstruction as :func:`write_planar_regions_geo`.
    Coincident curves are emitted once; a curve used by multiple Properties is
    placed on a combined ``SHARED_*`` layer. No mesh faces, surfaces or volumes
    are written.
    """

    groups = collect_representative_curve_groups(
        results,
        curve_config=curve_config,
        ruled_surface_config=ruled_surface_config,
        volume_topology_config=volume_topology_config,
    )
    primitives = tuple(
        primitive
        for group in groups
        for primitive in group.primitives
    )
    tolerance = _representative_curve_tolerance(primitives, volume_topology_config)
    records: dict[tuple[Any, ...], tuple[CurvePrimitive, set[str]]] = {}
    for group in groups:
        for primitive in group.primitives:
            key = _representative_curve_key(primitive, tolerance)
            if key in records:
                records[key][1].add(group.label)
            else:
                records[key] = (primitive, {group.label})

    ezdxf = _import_ezdxf()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    document = ezdxf.new("R12")
    modelspace = document.modelspace()
    for primitive, labels in records.values():
        entity_layer = _shared_curve_layer(labels)
        _ensure_layer(document, entity_layer)
        attributes = {"layer": entity_layer}
        if isinstance(primitive, CircularArcPrimitive):
            _add_circular_arc(modelspace, primitive, attributes)
        elif isinstance(primitive, LinePrimitive):
            modelspace.add_line(
                _as_xyz(primitive.start),
                _as_xyz(primitive.end),
                dxfattribs=attributes,
            )
        else:  # pragma: no cover - CurvePrimitive currently has two variants
            raise DXFExportError(
                f"unsupported representative curve primitive: {type(primitive).__name__}"
            )
    document.saveas(output)
    return output


def _add_circular_arc(modelspace: Any, arc: CircularArcPrimitive, attributes: dict[str, str]) -> None:
    """Create an arbitrary-plane DXF ARC whose WCS endpoints match ``arc``."""

    from ezdxf.math import Matrix44

    start_vector = np.asarray(arc.start, dtype=float) - arc.center
    end_vector = np.asarray(arc.end, dtype=float) - arc.center
    radius = float(np.linalg.norm(start_vector))
    end_radius = float(np.linalg.norm(end_vector))
    if radius <= 0.0 or end_radius <= 0.0:
        raise DXFExportError("DXF ARC export requires a positive radius")
    unit_x = start_vector / radius
    unit_end = end_vector / end_radius
    normal = (
        np.asarray(arc.plane_normal, dtype=float)
        if arc.plane_normal is not None
        else np.cross(unit_x, unit_end)
    )
    normal_length = float(np.linalg.norm(normal))
    if normal_length <= 0.0:
        raise DXFExportError(
            "DXF ARC export requires non-collinear endpoint radii or a plane normal"
        )
    unit_z = normal / normal_length
    unit_y = np.cross(unit_z, unit_x)
    angle_degrees = float(arc.angle_degrees)
    entity = modelspace.add_arc(
        (0.0, 0.0, 0.0),
        radius,
        0.0,
        angle_degrees,
        dxfattribs=attributes,
    )
    entity.transform(
        Matrix44.ucs(
            ux=unit_x,
            uy=unit_y,
            uz=unit_z,
            origin=np.asarray(arc.center, dtype=float),
        )
    )


def _representative_curve_tolerance(
    primitives: tuple[CurvePrimitive, ...],
    volume_topology_config: VolumeTopologyConfig | None,
) -> float:
    settings = volume_topology_config or VolumeTopologyConfig()
    points = np.vstack(
        tuple(
            point
            for primitive in primitives
            for point in (primitive.start, primitive.end)
        )
    )
    extent = float(np.linalg.norm(np.ptp(points, axis=0)))
    return max(
        settings.point_absolute_tolerance,
        extent * settings.point_relative_tolerance,
    )


def _representative_curve_key(
    primitive: CurvePrimitive,
    tolerance: float,
) -> tuple[Any, ...]:
    start = _quantized_point(primitive.start, tolerance)
    end = _quantized_point(primitive.end, tolerance)
    endpoints = tuple(sorted((start, end)))
    if isinstance(primitive, CircularArcPrimitive):
        midpoint = _representative_arc_midpoint(primitive)
        return (
            "ARC",
            _quantized_point(primitive.center, tolerance),
            int(round(float(primitive.radius) / tolerance)),
            endpoints,
            None if midpoint is None else _quantized_point(midpoint, tolerance),
        )
    return ("LINE", endpoints)


def _quantized_point(point: np.ndarray, tolerance: float) -> tuple[int, int, int]:
    coordinate = np.asarray(point, dtype=float).reshape(3)
    return tuple(int(round(float(value) / tolerance)) for value in coordinate)


def _representative_arc_midpoint(arc: CircularArcPrimitive) -> np.ndarray | None:
    center = np.asarray(arc.center, dtype=float)
    start_vector = np.asarray(arc.start, dtype=float) - center
    radius = float(np.linalg.norm(start_vector))
    if radius <= 0.0:
        return None
    unit_start = start_vector / radius
    normal = (
        np.asarray(arc.plane_normal, dtype=float)
        if arc.plane_normal is not None
        else np.cross(unit_start, np.asarray(arc.end, dtype=float) - center)
    )
    normal_length = float(np.linalg.norm(normal))
    if normal_length <= 0.0:
        return None
    normal /= normal_length
    half_angle = radians(float(arc.angle_degrees)) / 2.0
    direction = (
        cos(half_angle) * unit_start
        + sin(half_angle) * np.cross(normal, unit_start)
        + (1.0 - cos(half_angle)) * np.dot(normal, unit_start) * normal
    )
    return center + radius * direction


def _shared_curve_layer(labels: set[str]) -> str:
    ordered = sorted(labels)
    if len(ordered) == 1:
        return ordered[0]
    return "SHARED_PROPERTIES"


def _import_ezdxf() -> Any:
    try:
        import ezdxf
    except ImportError as error:
        raise DXFExportError(
            "DXF export requires ezdxf. Install the optional dependency with "
            'pip install "ems-mesh-checker[dxf]".'
        ) from error
    return ezdxf


def _add_surface_mesh(document: Any, surface_mesh: Any, *, layer: str | None) -> None:
    modelspace = document.modelspace()
    points = np.asarray(surface_mesh.points)
    for block_index, block in enumerate(surface_mesh.cells):
        cell_type = str(block.type)
        for local_index, connectivity in enumerate(np.asarray(block.data)):
            entity_layer = layer or _layer_from_metadata(surface_mesh, block_index, local_index)
            _ensure_layer(document, entity_layer)
            vertices = [_as_xyz(points[int(point_index)]) for point_index in connectivity]
            if cell_type == "line":
                if len(vertices) != 2:
                    raise DXFExportError("DXF LINE export requires two-node cells")
                modelspace.add_line(vertices[0], vertices[1], dxfattribs={"layer": entity_layer})
            elif cell_type in {"triangle", "quad"}:
                if len(vertices) == 3:
                    vertices.append(vertices[-1])
                if len(vertices) != 4:
                    raise DXFExportError(f"DXF 3DFACE export requires 3 or 4 nodes, got {len(vertices)}")
                modelspace.add_3dface(vertices, dxfattribs={"layer": entity_layer})
            else:
                raise DXFExportError(f"DXF export does not support surface cell type: {cell_type}")


def _as_xyz(point: np.ndarray) -> tuple[float, float, float]:
    if len(point) == 2:
        return (float(point[0]), float(point[1]), 0.0)
    return (float(point[0]), float(point[1]), float(point[2]))


def _layer_from_metadata(surface_mesh: Any, block_index: int, local_index: int) -> str:
    owner_property = _cell_value(surface_mesh, "owner_property_id", block_index, local_index)
    neighbor_property = _cell_value(surface_mesh, "neighbor_property_id", block_index, local_index)
    if owner_property is None or owner_property < 0:
        return "BOUNDARY"
    if neighbor_property is None or neighbor_property < 0:
        return f"EXTERNAL_P{owner_property}"
    property_a, property_b = sorted((owner_property, neighbor_property))
    return f"INTERFACE_P{property_a}_P{property_b}"


def _cell_value(surface_mesh: Any, name: str, block_index: int, local_index: int) -> int | None:
    values = getattr(surface_mesh, "cell_data", {}).get(name)
    if values is None or block_index >= len(values):
        return None
    array = np.asarray(values[block_index])
    if local_index >= len(array):
        return None
    return int(array[local_index])


def _ensure_layer(document: Any, name: str) -> None:
    if not document.layers.has_entry(name):
        document.layers.add(name)


def _feature_edge_layer(result: Any) -> str:
    if getattr(result, "source_kind", None) == "property":
        property_ids = tuple(getattr(result, "property_ids", ()))
        if len(property_ids) > 1:
            suffix = "_".join(f"P{property_id}" for property_id in property_ids)
            return f"REP_PROPERTY_GROUP_{suffix}"
        property_id = getattr(result, "property_id", None)
        if property_id is not None:
            return f"REP_PROPERTY_P{property_id}"
    if getattr(result, "source_kind", None) == "interface":
        property_pair = getattr(result, "property_pair", None)
        patch_index = getattr(result, "patch_index", None)
        if property_pair is not None:
            suffix = "" if patch_index is None else f"_N{patch_index}"
            return f"REP_INT_P{property_pair[0]}_P{property_pair[1]}{suffix}"
    if getattr(result, "source_kind", None) == "exterior":
        return "REP_EXTERIOR"
    return "REP_FEATURE"
