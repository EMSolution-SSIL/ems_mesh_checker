"""Femap Neutral export for representative point and curve geometry."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from itertools import product
from math import cos, radians, sin
from pathlib import Path
from typing import Any

import numpy as np

from ..geometry.curve_simplification import (
    CircularArcPrimitive,
    CurvePrimitive,
    CurveSimplificationConfig,
    LinePrimitive,
)
from ..geometry.planar_regions import PlanarRegionResult
from ..geometry.ruled_surfaces import RuledSurfaceConfig
from .dxf import (
    _representative_curve_key,
    _representative_curve_tolerance,
    _shared_curve_layer,
)
from .representative_geo import (
    RepresentativeCurveGroup,
    RepresentativeGeometryExportError,
    collect_representative_curve_groups,
)
from .volume_topology import VolumeTopologyConfig


class FemapNeutralGeometryExportError(RuntimeError):
    """Raised when representative curves cannot be written as Femap geometry."""


def write_planar_regions_femap_neutral(
    results: PlanarRegionResult | Iterable[PlanarRegionResult],
    path: str | Path,
    *,
    curve_config: CurveSimplificationConfig | None = None,
    ruled_surface_config: RuledSurfaceConfig | None = None,
    volume_topology_config: VolumeTopologyConfig | None = None,
    title: str = "ems_mesh_checker representative curve geometry",
) -> Path:
    """Write shared representative curves as Femap geometry blocks 570 and 571.

    The output contains editable Femap points, lines and circular arcs.  It is a
    curve-geometry interchange file: surface block 572 and Parasolid/ACIS solid
    block 573 are intentionally not emitted.
    """

    groups = collect_representative_curve_groups(
        results,
        curve_config=curve_config,
        ruled_surface_config=ruled_surface_config,
        volume_topology_config=volume_topology_config,
    )
    return _write_representative_curve_groups_femap_neutral(
        groups,
        path,
        volume_topology_config=volume_topology_config,
        title=title,
    )


def _write_representative_curve_groups_femap_neutral(
    groups: Iterable[RepresentativeCurveGroup],
    path: str | Path,
    *,
    volume_topology_config: VolumeTopologyConfig | None = None,
    title: str = "ems_mesh_checker representative curve geometry",
) -> Path:
    curve_groups = tuple(groups)
    primitives = tuple(
        primitive for group in curve_groups for primitive in group.primitives
    )
    if not primitives:
        raise RepresentativeGeometryExportError(
            "no representative line or arc geometry is available"
        )

    tolerance = _representative_curve_tolerance(
        primitives,
        volume_topology_config,
    )
    records: dict[tuple[Any, ...], tuple[CurvePrimitive, set[str]]] = {}
    for group in curve_groups:
        for primitive in group.primitives:
            key = _representative_curve_key(primitive, tolerance)
            if key in records:
                records[key][1].add(group.label)
            else:
                records[key] = (primitive, {group.label})

    curve_layers = sorted(
        {_shared_curve_layer(labels) for _, labels in records.values()}
    )
    layer_ids = {name: index + 2 for index, name in enumerate(curve_layers)}
    point_registry = _PointRegistry(tolerance)
    curve_rows: list[tuple[CurvePrimitive, str, tuple[int, ...]]] = []
    for primitive, labels in records.values():
        definition_points = _curve_definition_points(primitive, tolerance)
        point_ids = tuple(point_registry.add(point) for point in definition_points)
        curve_rows.append((primitive, _shared_curve_layer(labels), point_ids))

    lines = _header_block(title)
    lines.extend(_layer_block(curve_layers))
    lines.extend(_point_block(point_registry.points))
    lines.extend(_curve_block(curve_rows, layer_ids))

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="ascii", newline="\r\n") as stream:
        stream.write("\n".join(lines))
        stream.write("\n")
    return output


class _PointRegistry:
    def __init__(self, tolerance: float) -> None:
        self.tolerance = tolerance
        self.points: list[np.ndarray] = []
        self._bins: dict[tuple[int, int, int], list[int]] = defaultdict(list)

    def add(self, coordinate: np.ndarray) -> int:
        point = np.asarray(coordinate, dtype=float)
        key = tuple(int(round(value / self.tolerance)) for value in point)
        for offset in product((-1, 0, 1), repeat=3):
            neighbor = tuple(key[axis] + offset[axis] for axis in range(3))
            for identifier in self._bins[neighbor]:
                if np.linalg.norm(self.points[identifier - 1] - point) <= self.tolerance:
                    return identifier
        identifier = len(self.points) + 1
        self.points.append(point.copy())
        self._bins[key].append(identifier)
        return identifier


def _curve_definition_points(
    primitive: CurvePrimitive,
    tolerance: float,
) -> tuple[np.ndarray, ...]:
    if isinstance(primitive, LinePrimitive):
        return (primitive.start, primitive.end)
    if not isinstance(primitive, CircularArcPrimitive):  # pragma: no cover
        raise FemapNeutralGeometryExportError(
            f"unsupported representative curve primitive: {type(primitive).__name__}"
        )

    center = np.asarray(primitive.center, dtype=float)
    start_vector = np.asarray(primitive.start, dtype=float) - center
    end_vector = np.asarray(primitive.end, dtype=float) - center
    start_radius = float(np.linalg.norm(start_vector))
    end_radius = float(np.linalg.norm(end_vector))
    if start_radius <= tolerance or end_radius <= tolerance:
        raise FemapNeutralGeometryExportError("circular arc has a degenerate radius")
    start_direction = start_vector / start_radius
    end_direction = end_vector / end_radius
    normal = (
        np.asarray(primitive.plane_normal, dtype=float)
        if primitive.plane_normal is not None
        else np.cross(start_direction, end_direction)
    )
    normal_length = float(np.linalg.norm(normal))
    if normal_length <= tolerance:
        raise FemapNeutralGeometryExportError(
            "a 180-degree arc cannot be oriented without an arc-plane normal"
        )
    normal /= normal_length
    half_angle = radians(float(primitive.angle_degrees)) / 2.0
    midpoint_direction = (
        cos(half_angle) * start_direction
        + sin(half_angle) * np.cross(normal, start_direction)
        + (1.0 - cos(half_angle)) * np.dot(normal, start_direction) * normal
    )
    midpoint_direction /= np.linalg.norm(midpoint_direction)
    midpoint = center + start_radius * midpoint_direction
    endpoint = center + start_radius * end_direction
    # Femap's 3D arc definition follows center, start, through-point, end.
    return (center, np.asarray(primitive.start, dtype=float), midpoint, endpoint)


def _header_block(title: str) -> list[str]:
    return [
        "   -1",
        "100",
        _text(title),
        "10.3,",
        "   -1",
    ]


def _layer_block(curve_layers: list[str]) -> list[str]:
    lines = ["   -1", "413", "1,24,", "REPRESENTATIVE_POINTS"]
    for identifier, name in enumerate(curve_layers, start=2):
        lines.extend((f"{identifier},120,", _text(name)))
    lines.append("   -1")
    return lines


def _point_block(points: list[np.ndarray]) -> list[str]:
    lines = ["   -1", "570"]
    for identifier, point in enumerate(points, start=1):
        lines.append(f"{identifier},0,0,0,1,24,0.,0,0,0,")
        lines.append(",".join(_number(value) for value in point) + ",")
    lines.append("   -1")
    return lines


def _curve_block(
    rows: list[tuple[CurvePrimitive, str, tuple[int, ...]]],
    layer_ids: dict[str, int],
) -> list[str]:
    lines = ["   -1", "571"]
    for identifier, (primitive, layer_name, point_ids) in enumerate(rows, start=1):
        curve_type = 1 if isinstance(primitive, CircularArcPrimitive) else 0
        points = (*point_ids, *(0 for _ in range(5 - len(point_ids))))
        lines.extend(
            (
                f"{identifier},120,{curve_type},{layer_ids[layer_name]},0,1.,0,0,0,0,0,",
                "0,0,0,",
                "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,",
                "0.,0.,0.,",
                "0.,0.,0.,0.,0.,0.,",
                ",".join(str(value) for value in points) + ",",
            )
        )
    lines.append("   -1")
    return lines


def _number(value: float) -> str:
    numeric = float(value)
    if numeric == 0.0:
        numeric = 0.0
    return f"{numeric:.16g}"


def _text(value: str) -> str:
    text = value.encode("ascii", errors="replace").decode("ascii")[:79]
    return text or "<NULL>"
