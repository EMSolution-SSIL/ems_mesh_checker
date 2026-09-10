"""Conservatively replace planar-loop polylines with lines and circular arcs."""

from __future__ import annotations

from dataclasses import dataclass
from math import radians

import numpy as np

from .planar_regions import PlanarLoop


@dataclass(frozen=True)
class CurveSimplificationConfig:
    """Controls conservative circular-arc recognition.

    At least four sampled points are required because three points always define
    a circle.  ``max_sample_angle_degrees`` also prevents a square or another
    coarsely sampled polygon from being mistaken for a circular boundary.
    """

    enabled: bool = True
    min_arc_points: int = 4
    min_arc_angle_degrees: float = 10.0
    max_arc_angle_degrees: float = 179.0
    max_sample_angle_degrees: float = 30.0
    radial_absolute_tolerance: float = 1.0e-10
    radial_relative_tolerance: float = 1.0e-4
    max_radius_to_loop_extent: float = 1.0e4

    def __post_init__(self) -> None:
        if self.min_arc_points < 4:
            raise ValueError("min_arc_points must be at least 4")
        if not 0.0 < self.min_arc_angle_degrees < 180.0:
            raise ValueError("min_arc_angle_degrees must be in (0, 180)")
        if not self.min_arc_angle_degrees < self.max_arc_angle_degrees < 180.0:
            raise ValueError("max_arc_angle_degrees must be between min_arc_angle_degrees and 180")
        if not 0.0 < self.max_sample_angle_degrees < 180.0:
            raise ValueError("max_sample_angle_degrees must be in (0, 180)")
        if self.radial_absolute_tolerance < 0.0 or self.radial_relative_tolerance < 0.0:
            raise ValueError("radial tolerances must be non-negative")
        if self.max_radius_to_loop_extent <= 0.0:
            raise ValueError("max_radius_to_loop_extent must be positive")


@dataclass(frozen=True)
class LinePrimitive:
    """One straight loop segment."""

    start: np.ndarray
    end: np.ndarray
    source_edge_count: int = 1


@dataclass(frozen=True)
class CircularArcPrimitive:
    """One circular arc whose endpoints remain shared with adjacent primitives."""

    start: np.ndarray
    center: np.ndarray
    end: np.ndarray
    radius: float
    angle_degrees: float
    max_radial_error: float
    source_edge_count: int
    plane_normal: np.ndarray | None = None


CurvePrimitive = LinePrimitive | CircularArcPrimitive


@dataclass(frozen=True)
class SimplifiedCurveLoop:
    """Ordered line/arc primitives forming the same closed loop topology."""

    primitives: tuple[CurvePrimitive, ...]
    original_edge_count: int

    @property
    def line_count(self) -> int:
        return sum(isinstance(item, LinePrimitive) for item in self.primitives)

    @property
    def arc_count(self) -> int:
        return sum(isinstance(item, CircularArcPrimitive) for item in self.primitives)

    @property
    def max_radial_error(self) -> float:
        return max(
            (item.max_radial_error for item in self.primitives if isinstance(item, CircularArcPrimitive)),
            default=0.0,
        )


@dataclass(frozen=True)
class _ArcCandidate:
    start_index: int
    edge_count: int
    center_2d: np.ndarray
    radius: float
    angle_degrees: float
    max_radial_error: float

    @property
    def edge_indices(self) -> tuple[int, ...]:
        return tuple(self.start_index + offset for offset in range(self.edge_count))


def simplify_planar_loop(
    loop: PlanarLoop,
    *,
    normal: np.ndarray,
    config: CurveSimplificationConfig | None = None,
) -> SimplifiedCurveLoop:
    """Return a closed sequence of line/arc primitives for ``loop``.

    Arc endpoints are never moved for partial arcs.  The fitted center is
    constrained to the endpoint perpendicular bisector, so Gmsh receives equal
    start/end radii while adjacent line connectivity remains exact.
    """

    settings = config or CurveSimplificationConfig()
    points = np.asarray(loop.points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ValueError("planar loop points must have shape (n, 3), n >= 3")
    if not settings.enabled:
        return _line_only_loop(points)

    origin, basis_u, basis_v = _plane_basis(points, np.asarray(normal, dtype=float))
    projected = np.column_stack(((points - origin) @ basis_u, (points - origin) @ basis_v))
    loop_extent = float(np.linalg.norm(np.ptp(projected, axis=0)))
    if loop_extent <= 0.0:
        return _line_only_loop(points)

    full_circle = _fit_full_circle(projected, loop_extent, settings)
    if full_circle is not None:
        center_2d, radius, max_error, angular_steps = full_circle
        return _full_circle_loop(
            points,
            projected,
            origin,
            basis_u,
            basis_v,
            center_2d,
            radius,
            max_error,
            angular_steps,
            settings,
        )

    candidates = _arc_candidates(projected, loop_extent, settings)
    selected = _select_non_overlapping(candidates, len(points))
    if not selected:
        return _line_only_loop(points)
    primitives = _assemble_primitives(
        points,
        origin,
        basis_u,
        basis_v,
        selected,
    )
    return SimplifiedCurveLoop(tuple(primitives), len(points))


def _line_only_loop(points: np.ndarray) -> SimplifiedCurveLoop:
    primitives = tuple(
        LinePrimitive(points[index].copy(), points[(index + 1) % len(points)].copy())
        for index in range(len(points))
    )
    return SimplifiedCurveLoop(primitives, len(points))


def _plane_basis(
    points: np.ndarray,
    normal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    length = float(np.linalg.norm(normal))
    if length == 0.0:
        raise ValueError("loop normal must be non-zero")
    unit_normal = normal / length
    axis = np.eye(3)[int(np.argmin(np.abs(unit_normal)))]
    basis_u = np.cross(axis, unit_normal)
    basis_u /= np.linalg.norm(basis_u)
    basis_v = np.cross(unit_normal, basis_u)
    return np.mean(points, axis=0), basis_u, basis_v


def _arc_candidates(
    points: np.ndarray,
    loop_extent: float,
    config: CurveSimplificationConfig,
) -> tuple[_ArcCandidate, ...]:
    candidates: list[_ArcCandidate] = []
    count = len(points)
    minimum_edges = config.min_arc_points - 1
    for start in range(count):
        for edge_count in range(minimum_edges, count):
            indices = [(start + offset) % count for offset in range(edge_count + 1)]
            candidate = _fit_partial_arc(
                points[np.asarray(indices, dtype=int)],
                start,
                edge_count,
                loop_extent,
                config,
            )
            if candidate is not None:
                candidates.append(candidate)
    return tuple(candidates)


def _fit_partial_arc(
    points: np.ndarray,
    start_index: int,
    edge_count: int,
    loop_extent: float,
    config: CurveSimplificationConfig,
) -> _ArcCandidate | None:
    start = points[0]
    end = points[-1]
    chord = end - start
    chord_length = float(np.linalg.norm(chord))
    if chord_length <= loop_extent * 1.0e-12:
        return None
    midpoint = 0.5 * (start + end)
    perpendicular = np.array([-chord[1], chord[0]], dtype=float) / chord_length

    offsets = points[1:-1] - start
    constant = np.sum(points[1:-1] ** 2, axis=1) - float(np.dot(start, start))
    constant -= 2.0 * (offsets @ midpoint)
    coefficient = -2.0 * (offsets @ perpendicular)
    denominator = float(np.dot(coefficient, coefficient))
    if denominator <= loop_extent * loop_extent * 1.0e-24:
        return None
    distance = -float(np.dot(coefficient, constant)) / denominator
    center = midpoint + distance * perpendicular
    radius = float(np.linalg.norm(start - center))
    if radius <= loop_extent * 1.0e-12:
        return None
    if radius > loop_extent * config.max_radius_to_loop_extent:
        return None

    radii = np.linalg.norm(points - center, axis=1)
    error = float(np.max(np.abs(radii - radius)))
    tolerance = max(config.radial_absolute_tolerance, radius * config.radial_relative_tolerance)
    if error > tolerance:
        return None
    angle_data = _monotonic_angles(points, center, include_closure=False)
    if angle_data is None:
        return None
    steps, span = angle_data
    if np.max(np.abs(steps)) > radians(config.max_sample_angle_degrees) * (1.0 + 1.0e-10):
        return None
    angle_degrees = abs(float(np.degrees(span)))
    if not config.min_arc_angle_degrees <= angle_degrees <= config.max_arc_angle_degrees:
        return None
    return _ArcCandidate(start_index, edge_count, center, radius, angle_degrees, error)


def _fit_full_circle(
    points: np.ndarray,
    loop_extent: float,
    config: CurveSimplificationConfig,
) -> tuple[np.ndarray, float, float, np.ndarray] | None:
    if len(points) < max(config.min_arc_points, 6):
        return None
    matrix = np.column_stack((2.0 * points[:, 0], 2.0 * points[:, 1], np.ones(len(points))))
    target = np.sum(points**2, axis=1)
    center_x, center_y, constant = np.linalg.lstsq(matrix, target, rcond=None)[0]
    center = np.array([center_x, center_y], dtype=float)
    radius_squared = float(constant + np.dot(center, center))
    if radius_squared <= 0.0:
        return None
    radius = float(np.sqrt(radius_squared))
    if radius > loop_extent * config.max_radius_to_loop_extent:
        return None
    radii = np.linalg.norm(points - center, axis=1)
    error = float(np.max(np.abs(radii - radius)))
    tolerance = max(config.radial_absolute_tolerance, radius * config.radial_relative_tolerance)
    if error > tolerance:
        return None
    angle_data = _monotonic_angles(points, center, include_closure=True)
    if angle_data is None:
        return None
    steps, span = angle_data
    if np.max(np.abs(steps)) > radians(config.max_sample_angle_degrees) * (1.0 + 1.0e-10):
        return None
    if abs(abs(span) - 2.0 * np.pi) > 1.0e-6:
        return None
    return center, radius, error, steps


def _monotonic_angles(
    points: np.ndarray,
    center: np.ndarray,
    *,
    include_closure: bool,
) -> tuple[np.ndarray, float] | None:
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    if include_closure:
        angles = np.append(angles, angles[0])
    unwrapped = np.unwrap(angles)
    steps = np.diff(unwrapped)
    threshold = 1.0e-12
    if not len(steps) or np.any(np.abs(steps) <= threshold):
        return None
    if not (np.all(steps > 0.0) or np.all(steps < 0.0)):
        return None
    return steps, float(np.sum(steps))


def _select_non_overlapping(
    candidates: tuple[_ArcCandidate, ...],
    point_count: int,
) -> tuple[_ArcCandidate, ...]:
    selected: list[_ArcCandidate] = []
    occupied: set[int] = set()
    ordered = sorted(
        candidates,
        key=lambda item: (-item.edge_count, item.max_radial_error, item.start_index),
    )
    for candidate in ordered:
        edges = {(index % point_count) for index in candidate.edge_indices}
        if occupied.isdisjoint(edges):
            selected.append(candidate)
            occupied.update(edges)
    return tuple(sorted(selected, key=lambda item: item.start_index))


def _assemble_primitives(
    points: np.ndarray,
    origin: np.ndarray,
    basis_u: np.ndarray,
    basis_v: np.ndarray,
    selected: tuple[_ArcCandidate, ...],
) -> list[CurvePrimitive]:
    count = len(points)
    occupied = {
        index % count
        for candidate in selected
        for index in candidate.edge_indices
    }
    uncovered = sorted(set(range(count)) - occupied)
    current = (uncovered[0] + 1) % count if uncovered else min(item.start_index for item in selected)
    by_start = {item.start_index % count: item for item in selected}
    primitives: list[CurvePrimitive] = []
    processed = 0
    while processed < count:
        candidate = by_start.get(current)
        if candidate is not None and candidate.edge_count <= count - processed:
            end_index = (current + candidate.edge_count) % count
            center = origin + candidate.center_2d[0] * basis_u + candidate.center_2d[1] * basis_v
            primitives.append(
                CircularArcPrimitive(
                    start=points[current].copy(),
                    center=center,
                    end=points[end_index].copy(),
                    radius=candidate.radius,
                    angle_degrees=candidate.angle_degrees,
                    max_radial_error=candidate.max_radial_error,
                    source_edge_count=candidate.edge_count,
                )
            )
            current = end_index
            processed += candidate.edge_count
        else:
            following = (current + 1) % count
            primitives.append(LinePrimitive(points[current].copy(), points[following].copy()))
            current = following
            processed += 1
    return primitives


def _full_circle_loop(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    origin: np.ndarray,
    basis_u: np.ndarray,
    basis_v: np.ndarray,
    center_2d: np.ndarray,
    radius: float,
    max_error: float,
    angular_steps: np.ndarray,
    config: CurveSimplificationConfig,
) -> SimplifiedCurveLoop:
    unit = (points_2d - center_2d) / np.linalg.norm(points_2d - center_2d, axis=1)[:, None]
    projected_2d = center_2d + radius * unit
    projected_3d = origin + projected_2d[:, 0, None] * basis_u + projected_2d[:, 1, None] * basis_v
    center_3d = origin + center_2d[0] * basis_u + center_2d[1] * basis_v
    maximum_span = radians(config.max_arc_angle_degrees)
    primitives: list[CurvePrimitive] = []
    start = 0
    consumed = 0
    count = len(points_3d)
    while consumed < count:
        edge_count = 0
        span = 0.0
        while consumed + edge_count < count:
            next_span = span + abs(float(angular_steps[consumed + edge_count]))
            if edge_count and next_span > maximum_span:
                break
            span = next_span
            edge_count += 1
        end = (start + edge_count) % count
        primitives.append(
            CircularArcPrimitive(
                start=projected_3d[start].copy(),
                center=center_3d.copy(),
                end=projected_3d[end].copy(),
                radius=radius,
                angle_degrees=float(np.degrees(span)),
                max_radial_error=max_error,
                source_edge_count=edge_count,
            )
        )
        start = end
        consumed += edge_count
    return SimplifiedCurveLoop(tuple(primitives), count)
