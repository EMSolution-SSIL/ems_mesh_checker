"""Normalize and consolidate approximately coincident circular geometry."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import ceil, cos, pi, radians, sin
from typing import Iterable

import numpy as np

from .curve_simplification import (
    CircularArcPrimitive,
    CurvePrimitive,
    CurveSimplificationConfig,
    LinePrimitive,
)


@dataclass(frozen=True)
class LabeledPrimitive:
    """One representative primitive and all labels that contributed to it."""

    primitive: CurvePrimitive
    labels: frozenset[str]


@dataclass
class _RawCluster:
    arcs: list[CircularArcPrimitive]
    normal: np.ndarray


@dataclass(frozen=True)
class _CircleSupport:
    center: np.ndarray
    radius: float
    normal: np.ndarray
    basis_u: np.ndarray
    basis_v: np.ndarray
    match_tolerance: float
    max_residual: float
    source_arcs: tuple[CircularArcPrimitive, ...]


@dataclass
class _Interval:
    start: float
    end: float
    labels: set[str]
    max_radial_error: float
    source_edge_count: int


class CircularGeometryNormalizer:
    """Give noisy representations of the same circle one canonical support.

    Short arcs fitted independently to rounded mesh nodes can have noticeably
    different centers even though all samples describe one design circle.  The
    normalizer clusters those observations, refits one support circle using all
    endpoints, and snaps only endpoints known to belong to the cluster.
    """

    def __init__(
        self,
        arcs: Iterable[CircularArcPrimitive],
        *,
        point_tolerance: float,
        curve_config: CurveSimplificationConfig | None = None,
    ) -> None:
        self.point_tolerance = float(point_tolerance)
        self.settings = curve_config or CurveSimplificationConfig()
        self.supports = self._build_supports(tuple(arcs))
        self._endpoint_tolerance = max(
            self.point_tolerance,
            self.settings.radial_absolute_tolerance,
        )
        self._endpoint_lookup_tolerance = max(
            (self._endpoint_tolerance, *(support.match_tolerance for support in self.supports))
        )
        self._endpoint_bins: dict[
            tuple[int, int, int],
            list[tuple[np.ndarray, np.ndarray, _CircleSupport]],
        ] = {}
        self._register_endpoints()

    def normalize_primitive(self, primitive: CurvePrimitive) -> CurvePrimitive:
        """Return a primitive whose shared circular endpoints use canonical points."""

        if isinstance(primitive, LinePrimitive):
            return LinePrimitive(
                start=self.snap_point(primitive.start),
                end=self.snap_point(primitive.end),
                source_edge_count=primitive.source_edge_count,
            )
        support = self.support_for(primitive)
        if support is None:
            return primitive
        start = self._project_point(primitive.start, support)
        end = self._project_point(primitive.end, support)
        raw_normal = _arc_normal(primitive)
        if raw_normal is None:
            raw_normal = support.normal
        plane_normal = support.normal if np.dot(raw_normal, support.normal) >= 0.0 else -support.normal
        angle = _directed_angle(start - support.center, end - support.center, plane_normal)
        if angle <= 1.0e-12:
            angle = radians(float(primitive.angle_degrees))
        return CircularArcPrimitive(
            start=start,
            center=support.center.copy(),
            end=end,
            radius=support.radius,
            angle_degrees=float(np.degrees(angle)),
            max_radial_error=max(primitive.max_radial_error, support.max_residual),
            source_edge_count=primitive.source_edge_count,
            plane_normal=plane_normal.copy(),
        )

    def snap_point(self, point: np.ndarray) -> np.ndarray:
        """Snap a point only when it is a known endpoint of a clustered arc."""

        coordinate = np.asarray(point, dtype=float).reshape(3)
        if not self.supports:
            return coordinate.copy()
        best: tuple[float, np.ndarray] | None = None
        for key in self._neighbor_bins(coordinate):
            for original, snapped, support in self._endpoint_bins.get(key, ()):
                projected = self._project_point(coordinate, support)
                circle_error = float(np.linalg.norm(coordinate - projected))
                endpoint_error = float(np.linalg.norm(projected - snapped))
                if max(circle_error, endpoint_error) <= support.match_tolerance:
                    score = circle_error + endpoint_error
                    if best is None or score < best[0]:
                        best = (score, snapped)
        return coordinate.copy() if best is None else best[1].copy()

    def support_for(self, arc: CircularArcPrimitive) -> _CircleSupport | None:
        """Return the closest compatible canonical support for ``arc``."""

        normal = _arc_normal(arc)
        if normal is None:
            return None
        best: tuple[float, _CircleSupport] | None = None
        for support in self.supports:
            alignment = abs(float(np.dot(normal, support.normal)))
            if alignment < cos(radians(1.0)):
                continue
            tolerance = max(support.match_tolerance, self._arc_match_tolerance(arc))
            delta = np.asarray(arc.center, dtype=float) - support.center
            plane_offset = abs(float(np.dot(delta, support.normal)))
            in_plane_offset = float(
                np.linalg.norm(delta - np.dot(delta, support.normal) * support.normal)
            )
            radius_error = abs(float(arc.radius) - support.radius)
            if max(plane_offset, in_plane_offset, radius_error) > tolerance:
                continue
            score = plane_offset + in_plane_offset + radius_error
            if best is None or score < best[0]:
                best = (score, support)
        return None if best is None else best[1]

    def tolerance_for_arc(self, arc: CircularArcPrimitive) -> float:
        """Return the coordinate tolerance of an arc's canonical support."""

        support = self.support_for(arc)
        return self.point_tolerance if support is None else support.match_tolerance

    @property
    def circular_point_tolerance(self) -> float:
        """Largest local lookup tolerance used for points on known circles."""

        return self._endpoint_lookup_tolerance

    def circular_points_coincident(
        self,
        first: np.ndarray,
        second: np.ndarray,
    ) -> bool:
        """Return true for nearby points lying on the same canonical circle."""

        first_coordinate = np.asarray(first, dtype=float).reshape(3)
        second_coordinate = np.asarray(second, dtype=float).reshape(3)
        distance = float(np.linalg.norm(first_coordinate - second_coordinate))
        for support in self.supports:
            if distance > support.match_tolerance:
                continue
            first_error = float(
                np.linalg.norm(first_coordinate - self._project_point(first_coordinate, support))
            )
            second_error = float(
                np.linalg.norm(second_coordinate - self._project_point(second_coordinate, support))
            )
            if max(first_error, second_error) <= support.match_tolerance:
                return True
        return False

    def consolidate_labeled(
        self,
        items: Iterable[LabeledPrimitive],
    ) -> tuple[LabeledPrimitive, ...]:
        """Union overlapping/adjacent arc coverage on every canonical circle."""

        normalized = tuple(
            LabeledPrimitive(self.normalize_primitive(item.primitive), item.labels)
            for item in items
        )
        output = [item for item in normalized if not isinstance(item.primitive, CircularArcPrimitive)]
        by_support: dict[int, list[LabeledPrimitive]] = {}
        unmatched: list[LabeledPrimitive] = []
        for item in normalized:
            if not isinstance(item.primitive, CircularArcPrimitive):
                continue
            support = self.support_for(item.primitive)
            if support is None:
                unmatched.append(item)
                continue
            support_index = next(
                index for index, candidate in enumerate(self.supports) if candidate is support
            )
            by_support.setdefault(support_index, []).append(item)
        output.extend(unmatched)
        for support_index in sorted(by_support):
            output.extend(self._consolidate_support(self.supports[support_index], by_support[support_index]))
        return tuple(output)

    def _build_supports(
        self,
        arcs: tuple[CircularArcPrimitive, ...],
    ) -> tuple[_CircleSupport, ...]:
        raw_clusters: list[_RawCluster] = []
        ordered = sorted(arcs, key=_arc_sort_key)
        for arc in ordered:
            normal = _arc_normal(arc)
            if normal is None:
                continue
            matched: _RawCluster | None = None
            best_score = float("inf")
            for cluster in raw_clusters:
                score = self._raw_match_score(arc, normal, cluster)
                if score is not None and score < best_score:
                    matched = cluster
                    best_score = score
            if matched is None:
                raw_clusters.append(_RawCluster([arc], _canonical_normal(normal)))
            else:
                matched.arcs.append(arc)
                aligned = normal if np.dot(normal, matched.normal) >= 0.0 else -normal
                matched.normal = _canonical_normal(matched.normal + aligned)
        return tuple(self._fit_support(cluster) for cluster in raw_clusters)

    def _raw_match_score(
        self,
        arc: CircularArcPrimitive,
        normal: np.ndarray,
        cluster: _RawCluster,
    ) -> float | None:
        if abs(float(np.dot(normal, cluster.normal))) < cos(radians(1.0)):
            return None
        radii = [float(item.radius) for item in cluster.arcs]
        centers = np.vstack([np.asarray(item.center, dtype=float) for item in cluster.arcs])
        radius = float(np.mean(radii))
        center = np.mean(centers, axis=0)
        tolerance = max(
            self._arc_match_tolerance(arc),
            *(self._arc_match_tolerance(item) for item in cluster.arcs),
        )
        delta = np.asarray(arc.center, dtype=float) - center
        plane_offset = abs(float(np.dot(delta, cluster.normal)))
        in_plane_offset = float(
            np.linalg.norm(delta - np.dot(delta, cluster.normal) * cluster.normal)
        )
        radius_error = abs(float(arc.radius) - radius)
        if max(plane_offset, in_plane_offset, radius_error) > tolerance:
            return None
        return plane_offset + in_plane_offset + radius_error

    def _arc_match_tolerance(self, arc: CircularArcPrimitive) -> float:
        return max(
            self.point_tolerance,
            6.0 * self.settings.radial_absolute_tolerance,
            6.0 * float(arc.radius) * self.settings.radial_relative_tolerance,
            3.0 * float(arc.max_radial_error),
        )

    def _fit_support(self, cluster: _RawCluster) -> _CircleSupport:
        points = _unique_points(
            tuple(
                np.asarray(point, dtype=float)
                for arc in cluster.arcs
                for point in (arc.start, arc.end)
            ),
            self.point_tolerance,
        )
        normal = _canonical_normal(cluster.normal)
        origin = np.mean(points, axis=0)
        basis_u, basis_v = _plane_axes(normal)
        projected = np.column_stack(((points - origin) @ basis_u, (points - origin) @ basis_v))
        if len(points) >= 3 and np.linalg.matrix_rank(projected - np.mean(projected, axis=0)) >= 2:
            matrix = np.column_stack(
                (2.0 * projected[:, 0], 2.0 * projected[:, 1], np.ones(len(projected)))
            )
            target = np.sum(projected**2, axis=1)
            center_x, center_y, constant = np.linalg.lstsq(matrix, target, rcond=None)[0]
            radius_squared = float(constant + center_x * center_x + center_y * center_y)
            center = origin + center_x * basis_u + center_y * basis_v
            radius = float(np.sqrt(max(radius_squared, 0.0)))
        else:
            center = np.mean(
                np.vstack([np.asarray(arc.center, dtype=float) for arc in cluster.arcs]), axis=0
            )
            radius = float(np.mean([arc.radius for arc in cluster.arcs]))
        distances = np.linalg.norm(points - center, axis=1)
        residual = float(np.max(np.abs(distances - radius)))
        tolerance = max(self._arc_match_tolerance(arc) for arc in cluster.arcs)
        return _CircleSupport(
            center=np.asarray(center, dtype=float),
            radius=radius,
            normal=normal,
            basis_u=basis_u,
            basis_v=basis_v,
            match_tolerance=tolerance,
            max_residual=residual,
            source_arcs=tuple(cluster.arcs),
        )

    def _register_endpoints(self) -> None:
        for support in self.supports:
            for arc in support.source_arcs:
                for point in (arc.start, arc.end):
                    original = np.asarray(point, dtype=float).reshape(3)
                    snapped = self._project_point(original, support)
                    key = self._point_bin(original)
                    entries = self._endpoint_bins.setdefault(key, [])
                    existing = next(
                        (
                            index
                            for index, (item, _, item_support) in enumerate(entries)
                            if np.linalg.norm(item - original)
                            <= min(item_support.match_tolerance, support.match_tolerance)
                        ),
                        None,
                    )
                    if existing is None:
                        entries.append((original.copy(), snapped, support))
                    else:
                        old_original, old_snapped, old_support = entries[existing]
                        if np.linalg.norm(snapped - original) < np.linalg.norm(old_snapped - old_original):
                            entries[existing] = (original.copy(), snapped, old_support)

    def _project_point(self, point: np.ndarray, support: _CircleSupport) -> np.ndarray:
        coordinate = np.asarray(point, dtype=float).reshape(3)
        planar = coordinate - np.dot(coordinate - support.center, support.normal) * support.normal
        radial = planar - support.center
        length = float(np.linalg.norm(radial))
        if length <= 0.0:
            return coordinate.copy()
        return support.center + support.radius * radial / length

    def _consolidate_support(
        self,
        support: _CircleSupport,
        items: list[LabeledPrimitive],
    ) -> tuple[LabeledPrimitive, ...]:
        intervals: list[_Interval] = []
        for item in items:
            arc = item.primitive
            assert isinstance(arc, CircularArcPrimitive)
            start_angle = _point_angle(arc.start, support)
            end_angle = _point_angle(arc.end, support)
            plane_normal = arc.plane_normal if arc.plane_normal is not None else support.normal
            if np.dot(plane_normal, support.normal) < 0.0:
                start_angle, end_angle = end_angle, start_angle
            span = radians(float(arc.angle_degrees))
            if span <= 0.0:
                continue
            while end_angle < start_angle:
                end_angle += 2.0 * pi
            if abs((end_angle - start_angle) - span) > radians(0.1):
                end_angle = start_angle + span
            if end_angle <= 2.0 * pi + 1.0e-12:
                intervals.append(
                    _Interval(start_angle, min(end_angle, 2.0 * pi), set(item.labels), arc.max_radial_error, arc.source_edge_count)
                )
            else:
                intervals.append(
                    _Interval(start_angle, 2.0 * pi, set(item.labels), arc.max_radial_error, arc.source_edge_count)
                )
                intervals.append(
                    _Interval(0.0, end_angle - 2.0 * pi, set(item.labels), arc.max_radial_error, arc.source_edge_count)
                )
        angle_tolerance = max(1.0e-10, support.match_tolerance / max(support.radius, 1.0e-30))
        merged = _merge_intervals(intervals, angle_tolerance)
        if len(merged) > 1 and merged[0].start <= angle_tolerance and merged[-1].end >= 2.0 * pi - angle_tolerance:
            first, last = merged[0], merged[-1]
            wrapped = _Interval(
                last.start,
                first.end + 2.0 * pi,
                last.labels | first.labels,
                max(last.max_radial_error, first.max_radial_error),
                last.source_edge_count + first.source_edge_count,
            )
            merged = [wrapped, *merged[1:-1]]

        output: list[LabeledPrimitive] = []
        for interval in sorted(merged, key=lambda item: item.start):
            span = interval.end - interval.start
            piece_count = max(1, int(ceil(span / pi - 1.0e-12)))
            piece_span = span / piece_count
            for piece_index in range(piece_count):
                start_angle = interval.start + piece_index * piece_span
                end_angle = interval.start + (piece_index + 1) * piece_span
                primitive = CircularArcPrimitive(
                    start=_circle_point(support, start_angle),
                    center=support.center.copy(),
                    end=_circle_point(support, end_angle),
                    radius=support.radius,
                    angle_degrees=float(np.degrees(piece_span)),
                    max_radial_error=max(interval.max_radial_error, support.max_residual),
                    source_edge_count=max(1, interval.source_edge_count),
                    plane_normal=support.normal.copy(),
                )
                output.append(LabeledPrimitive(primitive, frozenset(interval.labels)))
        return tuple(output)

    def _point_bin(self, coordinate: np.ndarray) -> tuple[int, int, int]:
        if self._endpoint_lookup_tolerance <= 0.0:
            return (0, 0, 0)
        return tuple(
            int(np.floor(float(value) / self._endpoint_lookup_tolerance)) for value in coordinate
        )

    def _neighbor_bins(self, coordinate: np.ndarray) -> Iterable[tuple[int, int, int]]:
        key = self._point_bin(coordinate)
        if self._endpoint_lookup_tolerance <= 0.0:
            return (key,)
        return (
            (key[0] + dx, key[1] + dy, key[2] + dz)
            for dx, dy, dz in product((-1, 0, 1), repeat=3)
        )


def _arc_normal(arc: CircularArcPrimitive) -> np.ndarray | None:
    if arc.plane_normal is not None:
        normal = np.asarray(arc.plane_normal, dtype=float).reshape(3)
    else:
        normal = np.cross(
            np.asarray(arc.start, dtype=float) - arc.center,
            np.asarray(arc.end, dtype=float) - arc.center,
        )
    length = float(np.linalg.norm(normal))
    return None if length <= 1.0e-30 else normal / length


def _canonical_normal(normal: np.ndarray) -> np.ndarray:
    unit = np.asarray(normal, dtype=float).reshape(3)
    unit /= np.linalg.norm(unit)
    index = int(np.argmax(np.abs(unit)))
    return -unit if unit[index] < 0.0 else unit


def _plane_axes(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = np.eye(3)[int(np.argmin(np.abs(normal)))]
    basis_u = np.cross(axis, normal)
    basis_u /= np.linalg.norm(basis_u)
    basis_v = np.cross(normal, basis_u)
    return basis_u, basis_v


def _unique_points(points: tuple[np.ndarray, ...], tolerance: float) -> np.ndarray:
    unique: list[np.ndarray] = []
    for point in points:
        if not any(np.linalg.norm(point - existing) <= tolerance for existing in unique):
            unique.append(point)
    return np.vstack(unique)


def _directed_angle(first: np.ndarray, second: np.ndarray, normal: np.ndarray) -> float:
    return float(
        np.arctan2(
            np.dot(normal, np.cross(first, second)),
            np.dot(first, second),
        )
        % (2.0 * pi)
    )


def _point_angle(point: np.ndarray, support: _CircleSupport) -> float:
    vector = np.asarray(point, dtype=float) - support.center
    return float(
        np.arctan2(np.dot(vector, support.basis_v), np.dot(vector, support.basis_u))
        % (2.0 * pi)
    )


def _circle_point(support: _CircleSupport, angle: float) -> np.ndarray:
    return support.center + support.radius * (
        cos(angle) * support.basis_u + sin(angle) * support.basis_v
    )


def _merge_intervals(intervals: list[_Interval], tolerance: float) -> list[_Interval]:
    merged: list[_Interval] = []
    for item in sorted(intervals, key=lambda value: (value.start, value.end)):
        if not merged or item.start > merged[-1].end + tolerance:
            merged.append(item)
            continue
        previous = merged[-1]
        previous.end = max(previous.end, item.end)
        previous.labels.update(item.labels)
        previous.max_radial_error = max(previous.max_radial_error, item.max_radial_error)
        previous.source_edge_count += item.source_edge_count
    return merged


def _arc_sort_key(arc: CircularArcPrimitive) -> tuple[float, ...]:
    center = np.asarray(arc.center, dtype=float).reshape(3)
    start = np.asarray(arc.start, dtype=float).reshape(3)
    end = np.asarray(arc.end, dtype=float).reshape(3)
    return (*center, float(arc.radius), *start, *end)
