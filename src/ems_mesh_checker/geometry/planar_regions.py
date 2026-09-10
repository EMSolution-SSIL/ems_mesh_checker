"""Build validated planar regions from Property-aware feature-edge results."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from itertools import product
from math import cos, radians
from typing import Any, Iterable, Mapping

import numpy as np

from .feature_edges import FeatureEdgeResult


@dataclass(frozen=True)
class PlanarRegionConfig:
    """Controls conservative face merging and representative-loop creation."""

    coplanar_angle_degrees: float = 1.0
    planarity_absolute_tolerance: float = 1.0e-10
    planarity_relative_tolerance: float = 1.0e-8
    vertex_match_tolerance: float | None = None
    simplify_angle_degrees: float = 0.1
    simplify_absolute_tolerance: float = 1.0e-10
    simplify_relative_tolerance: float = 1.0e-8
    validate_pyvista_feature_edges: bool = True
    preserve_neighbor_property_seams: bool = True
    remove_small_open_components: bool = False
    max_open_component_edges: int = 10

    def __post_init__(self) -> None:
        if not 0.0 <= self.coplanar_angle_degrees < 180.0:
            raise ValueError("coplanar_angle_degrees must be in [0, 180)")
        if not 0.0 <= self.simplify_angle_degrees < 180.0:
            raise ValueError("simplify_angle_degrees must be in [0, 180)")
        for name in (
            "planarity_absolute_tolerance",
            "planarity_relative_tolerance",
            "simplify_absolute_tolerance",
            "simplify_relative_tolerance",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.vertex_match_tolerance is not None and self.vertex_match_tolerance < 0.0:
            raise ValueError("vertex_match_tolerance must be non-negative or None")
        if self.max_open_component_edges < 3:
            raise ValueError("max_open_component_edges must be at least 3")


@dataclass(frozen=True)
class PlanarLoop:
    """One simplified closed loop; holes have negative signed area."""

    points: np.ndarray
    signed_area: float
    is_hole: bool


@dataclass(frozen=True)
class PlanarRegion:
    """One representative plane built from one connected face component."""

    region_index: int
    source_face_indices: tuple[int, ...]
    origin: np.ndarray
    normal: np.ndarray
    outer_loop: PlanarLoop
    hole_loops: tuple[PlanarLoop, ...]
    max_planarity_error: float
    rms_planarity_error: float


@dataclass(frozen=True)
class UnsupportedRegion:
    """A source component that was deliberately not converted to a plane."""

    source_face_indices: tuple[int, ...]
    reason: str


@dataclass(frozen=True)
class PlanarRegionResult:
    """Representative planes and auditable failures for one feature result."""

    feature_edges: FeatureEdgeResult
    regions: tuple[PlanarRegion, ...]
    unsupported_regions: tuple[UnsupportedRegion, ...]
    diagnostics: tuple[str, ...]


def extract_planar_regions(
    feature_edges: FeatureEdgeResult,
    *,
    config: PlanarRegionConfig | None = None,
) -> PlanarRegionResult:
    """Merge coplanar source faces and recover their outer/hole loops.

    PyVista feature edges are checked against source-face normals by default.
    This suppresses false-positive edges on truly coplanar faces, while an
    independent angle check prevents a missed edge from joining a real corner.
    """

    settings = config or PlanarRegionConfig()
    points = _points3d(feature_edges.source_mesh.points)
    faces = _surface_faces(feature_edges.source_mesh)
    neighbor_properties = _surface_face_data(feature_edges.source_mesh, "neighbor_property_id")
    if not faces:
        return _regions_from_closed_curves(feature_edges, settings)

    normals: dict[int, np.ndarray] = {}
    unsupported: list[UnsupportedRegion] = []
    for face_index, face in enumerate(faces):
        normal = _face_normal(points[np.asarray(face, dtype=int)])
        if normal is None:
            unsupported.append(UnsupportedRegion((face_index,), "degenerate_source_face"))
        else:
            normals[face_index] = normal

    edge_owners: dict[tuple[int, int], list[int]] = defaultdict(list)
    for face_index, face in enumerate(faces):
        if face_index not in normals:
            continue
        for edge in _face_edges(face):
            edge_owners[edge].append(face_index)

    match_tolerance = (
        _scale_tolerance(points, 1.0e-10, 1.0e-12)
        if settings.vertex_match_tolerance is None
        else settings.vertex_match_tolerance
    )
    feature_barriers, unmatched_segments = _feature_barrier_edges(
        feature_edges.edge_mesh, points, match_tolerance
    )
    diagnostics: list[str] = []
    if unmatched_segments:
        diagnostics.append(
            f"unmatched_pyvista_segments={unmatched_segments}: source surface retained for inspection"
        )

    adjacency: dict[int, set[int]] = {face_index: set() for face_index in normals}
    ignored_coplanar_features = 0
    for edge, owners in sorted(edge_owners.items()):
        if len(owners) > 2:
            diagnostics.append(f"nonmanifold_source_edge={edge}: owner_count={len(owners)}")
            continue
        if len(owners) != 2:
            continue
        first, second = owners
        if (
            settings.preserve_neighbor_property_seams
            and neighbor_properties is not None
            and neighbor_properties[first] != neighbor_properties[second]
        ):
            continue
        angle = _normal_angle_degrees(normals[first], normals[second])
        pyvista_barrier = edge in feature_barriers
        if angle > settings.coplanar_angle_degrees:
            continue
        if pyvista_barrier and not settings.validate_pyvista_feature_edges:
            continue
        if pyvista_barrier:
            ignored_coplanar_features += 1
        adjacency[first].add(second)
        adjacency[second].add(first)
    if ignored_coplanar_features:
        diagnostics.append(
            f"ignored_coplanar_pyvista_edges={ignored_coplanar_features}: validated against source normals"
        )

    regions: list[PlanarRegion] = []
    for component in _connected_face_components(adjacency):
        region, failure = _component_to_region(len(regions), component, faces, points, settings)
        if region is not None:
            regions.append(region)
        else:
            assert failure is not None
            unsupported.append(failure)

    if settings.remove_small_open_components:
        regions, removal_diagnostic = _remove_small_open_surface_components(
            regions,
            faces,
            points,
            settings,
        )
        if removal_diagnostic is not None:
            diagnostics.append(removal_diagnostic)

    return PlanarRegionResult(
        feature_edges=feature_edges,
        regions=tuple(regions),
        unsupported_regions=tuple(unsupported),
        diagnostics=tuple(diagnostics),
    )


def _regions_from_closed_curves(
    feature_edges: FeatureEdgeResult,
    config: PlanarRegionConfig,
) -> PlanarRegionResult:
    regions: list[PlanarRegion] = []
    unsupported: list[UnsupportedRegion] = []
    for curve_index, curve in enumerate(feature_edges.curves):
        if not curve.closed:
            unsupported.append(UnsupportedRegion((), f"open_feature_curve={curve_index}"))
            continue
        fitted = _fit_plane(np.asarray(curve.points, dtype=float), expected_normal=None)
        if fitted is None:
            unsupported.append(UnsupportedRegion((), f"nonplanar_or_degenerate_curve={curve_index}"))
            continue
        origin, normal, max_error, rms_error = fitted
        tolerance = _planarity_tolerance(curve.points, config)
        if max_error > tolerance:
            unsupported.append(
                UnsupportedRegion((), f"curve_planarity_error={max_error:.6g}>tolerance={tolerance:.6g}")
            )
            continue
        simplified = _simplify_closed_loop(np.asarray(curve.points), config)
        oriented = _oriented_loops((simplified,), origin, normal)
        if oriented is None:
            unsupported.append(UnsupportedRegion((), f"invalid_closed_curve={curve_index}"))
            continue
        outer, holes = oriented
        regions.append(
            PlanarRegion(
                region_index=len(regions),
                source_face_indices=(),
                origin=origin,
                normal=normal,
                outer_loop=outer,
                hole_loops=holes,
                max_planarity_error=max_error,
                rms_planarity_error=rms_error,
            )
        )
    return PlanarRegionResult(
        feature_edges=feature_edges,
        regions=tuple(regions),
        unsupported_regions=tuple(unsupported),
        diagnostics=(),
    )


def _component_to_region(
    region_index: int,
    component: tuple[int, ...],
    faces: list[tuple[int, ...]],
    points: np.ndarray,
    config: PlanarRegionConfig,
) -> tuple[PlanarRegion | None, UnsupportedRegion | None]:
    node_ids = sorted({node for face_index in component for node in faces[face_index]})
    component_normals = [
        normal
        for index in component
        if (normal := _face_normal(points[np.asarray(faces[index], dtype=int)])) is not None
    ]
    expected = np.sum(component_normals, axis=0)
    fitted = _fit_plane(points[np.asarray(node_ids, dtype=int)], expected_normal=expected)
    if fitted is None:
        return None, UnsupportedRegion(component, "component_is_not_a_valid_plane")
    origin, normal, max_error, rms_error = fitted
    tolerance = _planarity_tolerance(points[np.asarray(node_ids, dtype=int)], config)
    if max_error > tolerance:
        return None, UnsupportedRegion(
            component, f"planarity_error={max_error:.6g}>tolerance={tolerance:.6g}"
        )

    loops, loop_error = _component_boundary_loops(component, faces)
    if loop_error is not None:
        return None, UnsupportedRegion(component, loop_error)
    simplified = tuple(_simplify_closed_loop(points[np.asarray(loop, dtype=int)], config) for loop in loops)
    oriented = _oriented_loops(simplified, origin, normal)
    if oriented is None:
        return None, UnsupportedRegion(component, "invalid_outer_or_hole_loops")
    outer, holes = oriented
    return (
        PlanarRegion(
            region_index=region_index,
            source_face_indices=component,
            origin=origin,
            normal=normal,
            outer_loop=outer,
            hole_loops=holes,
            max_planarity_error=max_error,
            rms_planarity_error=rms_error,
        ),
        None,
    )


def _surface_faces(mesh: Any) -> list[tuple[int, ...]]:
    faces: list[tuple[int, ...]] = []
    for block in mesh.cells:
        if block.type in {"triangle", "quad"}:
            faces.extend(tuple(int(node) for node in row) for row in np.asarray(block.data))
    return faces


def _surface_face_data(mesh: Any, name: str) -> np.ndarray | None:
    """Return cell data flattened in the same order as ``_surface_faces``."""

    blocks = getattr(mesh, "cell_data", {}).get(name)
    if blocks is None or len(blocks) != len(mesh.cells):
        return None
    values: list[np.ndarray] = []
    for block, data in zip(mesh.cells, blocks):
        if block.type not in {"triangle", "quad"}:
            continue
        array = np.asarray(data).reshape(-1)
        if len(array) != len(block.data):
            return None
        values.append(array)
    return np.concatenate(values) if values else np.empty(0, dtype=int)


def _points3d(points: Any) -> np.ndarray:
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[1] not in {2, 3}:
        raise ValueError("surface points must have shape (n, 2) or (n, 3)")
    return np.column_stack((array, np.zeros(len(array)))) if array.shape[1] == 2 else array


def _face_edges(face: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    return tuple(
        tuple(sorted((int(face[index]), int(face[(index + 1) % len(face)]))))
        for index in range(len(face))
    )


def _face_normal(face_points: np.ndarray) -> np.ndarray | None:
    normal = np.zeros(3, dtype=float)
    for index, current in enumerate(face_points):
        normal += np.cross(current, face_points[(index + 1) % len(face_points)])
    length = float(np.linalg.norm(normal))
    scale = max(float(np.max(np.ptp(face_points, axis=0))), 1.0)
    return None if length <= scale * scale * 1.0e-14 else normal / length


def _normal_angle_degrees(first: np.ndarray, second: np.ndarray) -> float:
    cosine = float(np.clip(np.dot(first, second), -1.0, 1.0))
    return float(np.degrees(np.arccos(abs(cosine))))


def _connected_face_components(adjacency: Mapping[int, set[int]]) -> tuple[tuple[int, ...], ...]:
    visited: set[int] = set()
    components: list[tuple[int, ...]] = []
    for start in sorted(adjacency):
        if start in visited:
            continue
        queue: deque[int] = deque([start])
        visited.add(start)
        component: list[int] = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in sorted(adjacency[current]):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        components.append(tuple(sorted(component)))
    return tuple(components)


def _remove_small_open_surface_components(
    regions: list[PlanarRegion],
    source_faces: list[tuple[int, ...]],
    source_points: np.ndarray,
    config: PlanarRegionConfig,
) -> tuple[list[PlanarRegion], str | None]:
    """Drop isolated, hole-free planar patches with a small outer boundary.

    A single planar region cannot form a closed three-dimensional shell.  The
    filter therefore removes only one-region surface components and preserves
    every component whose regions share at least one complete boundary edge.
    Two-dimensional closed-curve regions have no source-face provenance and
    are deliberately outside the scope of this cleanup.
    """

    if not regions:
        return regions, None
    tolerance = (
        _scale_tolerance(source_points, 1.0e-8, 1.0e-10)
        if config.vertex_match_tolerance is None
        else config.vertex_match_tolerance
    )
    lookup = _PointLookup(source_points, tolerance)
    adjacency: dict[int, set[int]] = {index: set() for index in range(len(regions))}
    edge_owners: dict[tuple[int, int], set[int]] = defaultdict(set)
    region_edges: list[set[tuple[int, int]]] = []
    unmatched_regions: set[int] = set()
    for region_index, region in enumerate(regions):
        edges: set[tuple[int, int]] = set()
        for loop in (region.outer_loop, *region.hole_loops):
            loop_points = np.asarray(loop.points, dtype=float)
            for index, first in enumerate(loop_points):
                second = loop_points[(index + 1) % len(loop_points)]
                first_id = lookup.find(first)
                second_id = lookup.find(second)
                if first_id is None or second_id is None:
                    unmatched_regions.add(region_index)
                    continue
                if first_id != second_id:
                    edges.add(tuple(sorted((first_id, second_id))))
        region_edges.append(edges)
        for edge in edges:
            edge_owners[edge].add(region_index)

    for owners in edge_owners.values():
        ordered = sorted(owners)
        for index, first in enumerate(ordered):
            for second in ordered[index + 1 :]:
                adjacency[first].add(second)
                adjacency[second].add(first)

    face_to_region = {
        face_index: region_index
        for region_index, region in enumerate(regions)
        for face_index in region.source_face_indices
    }
    source_edge_owners: dict[tuple[int, int], set[int]] = defaultdict(set)
    for face_index, region_index in face_to_region.items():
        for edge in _face_edges(source_faces[face_index]):
            source_edge_owners[edge].add(region_index)
    for owners in source_edge_owners.values():
        ordered = sorted(owners)
        for index, first in enumerate(ordered):
            for second in ordered[index + 1 :]:
                adjacency[first].add(second)
                adjacency[second].add(first)

    removed: set[int] = set()
    removed_boundary_edges = 0
    removed_source_faces = 0
    for component in _connected_face_components(adjacency):
        if len(component) != 1:
            continue
        region_index = component[0]
        region = regions[region_index]
        boundary_edge_count = len(region.outer_loop.points)
        if (
            region_index in unmatched_regions
            or not region.source_face_indices
            or region.hole_loops
            or boundary_edge_count < 3
            or boundary_edge_count > config.max_open_component_edges
        ):
            continue
        removed.add(region_index)
        removed_boundary_edges += boundary_edge_count
        removed_source_faces += len(region.source_face_indices)

    if not removed:
        return regions, None
    kept = [region for index, region in enumerate(regions) if index not in removed]
    diagnostic = (
        f"small_open_surface_components_removed={len(removed)}: "
        f"boundary_edges={removed_boundary_edges}, "
        f"source_faces={removed_source_faces}, "
        f"max_open_component_edges={config.max_open_component_edges}"
    )
    return kept, diagnostic


def _component_boundary_loops(
    component: tuple[int, ...],
    faces: list[tuple[int, ...]],
) -> tuple[tuple[tuple[int, ...], ...], str | None]:
    counts: Counter[tuple[int, int]] = Counter()
    for face_index in component:
        counts.update(_face_edges(faces[face_index]))
    boundary_edges = {edge for edge, count in counts.items() if count == 1}
    if not boundary_edges:
        return (), "component_has_no_boundary_loop"
    graph: dict[int, set[int]] = defaultdict(set)
    for first, second in boundary_edges:
        graph[first].add(second)
        graph[second].add(first)
    invalid = sorted((node, len(neighbors)) for node, neighbors in graph.items() if len(neighbors) != 2)
    if invalid:
        return (), f"boundary_loop_degree_error={invalid[:4]}"

    remaining = set(boundary_edges)
    loops: list[tuple[int, ...]] = []
    while remaining:
        start, neighbor = min(remaining)
        path = [start]
        previous, current = start, neighbor
        remaining.remove(tuple(sorted((previous, current))))
        while current != start:
            path.append(current)
            candidates = sorted(node for node in graph[current] if node != previous)
            if len(candidates) != 1:
                return (), "boundary_loop_traversal_failed"
            next_node = candidates[0]
            edge = tuple(sorted((current, next_node)))
            if edge not in remaining:
                return (), "boundary_loop_closed_on_used_edge"
            remaining.remove(edge)
            previous, current = current, next_node
        loops.append(tuple(path))
    return tuple(loops), None


def _fit_plane(
    points: np.ndarray,
    *,
    expected_normal: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    if len(points) < 3:
        return None
    origin = np.mean(points, axis=0)
    centered = points - origin
    _, singular_values, vectors = np.linalg.svd(centered, full_matrices=False)
    scale = max(float(singular_values[0]), 1.0) if len(singular_values) else 1.0
    if int(np.sum(singular_values > scale * 1.0e-12)) < 2:
        return None
    normal = vectors[-1]
    if expected_normal is not None and np.dot(normal, expected_normal) < 0.0:
        normal = -normal
    distances = np.abs(centered @ normal)
    return origin, normal, float(np.max(distances)), float(np.sqrt(np.mean(distances**2)))


def _planarity_tolerance(points: np.ndarray, config: PlanarRegionConfig) -> float:
    diagonal = float(np.linalg.norm(np.ptp(np.asarray(points), axis=0)))
    return max(config.planarity_absolute_tolerance, diagonal * config.planarity_relative_tolerance)


def _simplify_closed_loop(points: np.ndarray, config: PlanarRegionConfig) -> np.ndarray:
    simplified = [np.asarray(point, dtype=float) for point in points]
    if len(simplified) <= 3:
        return np.asarray(simplified)
    tolerance = max(
        config.simplify_absolute_tolerance,
        float(np.linalg.norm(np.ptp(points, axis=0))) * config.simplify_relative_tolerance,
    )
    cosine_limit = cos(radians(config.simplify_angle_degrees))
    changed = True
    while changed and len(simplified) > 3:
        changed = False
        for index in range(len(simplified)):
            previous = simplified[index - 1]
            current = simplified[index]
            following = simplified[(index + 1) % len(simplified)]
            incoming = current - previous
            outgoing = following - current
            incoming_length = float(np.linalg.norm(incoming))
            outgoing_length = float(np.linalg.norm(outgoing))
            if incoming_length == 0.0 or outgoing_length == 0.0:
                del simplified[index]
                changed = True
                break
            cosine_value = float(np.dot(incoming, outgoing) / (incoming_length * outgoing_length))
            line = following - previous
            line_length = float(np.linalg.norm(line))
            distance = (
                float(np.linalg.norm(np.cross(current - previous, line)) / line_length)
                if line_length > 0.0
                else float("inf")
            )
            if cosine_value >= cosine_limit and distance <= tolerance:
                del simplified[index]
                changed = True
                break
    return np.asarray(simplified)


def _oriented_loops(
    loops: Iterable[np.ndarray],
    origin: np.ndarray,
    normal: np.ndarray,
) -> tuple[PlanarLoop, tuple[PlanarLoop, ...]] | None:
    loop_arrays = [np.asarray(loop, dtype=float) for loop in loops if len(loop) >= 3]
    if not loop_arrays:
        return None
    axis = np.eye(3)[int(np.argmin(np.abs(normal)))]
    basis_u = np.cross(axis, normal)
    basis_u /= np.linalg.norm(basis_u)
    basis_v = np.cross(normal, basis_u)
    projected = [np.column_stack(((loop - origin) @ basis_u, (loop - origin) @ basis_v)) for loop in loop_arrays]
    areas = [_signed_area(loop) for loop in projected]
    outer_index = int(np.argmax(np.abs(areas)))
    if abs(areas[outer_index]) <= 1.0e-20:
        return None
    outer_2d = projected[outer_index]
    for index, loop_2d in enumerate(projected):
        if index != outer_index and not _point_in_polygon(loop_2d[0], outer_2d):
            return None

    outer_points = loop_arrays[outer_index]
    outer_area = areas[outer_index]
    if outer_area < 0.0:
        outer_points = outer_points[::-1]
        outer_area = -outer_area
    holes: list[PlanarLoop] = []
    for index in sorted(
        (idx for idx in range(len(loop_arrays)) if idx != outer_index),
        key=lambda idx: -abs(areas[idx]),
    ):
        hole_points = loop_arrays[index]
        hole_area = areas[index]
        if hole_area > 0.0:
            hole_points = hole_points[::-1]
            hole_area = -hole_area
        holes.append(PlanarLoop(hole_points, float(hole_area), True))
    return PlanarLoop(outer_points, float(outer_area), False), tuple(holes)


def _signed_area(points: np.ndarray) -> float:
    return 0.5 * float(
        np.sum(points[:, 0] * np.roll(points[:, 1], -1) - np.roll(points[:, 0], -1) * points[:, 1])
    )


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    inside = False
    x, y = float(point[0]), float(point[1])
    for index, first in enumerate(polygon):
        second = polygon[(index + 1) % len(polygon)]
        y1, y2 = float(first[1]), float(second[1])
        if (y1 > y) != (y2 > y):
            intersection = (float(second[0]) - float(first[0])) * (y - y1) / (y2 - y1) + float(first[0])
            if x < intersection:
                inside = not inside
    return inside


def _feature_barrier_edges(
    edge_mesh: Any,
    source_points: np.ndarray,
    tolerance: float,
) -> tuple[set[tuple[int, int]], int]:
    lookup = _PointLookup(source_points, tolerance)
    mapped: set[tuple[int, int]] = set()
    unmatched = 0
    for first, second in _polydata_segments(edge_mesh):
        first_id = lookup.find(np.asarray(edge_mesh.points[first], dtype=float))
        second_id = lookup.find(np.asarray(edge_mesh.points[second], dtype=float))
        if first_id is None or second_id is None:
            unmatched += 1
        elif first_id != second_id:
            mapped.add(tuple(sorted((first_id, second_id))))
    return mapped, unmatched


def _polydata_segments(edge_mesh: Any) -> Iterable[tuple[int, int]]:
    lines = np.asarray(edge_mesh.lines, dtype=np.int64)
    index = 0
    while index < len(lines):
        count = int(lines[index])
        index += 1
        cell = lines[index : index + count]
        index += count
        for first, second in zip(cell, cell[1:]):
            yield int(first), int(second)


class _PointLookup:
    def __init__(self, points: np.ndarray, tolerance: float) -> None:
        self.points = points
        self.tolerance = tolerance
        self.buckets: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        if tolerance > 0.0:
            for point_id, point in enumerate(points):
                self.buckets[self._bucket(point)].append(point_id)

    def _bucket(self, point: np.ndarray) -> tuple[int, int, int]:
        return tuple(np.floor(point / self.tolerance).astype(np.int64))

    def find(self, point: np.ndarray) -> int | None:
        if self.tolerance == 0.0:
            matches = np.flatnonzero(np.all(self.points == point, axis=1))
            return int(matches[0]) if len(matches) else None
        bucket = self._bucket(point)
        candidates: list[int] = []
        for offset in product((-1, 0, 1), repeat=3):
            candidates.extend(self.buckets.get(tuple(bucket[axis] + offset[axis] for axis in range(3)), ()))
        matches = [
            point_id
            for point_id in candidates
            if np.linalg.norm(self.points[point_id] - point) <= self.tolerance
        ]
        return min(matches) if matches else None


def _scale_tolerance(points: np.ndarray, relative: float, absolute: float) -> float:
    extent = float(np.max(np.ptp(points, axis=0))) if len(points) else 0.0
    return max(extent * relative, absolute)
