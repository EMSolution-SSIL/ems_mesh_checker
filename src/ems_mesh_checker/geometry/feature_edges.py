"""Extract and validate feature-edge curves from selected boundary surfaces.

Property ownership is resolved by :mod:`ems_mesh_checker.boundary` before this
module is called.  PyVista/VTK is deliberately responsible for geometric edge
classification, while this module keeps the selected source mesh and joins the
returned edge segments into deterministic curves for later DXF/Gmsh export.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from typing import Any, Iterable

import numpy as np

from ems_mesh_checker.boundary.result import BoundaryResult, PropertyPair


class FeatureEdgeDependencyError(ImportError):
    """Raised when representative-geometry support lacks PyVista/VTK."""


@dataclass(frozen=True)
class FeatureEdgeConfig:
    """Controls PyVista feature-edge extraction and conservative post-processing.

    ``min_curve_length`` is intentionally zero by default: deleting short
    curves can hide a genuine small geometric feature.  It is an explicit,
    opt-in filter for models where mesh-scale fragments are known to be noise.
    Likewise, ``remove_small_loops`` is disabled unless the caller explicitly
    chooses a ``max_loop_edges`` threshold suitable for the model.
    """

    feature_angle_degrees: float = 30.0
    boundary_edges: bool = True
    feature_edges: bool = True
    non_manifold_edges: bool = True
    manifold_edges: bool = False
    vertex_merge_tolerance: float | None = None
    min_curve_length: float = 0.0
    remove_small_loops: bool = False
    max_loop_edges: int = 10

    def __post_init__(self) -> None:
        if not 0.0 < self.feature_angle_degrees < 180.0:
            raise ValueError("feature_angle_degrees must be strictly between 0 and 180")
        if self.vertex_merge_tolerance is not None and self.vertex_merge_tolerance < 0.0:
            raise ValueError("vertex_merge_tolerance must be non-negative or None")
        if self.min_curve_length < 0.0:
            raise ValueError("min_curve_length must be non-negative")
        if self.max_loop_edges < 3:
            raise ValueError("max_loop_edges must be at least 3")


@dataclass(frozen=True)
class FeatureEdgeCurve:
    """One ordered feature polyline reconstructed from VTK edge segments."""

    points: np.ndarray
    closed: bool
    segment_count: int
    length: float


@dataclass(frozen=True)
class FeatureEdgeResult:
    """PyVista output plus traceable, post-processed feature curves."""

    source_mesh: Any
    surface: Any
    edge_mesh: Any
    curves: tuple[FeatureEdgeCurve, ...]
    diagnostics: tuple[str, ...]
    source_kind: str
    property_id: int | None = None
    property_ids: tuple[int, ...] = ()
    property_pair: PropertyPair | None = None
    patch_index: int | None = None

    @property
    def edge_count(self) -> int:
        """Return the number of unique VTK line segments before curve filtering."""

        return _segment_count(self.edge_mesh)


def meshio_surface_to_polydata(surface_mesh: Any) -> Any:
    """Convert a meshio line/triangle/quad surface mesh to ``pyvista.PolyData``.

    Points are copied to three dimensions for VTK.  The compact meshio surface
    remains on :class:`FeatureEdgeResult` and is the authoritative source of
    Property and element provenance.
    """

    pv = _require_pyvista()
    points = _three_dimensional_points(surface_mesh.points)
    face_cells: list[np.ndarray] = []
    line_cells: list[np.ndarray] = []
    for block in surface_mesh.cells:
        data = np.asarray(block.data, dtype=np.int64)
        if block.type in {"triangle", "quad"}:
            face_cells.extend(data)
        elif block.type == "line":
            line_cells.extend(data)
        else:
            raise ValueError(f"unsupported surface cell type for PyVista: {block.type!r}")

    polydata = pv.PolyData(points, faces=_vtk_cell_array(face_cells))
    if line_cells:
        polydata.lines = _vtk_cell_array(line_cells)
    return polydata


def extract_feature_edges(
    surface_mesh: Any,
    *,
    config: FeatureEdgeConfig | None = None,
    source_kind: str = "surface",
    property_pair: PropertyPair | None = None,
    patch_index: int | None = None,
    property_id: int | None = None,
    property_ids: tuple[int, ...] = (),
) -> FeatureEdgeResult:
    """Extract PyVista feature edges and assemble them into validated curves.

    A 2-D boundary is represented by line cells.  VTK feature-edge filtering is
    a polygonal-surface operation, so those lines are passed through and still
    receive the same deterministic curve assembly and diagnostics.
    """

    settings = config or FeatureEdgeConfig()
    surface = meshio_surface_to_polydata(surface_mesh)
    has_polygons = surface.n_faces > 0
    diagnostics: list[str] = []
    required_rank = 2 if has_polygons else 1
    coordinate_rank = _coordinate_rank(np.asarray(surface.points, dtype=float))
    if coordinate_rank < required_rank:
        diagnostics.append(
            "degenerate_source_coordinates: "
            f"affine_rank={coordinate_rank}, expected_at_least={required_rank}; "
            "feature curves were not generated"
        )
        return FeatureEdgeResult(
            source_mesh=surface_mesh,
            surface=surface,
            edge_mesh=_require_pyvista().PolyData(),
            curves=(),
            diagnostics=tuple(diagnostics),
            source_kind=source_kind,
            property_id=property_id,
            property_ids=property_ids,
            property_pair=property_pair,
            patch_index=patch_index,
        )
    if has_polygons:
        # The selected faces already have a canonical topology orientation.  We
        # do not auto-orient a disconnected interface here: its orientation is
        # owned by BoundaryExtractor's Property-pair contract.
        surface_with_normals = surface.compute_normals(
            cell_normals=True,
            point_normals=False,
            consistent_normals=True,
            auto_orient_normals=False,
            split_vertices=False,
        )
        edge_mesh = surface_with_normals.extract_feature_edges(
            feature_angle=settings.feature_angle_degrees,
            boundary_edges=settings.boundary_edges,
            non_manifold_edges=settings.non_manifold_edges,
            feature_edges=settings.feature_edges,
            manifold_edges=settings.manifold_edges,
            clear_data=False,
        )
    else:
        edge_mesh = surface.copy(deep=True)
        diagnostics.append("line_surface_passthrough: 2-D boundary lines were not sent to VTKFeatureEdges")

    curves, curve_diagnostics = _assemble_curves(edge_mesh, settings)
    diagnostics.extend(curve_diagnostics)
    if not curves:
        diagnostics.append("no_feature_edges: adjust feature_angle_degrees or selected edge categories")
    return FeatureEdgeResult(
        source_mesh=surface_mesh,
        surface=surface,
        edge_mesh=edge_mesh,
        curves=curves,
        diagnostics=tuple(diagnostics),
        source_kind=source_kind,
        property_id=property_id,
        property_ids=property_ids,
        property_pair=property_pair,
        patch_index=patch_index,
    )


def extract_exterior_feature_edges(
    boundaries: BoundaryResult,
    *,
    config: FeatureEdgeConfig | None = None,
) -> FeatureEdgeResult:
    """Extract representative curves for the complete external boundary."""

    return extract_feature_edges(
        boundaries.external_surface,
        config=config,
        source_kind="exterior",
    )


def extract_interface_feature_edges(
    boundaries: BoundaryResult,
    property_a: int,
    property_b: int,
    *,
    config: FeatureEdgeConfig | None = None,
) -> tuple[FeatureEdgeResult, ...]:
    """Extract one result per disconnected interface patch.

    Keeping patches separate prevents a shared Property ID from accidentally
    becoming one representative curve network merely because the user selected
    the same Property pair.
    """

    pair = tuple(sorted((int(property_a), int(property_b))))
    return tuple(
        extract_feature_edges(
            patch_mesh,
            config=config,
            source_kind="interface",
            property_pair=pair,
            patch_index=patch_index,
        )
        for patch_index, patch_mesh in enumerate(boundaries.get_interface_patches(*pair))
    )


def extract_property_feature_edges(
    boundaries: BoundaryResult,
    property_id: int,
    *,
    config: FeatureEdgeConfig | None = None,
) -> FeatureEdgeResult:
    """Extract the complete closed-boundary candidate for one Property.

    External faces owned by the Property and every adjacent interface are
    combined.  Interface orientation is reversed when the selected Property is
    the high-ID neighbor, so source-face normals remain Property-outward.
    """

    selected_property = int(property_id)
    return extract_property_group_feature_edges(
        boundaries,
        (selected_property,),
        config=config,
    )


def extract_property_group_feature_edges(
    boundaries: BoundaryResult,
    property_ids: Iterable[int],
    *,
    config: FeatureEdgeConfig | None = None,
) -> FeatureEdgeResult:
    """Extract the outside boundary of one Property union.

    Interfaces whose two Property IDs are both in ``property_ids`` are omitted
    before the surface is passed to PyVista/VTK.  Interfaces between the union
    and an unselected Property are retained and oriented outward from the
    selected union.  A one-item group is identical to
    :func:`extract_property_feature_edges`.
    """

    from ems_mesh_checker.boundary.result import BoundaryFace
    from ems_mesh_checker.boundary.surface_mesh import build_surface_mesh

    selected_properties = tuple(sorted({int(value) for value in property_ids}))
    if not selected_properties:
        raise ValueError("property_ids must contain at least one Property ID")
    selected_set = set(selected_properties)
    available = _property_ids(boundaries)
    missing = sorted(selected_set - available)
    if missing:
        raise ValueError(f"Property IDs are not present in the mesh: {missing}")

    faces: list[BoundaryFace] = [
        face for face in boundaries.external_faces if face.owner.property_id in selected_set
    ]
    for pair, interface_faces in boundaries.property_interfaces.items():
        selected_sides = selected_set.intersection(pair)
        if len(selected_sides) != 1:
            # No selected owner, or an interface internal to the selected union.
            continue
        selected_property = next(iter(selected_sides))
        for face in interface_faces:
            if face.owner.property_id == selected_property:
                faces.append(face)
            else:
                assert face.neighbor is not None and face.neighbor.property_id == selected_property
                faces.append(
                    BoundaryFace(
                        record=face.record,
                        owner=face.neighbor,
                        neighbor=face.owner,
                        oriented_nodes=tuple(reversed(face.oriented_nodes)),
                    )
                )
    return extract_feature_edges(
        build_surface_mesh(boundaries.mesh, faces),
        config=config,
        source_kind="property",
        property_id=selected_properties[0],
        property_ids=selected_properties,
    )


def extract_all_property_feature_edges(
    boundaries: BoundaryResult,
    *,
    config: FeatureEdgeConfig | None = None,
) -> tuple[FeatureEdgeResult, ...]:
    """Extract one representative-boundary candidate per Property ID."""

    property_ids = {
        int(face.owner.property_id)
        for face in boundaries.external_faces
        if face.owner.property_id is not None
    }
    for property_a, property_b in boundaries.property_interfaces:
        property_ids.update((property_a, property_b))
    return tuple(
        extract_property_feature_edges(boundaries, property_id, config=config)
        for property_id in sorted(property_ids)
    )


def _property_ids(boundaries: BoundaryResult) -> set[int]:
    property_ids = {
        int(face.owner.property_id)
        for face in boundaries.external_faces
        if face.owner.property_id is not None
    }
    for property_a, property_b in boundaries.property_interfaces:
        property_ids.update((property_a, property_b))
    return property_ids


def _require_pyvista() -> Any:
    try:
        import pyvista
    except ImportError as error:  # pragma: no cover - depends on local environment
        raise FeatureEdgeDependencyError(
            "PyVista/VTK is required for feature-edge extraction. "
            "Install the project geometry extra: pip install -e '.[geometry]'"
        ) from error
    return pyvista


def _three_dimensional_points(points: Any) -> np.ndarray:
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[1] not in {2, 3}:
        raise ValueError("surface points must have shape (n, 2) or (n, 3)")
    if array.shape[1] == 3:
        return array
    return np.column_stack((array, np.zeros(len(array), dtype=float)))


def _vtk_cell_array(cells: Iterable[np.ndarray]) -> np.ndarray:
    rows = [np.asarray(cell, dtype=np.int64).reshape(-1) for cell in cells]
    if not rows:
        return np.empty(0, dtype=np.int64)
    return np.concatenate([np.concatenate(([len(row)], row)) for row in rows])


def _assemble_curves(
    edge_mesh: Any,
    config: FeatureEdgeConfig,
) -> tuple[tuple[FeatureEdgeCurve, ...], tuple[str, ...]]:
    """Join VTK line cells, stopping at branches and preserving closed loops."""

    points, segments = _normalized_segments(edge_mesh, config.vertex_merge_tolerance)
    diagnostics: list[str] = []
    if config.remove_small_loops:
        original_segment_count = len(segments)
        segments, removed_loop_count = _remove_small_closed_loops(
            segments,
            max_loop_edges=config.max_loop_edges,
        )
        removed_segment_count = original_segment_count - len(segments)
        if removed_loop_count:
            diagnostics.append(
                "feature_edge_small_loops_removed="
                f"{removed_loop_count}: segments_removed={removed_segment_count}, "
                f"max_loop_edges={config.max_loop_edges}"
            )
    if not segments:
        return (), tuple(diagnostics)
    adjacency: dict[int, set[int]] = defaultdict(set)
    for first, second in segments:
        adjacency[first].add(second)
        adjacency[second].add(first)
    remaining = {tuple(sorted(segment)) for segment in segments}
    paths: list[tuple[tuple[int, ...], bool]] = []

    for start in sorted(node for node, neighbors in adjacency.items() if len(neighbors) != 2):
        for neighbor in sorted(adjacency[start]):
            edge = tuple(sorted((start, neighbor)))
            if edge in remaining:
                paths.append(_walk_path(start, neighbor, adjacency, remaining))

    while remaining:
        first_edge = min(remaining)
        start, neighbor = first_edge
        paths.append(_walk_path(start, neighbor, adjacency, remaining, close_cycle=True))

    curves: list[FeatureEdgeCurve] = []
    dropped = 0
    for point_ids, closed in paths:
        coordinates = points[np.asarray(point_ids, dtype=int)]
        length = _curve_length(coordinates, closed)
        if length < config.min_curve_length:
            dropped += 1
            continue
        curves.append(
            FeatureEdgeCurve(
                points=coordinates,
                closed=closed,
                segment_count=len(point_ids) if closed else len(point_ids) - 1,
                length=length,
            )
        )
    branch_count = sum(1 for neighbors in adjacency.values() if len(neighbors) > 2)
    if branch_count:
        diagnostics.append(f"feature_edge_branch_points={branch_count}: curves split at branches")
    if dropped:
        diagnostics.append(f"feature_edge_curves_dropped={dropped}: below min_curve_length")
    return tuple(curves), tuple(diagnostics)


def _remove_small_closed_loops(
    segments: set[tuple[int, int]],
    *,
    max_loop_edges: int,
) -> tuple[set[tuple[int, int]], int]:
    """Remove edges belonging to graph cycles up to a caller-selected size.

    This deliberately mirrors pyemsi's optional feature-edge cleanup.  It is
    conservative by default because a small closed loop can also be a real
    geometric detail; callers must explicitly enable it in
    :class:`FeatureEdgeConfig`.
    """

    if not segments:
        return set(), 0
    try:
        import networkx as nx
    except ImportError as error:  # pragma: no cover - packaging dependency
        raise FeatureEdgeDependencyError(
            "NetworkX is required when remove_small_loops is enabled"
        ) from error

    graph = nx.Graph()
    graph.add_edges_from(segments)
    edges_to_remove: set[tuple[int, int]] = set()
    removed_loop_count = 0
    for cycle in nx.simple_cycles(graph, length_bound=max_loop_edges):
        if len(cycle) < 3:
            continue
        removed_loop_count += 1
        cycle_nodes = cycle + [cycle[0]]
        for first, second in zip(cycle_nodes[:-1], cycle_nodes[1:]):
            edges_to_remove.add(tuple(sorted((int(first), int(second)))))
    return segments.difference(edges_to_remove), removed_loop_count


def _normalized_segments(edge_mesh: Any, tolerance: float | None) -> tuple[np.ndarray, set[tuple[int, int]]]:
    raw_points = np.asarray(edge_mesh.points, dtype=float)
    if not len(raw_points):
        return raw_points, set()
    merge_tolerance = _default_merge_tolerance(raw_points) if tolerance is None else tolerance
    point_ids = _merge_point_ids(raw_points, merge_tolerance)
    segments: set[tuple[int, int]] = set()
    for line in _line_cells(edge_mesh):
        for first, second in zip(line, line[1:]):
            normalized = tuple(sorted((point_ids[int(first)], point_ids[int(second)])))
            if normalized[0] != normalized[1]:
                segments.add(normalized)
    unique_points = np.empty((max(point_ids) + 1, 3), dtype=float)
    for raw_id, merged_id in enumerate(point_ids):
        unique_points[merged_id] = raw_points[raw_id]
    return unique_points, segments


def _line_cells(edge_mesh: Any) -> Iterable[np.ndarray]:
    data = np.asarray(edge_mesh.lines, dtype=np.int64)
    index = 0
    while index < len(data):
        count = int(data[index])
        index += 1
        if count < 2 or index + count > len(data):
            raise ValueError("invalid VTK line-cell array returned by PyVista")
        yield data[index : index + count]
        index += count


def _merge_point_ids(points: np.ndarray, tolerance: float) -> np.ndarray:
    """Merge equal/nearby VTK points with a deterministic spatial hash."""

    if tolerance == 0.0:
        _, inverse = np.unique(points, axis=0, return_inverse=True)
        return inverse.astype(int)
    buckets: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    representatives: list[np.ndarray] = []
    merged_ids = np.empty(len(points), dtype=int)
    for point_id, point in enumerate(points):
        bucket = tuple(np.floor(point / tolerance).astype(np.int64))
        found: int | None = None
        for offset in product((-1, 0, 1), repeat=3):
            for candidate in buckets.get(tuple(bucket[axis] + offset[axis] for axis in range(3)), ()):
                if np.linalg.norm(point - representatives[candidate]) <= tolerance:
                    found = candidate
                    break
            if found is not None:
                break
        if found is None:
            found = len(representatives)
            representatives.append(point)
            buckets[bucket].append(found)
        merged_ids[point_id] = found
    return merged_ids


def _default_merge_tolerance(points: np.ndarray) -> float:
    extent = float(np.max(np.ptp(points, axis=0))) if len(points) else 0.0
    return max(extent * 1.0e-12, 1.0e-12)


def _coordinate_rank(points: np.ndarray) -> int:
    if len(points) < 2:
        return 0
    centered = points - points[0]
    scale = max(float(np.max(np.abs(centered))), 1.0)
    return int(np.linalg.matrix_rank(centered, tol=scale * 1.0e-12))


def _walk_path(
    start: int,
    current: int,
    adjacency: dict[int, set[int]],
    remaining: set[tuple[int, int]],
    *,
    close_cycle: bool = False,
) -> tuple[tuple[int, ...], bool]:
    path = [start, current]
    remaining.remove(tuple(sorted((start, current))))
    previous = start
    while len(adjacency[current]) == 2:
        next_node = next(node for node in sorted(adjacency[current]) if node != previous)
        edge = tuple(sorted((current, next_node)))
        if edge not in remaining:
            break
        remaining.remove(edge)
        if next_node == start:
            return tuple(path), True
        path.append(next_node)
        previous, current = current, next_node
    return tuple(path), close_cycle and path[-1] == start


def _curve_length(points: np.ndarray, closed: bool) -> float:
    if len(points) < 2:
        return 0.0
    length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    if closed:
        length += float(np.linalg.norm(points[0] - points[-1]))
    return length


def _segment_count(edge_mesh: Any) -> int:
    return sum(max(len(line) - 1, 0) for line in _line_cells(edge_mesh))
