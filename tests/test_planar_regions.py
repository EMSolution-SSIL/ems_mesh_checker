from __future__ import annotations

import numpy as np
import pytest

import meshio

from ems_mesh_checker import (
    BoundaryExtractor,
    FeatureEdgeConfig,
    PlanarRegionConfig,
    extract_exterior_feature_edges,
    extract_feature_edges,
    extract_planar_regions,
)


pytest.importorskip("pyvista")


def _cube_mesh() -> meshio.Mesh:
    return meshio.Mesh(
        points=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ]
        ),
        cells=[("hexahedron", np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=int))],
        cell_data={"property_id": [np.array([1], dtype=int)]},
    )


def _cube_surface_with_detached_triangle() -> meshio.Mesh:
    return meshio.Mesh(
        points=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
                [3.0, 0.0, 0.0],
                [3.0, 1.0, 0.0],
                [3.0, 0.0, 1.0],
            ]
        ),
        cells=[
            (
                "quad",
                np.array(
                    [
                        [0, 3, 2, 1],
                        [4, 5, 6, 7],
                        [0, 1, 5, 4],
                        [1, 2, 6, 5],
                        [2, 3, 7, 6],
                        [3, 0, 4, 7],
                    ],
                    dtype=int,
                ),
            ),
            ("triangle", np.array([[8, 9, 10]], dtype=int)),
        ],
    )


def test_two_coplanar_quads_merge_and_collinear_boundary_nodes_are_removed() -> None:
    surface = meshio.Mesh(
        points=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [2.0, 1.0, 0.0],
            ]
        ),
        cells=[("quad", np.array([[0, 1, 4, 3], [1, 2, 5, 4]], dtype=int))],
    )

    result = extract_planar_regions(extract_feature_edges(surface))

    assert not result.unsupported_regions
    assert len(result.regions) == 1
    region = result.regions[0]
    assert region.source_face_indices == (0, 1)
    assert len(region.outer_loop.points) == 4
    assert not region.hole_loops
    assert region.max_planarity_error == pytest.approx(0.0)


def test_source_normal_check_splits_cube_even_when_pyvista_angle_misses_edges() -> None:
    boundaries = BoundaryExtractor(_cube_mesh()).extract()
    features = extract_exterior_feature_edges(
        boundaries,
        config=FeatureEdgeConfig(
            feature_angle_degrees=179.0,
            boundary_edges=False,
            non_manifold_edges=False,
        ),
    )
    assert features.edge_count == 0

    result = extract_planar_regions(features, config=PlanarRegionConfig(coplanar_angle_degrees=1.0))

    assert not result.unsupported_regions
    assert len(result.regions) == 6
    assert all(len(region.source_face_indices) == 1 for region in result.regions)
    assert all(len(region.outer_loop.points) == 4 for region in result.regions)


def test_annulus_is_one_planar_region_with_one_hole() -> None:
    surface = meshio.Mesh(
        points=np.array(
            [
                [-2.0, -2.0, 0.0],
                [2.0, -2.0, 0.0],
                [2.0, 2.0, 0.0],
                [-2.0, 2.0, 0.0],
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [1.0, 1.0, 0.0],
                [-1.0, 1.0, 0.0],
            ]
        ),
        cells=[
            (
                "quad",
                np.array(
                    [
                        [0, 1, 5, 4],
                        [1, 2, 6, 5],
                        [2, 3, 7, 6],
                        [3, 0, 4, 7],
                    ],
                    dtype=int,
                ),
            )
        ],
    )

    result = extract_planar_regions(extract_feature_edges(surface))

    assert not result.unsupported_regions
    assert len(result.regions) == 1
    region = result.regions[0]
    assert len(region.outer_loop.points) == 4
    assert len(region.hole_loops) == 1
    assert len(region.hole_loops[0].points) == 4
    assert region.outer_loop.signed_area > 0.0
    assert region.hole_loops[0].signed_area < 0.0


def test_closed_2d_feature_curve_can_become_a_planar_region() -> None:
    surface = meshio.Mesh(
        points=np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]]),
        cells=[("line", np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=int))],
    )

    result = extract_planar_regions(
        extract_feature_edges(surface),
        config=PlanarRegionConfig(
            remove_small_open_components=True,
            max_open_component_edges=4,
        ),
    )

    assert not result.unsupported_regions
    assert len(result.regions) == 1
    np.testing.assert_allclose(np.abs(result.regions[0].normal), [0.0, 0.0, 1.0])


def test_small_open_surface_filter_removes_detached_triangle_but_keeps_closed_cube() -> None:
    features = extract_feature_edges(
        _cube_surface_with_detached_triangle(),
        source_kind="property",
        property_id=1,
        property_ids=(1,),
    )

    unfiltered = extract_planar_regions(features)
    filtered = extract_planar_regions(
        features,
        config=PlanarRegionConfig(
            remove_small_open_components=True,
            max_open_component_edges=3,
        ),
    )

    assert len(unfiltered.regions) == 7
    assert len(filtered.regions) == 6
    assert {region.region_index for region in filtered.regions} == set(range(6))
    assert any(
        "small_open_surface_components_removed=1" in diagnostic
        and "boundary_edges=3" in diagnostic
        and "source_faces=1" in diagnostic
        for diagnostic in filtered.diagnostics
    )


def test_small_open_surface_edge_threshold_must_be_at_least_three() -> None:
    with pytest.raises(ValueError, match="max_open_component_edges"):
        PlanarRegionConfig(
            remove_small_open_components=True,
            max_open_component_edges=2,
        )


def test_nonplanar_component_is_reported_instead_of_flattened() -> None:
    surface = meshio.Mesh(
        points=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.01],
            ]
        ),
        cells=[("triangle", np.array([[0, 1, 2], [1, 3, 2]], dtype=int))],
    )
    features = extract_feature_edges(
        surface,
        config=FeatureEdgeConfig(feature_angle_degrees=30.0),
    )

    result = extract_planar_regions(
        features,
        config=PlanarRegionConfig(
            coplanar_angle_degrees=10.0,
            planarity_absolute_tolerance=1.0e-8,
            planarity_relative_tolerance=0.0,
        ),
    )

    assert not result.regions
    assert len(result.unsupported_regions) == 1
    assert "planarity_error" in result.unsupported_regions[0].reason
