from __future__ import annotations

import numpy as np
import pytest

import meshio

from ems_mesh_checker import (
    BoundaryExtractor,
    FeatureEdgeConfig,
    extract_all_property_feature_edges,
    extract_exterior_feature_edges,
    extract_feature_edges,
    extract_interface_feature_edges,
    extract_property_group_feature_edges,
    extract_property_feature_edges,
)
from ems_mesh_checker.export import write_feature_edges_dxf


pytest.importorskip("pyvista")


def _hexahedron_mesh() -> meshio.Mesh:
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
        cell_data={"property_id": [np.array([7], dtype=int)]},
    )


def _two_tetra_mesh() -> meshio.Mesh:
    return meshio.Mesh(
        points=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ]
        ),
        cells=[("tetra", np.array([[0, 1, 2, 3], [0, 2, 1, 4]], dtype=int))],
        cell_data={"property_id": [np.array([10, 20], dtype=int)]},
    )


def test_cube_exterior_uses_pyvista_to_extract_twelve_sharp_edges() -> None:
    boundaries = BoundaryExtractor(_hexahedron_mesh()).extract()
    result = extract_exterior_feature_edges(
        boundaries,
        config=FeatureEdgeConfig(
            boundary_edges=False,
            feature_edges=True,
            non_manifold_edges=False,
            manifold_edges=False,
        ),
    )

    assert result.source_kind == "exterior"
    assert result.edge_count == 12
    assert len(result.curves) == 12
    assert all(not curve.closed for curve in result.curves)
    assert all(curve.segment_count == 1 for curve in result.curves)
    assert all(curve.length == pytest.approx(1.0) for curve in result.curves)


def test_coplanar_element_edge_is_not_a_feature_and_boundary_becomes_one_loop() -> None:
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
    result = extract_feature_edges(
        surface,
        config=FeatureEdgeConfig(
            boundary_edges=True,
            feature_edges=True,
            non_manifold_edges=False,
            manifold_edges=False,
        ),
    )

    assert result.edge_count == 6
    assert len(result.curves) == 1
    curve = result.curves[0]
    assert curve.closed
    assert curve.segment_count == 6
    assert curve.length == pytest.approx(6.0)


def test_2d_line_boundary_is_preserved_and_assembled_without_vtk_feature_filter() -> None:
    surface = meshio.Mesh(
        points=np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ]
        ),
        cells=[("line", np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=int))],
    )
    result = extract_feature_edges(surface)

    assert result.edge_count == 4
    assert len(result.curves) == 1
    assert result.curves[0].closed
    assert result.curves[0].length == pytest.approx(4.0)
    assert any("line_surface_passthrough" in item for item in result.diagnostics)


def test_degenerate_source_coordinates_are_rejected_before_feature_extraction() -> None:
    surface = meshio.Mesh(
        points=np.zeros((2, 3)),
        cells=[("line", np.array([[0, 1]], dtype=int))],
    )
    result = extract_feature_edges(surface)

    assert result.edge_count == 0
    assert not result.curves
    assert any("degenerate_source_coordinates" in item for item in result.diagnostics)


def test_interface_feature_edges_keep_property_pair_and_patch_identity() -> None:
    boundaries = BoundaryExtractor(_two_tetra_mesh()).extract()
    results = extract_interface_feature_edges(boundaries, 20, 10)

    assert len(results) == 1
    result = results[0]
    assert result.source_kind == "interface"
    assert result.property_pair == (10, 20)
    assert result.patch_index == 0
    assert result.edge_count == 3
    assert len(result.curves) == 1
    assert result.curves[0].closed


def test_feature_edge_dxf_contains_curves_not_tessellated_faces(tmp_path) -> None:
    ezdxf = pytest.importorskip("ezdxf")
    result = extract_exterior_feature_edges(BoundaryExtractor(_hexahedron_mesh()).extract())
    output = write_feature_edges_dxf(result, tmp_path / "representative_edges.dxf")

    entities = list(ezdxf.readfile(output).modelspace())
    assert len(entities) == 12
    assert {entity.dxftype() for entity in entities} == {"POLYLINE"}
    assert {entity.dxf.layer for entity in entities} == {"REP_EXTERIOR"}


def test_2d_property_boundary_combines_exterior_and_interface_into_closed_loops() -> None:
    mesh = meshio.Mesh(
        points=np.array(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        ),
        cells=[("triangle", np.array([[0, 1, 2], [0, 2, 3]], dtype=int))],
        cell_data={"property_id": [np.array([10, 20], dtype=int)]},
    )
    boundaries = BoundaryExtractor(mesh).extract()

    property_results = extract_all_property_feature_edges(boundaries)

    assert [result.property_id for result in property_results] == [10, 20]
    assert all(result.source_kind == "property" for result in property_results)
    assert all(result.edge_count == 3 for result in property_results)
    assert all(len(result.curves) == 1 and result.curves[0].closed for result in property_results)
    assert extract_property_feature_edges(boundaries, 20).property_id == 20


def test_property_group_omits_mutual_interface_and_keeps_union_boundary() -> None:
    mesh = meshio.Mesh(
        points=np.array(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        ),
        cells=[("triangle", np.array([[0, 1, 2], [0, 2, 3]], dtype=int))],
        cell_data={"property_id": [np.array([10, 20], dtype=int)]},
    )
    boundaries = BoundaryExtractor(mesh).extract()

    result = extract_property_group_feature_edges(boundaries, (20, 10))

    assert result.source_kind == "property"
    assert result.property_id == 10
    assert result.property_ids == (10, 20)
    assert result.edge_count == 4
    assert len(result.curves) == 1
    assert result.curves[0].closed
    assert result.curves[0].length == pytest.approx(4.0)


def test_property_group_rejects_missing_property_id() -> None:
    boundaries = BoundaryExtractor(_two_tetra_mesh()).extract()

    with pytest.raises(ValueError, match="not present"):
        extract_property_group_feature_edges(boundaries, (10, 999))


@pytest.mark.parametrize("angle", [0.0, 180.0])
def test_feature_angle_must_be_in_open_range(angle: float) -> None:
    with pytest.raises(ValueError, match="feature_angle_degrees"):
        FeatureEdgeConfig(feature_angle_degrees=angle)
