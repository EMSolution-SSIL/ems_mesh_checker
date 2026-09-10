from __future__ import annotations

import numpy as np
import pytest

import meshio

from ems_mesh_checker.boundary import BoundaryExtractionError, BoundaryExtractor


def _two_tetra_mesh(
    properties: tuple[int, int] = (10, 20), *, point_ids: bool = False
) -> meshio.Mesh:
    point_data = {"id": np.array([101, 102, 103, 104, 105], dtype=int)} if point_ids else {}
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
        point_data=point_data,
        cell_data={
            "element_id": [np.array([1001, 1002], dtype=int)],
            "property_id": [np.array(properties, dtype=int)],
        },
    )


def _quad_tri_coupling_mesh(
    *,
    valid_diagonal: bool = True,
    properties: tuple[int, int, int] = (10, 10, 10),
) -> meshio.Mesh:
    second_base = [4, 6, 7, 8] if valid_diagonal else [4, 5, 7, 9]
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
                [0.5, 0.5, 2.0],
                [0.5, 0.5, 2.5],
            ]
        ),
        cells=[
            ("hexahedron", np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=int)),
            (
                "tetra",
                np.array([[4, 5, 6, 8], second_base], dtype=int),
            ),
        ],
        cell_data={
            "property_id": [
                np.array([properties[0]], dtype=int),
                np.array(properties[1:], dtype=int),
            ]
        },
    )


def test_same_property_shared_face_is_internal_not_interface() -> None:
    result = BoundaryExtractor(_two_tetra_mesh((10, 10))).extract()

    assert len(result.external_faces) == 6
    assert len(result.internal_faces) == 1
    assert not result.property_interfaces
    assert not result.warnings


def test_different_property_shared_face_is_a_canonically_oriented_interface() -> None:
    result = BoundaryExtractor(_two_tetra_mesh(point_ids=True)).extract()

    assert len(result.external_faces) == 6
    assert len(result.internal_faces) == 0
    assert list(result.property_interfaces) == [(10, 20)]
    interface = result.get_interface(20, 10)
    assert sum(len(block.data) for block in interface.cells) == 1
    assert interface.point_data["source_node_id"].tolist() == [101, 103, 102]
    assert interface.cell_data["owner_element_id"][0].tolist() == [1001]
    assert interface.cell_data["neighbor_element_id"][0].tolist() == [1002]
    assert interface.cell_data["owner_property_id"][0].tolist() == [10]
    assert interface.cell_data["neighbor_property_id"][0].tolist() == [20]

    triangle = interface.points[interface.cells[0].data[0]]
    normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    assert normal[2] < 0.0


def test_missing_property_is_reported_but_external_faces_remain_available() -> None:
    mesh = _two_tetra_mesh()
    mesh.cell_data.pop("property_id")

    result = BoundaryExtractor(mesh).extract()

    assert len(result.external_faces) == 6
    assert len(result.unclassified_faces) == 1
    assert len(result.warnings) == 1
    with pytest.raises(BoundaryExtractionError, match="missing_property_id"):
        BoundaryExtractor(mesh, strict=True).extract()


def test_nonmanifold_face_is_reported_and_not_emitted_as_an_interface() -> None:
    mesh = meshio.Mesh(
        points=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, 2.0],
            ]
        ),
        cells=[("tetra", np.array([[0, 1, 2, 3], [0, 2, 1, 4], [0, 1, 2, 5]], dtype=int))],
        cell_data={"property_id": [np.array([10, 20, 30], dtype=int)]},
    )

    result = BoundaryExtractor(mesh).extract()

    assert len(result.nonmanifold_faces) == 1
    assert not result.property_interfaces
    assert len(result.warnings) == 1
    assert "nonmanifold_face" in result.warnings[0]


def test_disconnected_property_pair_is_returned_as_two_patches() -> None:
    first = _two_tetra_mesh().points
    second = _two_tetra_mesh().points + np.array([10.0, 0.0, 0.0])
    points = np.vstack((first, second))
    cells = np.array(
        [[0, 1, 2, 3], [0, 2, 1, 4], [5, 6, 7, 8], [5, 7, 6, 9]], dtype=int
    )
    mesh = meshio.Mesh(
        points=points,
        cells=[("tetra", cells)],
        cell_data={"property_id": [np.array([10, 20, 10, 20], dtype=int)]},
    )

    result = BoundaryExtractor(mesh).extract()

    assert len(result.property_interfaces[(10, 20)]) == 2
    patches = result.get_interface_patches(10, 20)
    assert len(patches) == 2
    assert [sum(len(block.data) for block in patch.cells) for patch in patches] == [1, 1]


def test_quad_tri_coupling_faces_are_kept_by_default() -> None:
    result = BoundaryExtractor(_quad_tri_coupling_mesh()).extract()

    assert len(result.external_faces) == 12
    assert not result.suppressed_quad_tri_interfaces


def test_quad_tri_coupling_faces_are_suppressed_only_when_requested() -> None:
    result = BoundaryExtractor(
        _quad_tri_coupling_mesh(),
        suppress_quad_tri_interfaces=True,
    ).extract()

    assert len(result.external_faces) == 9
    assert len(result.suppressed_quad_tri_interfaces) == 1
    interface = result.suppressed_quad_tri_interfaces[0]
    assert interface.face_count == 3
    assert interface.property_ids == (10,)
    assert len(interface.quad_face.corner_key) == 4
    assert {face.corner_key for face in interface.triangle_faces} == {
        (4, 5, 6),
        (4, 6, 7),
    }
    assert not result.warnings


def test_quad_tri_coupling_between_properties_is_not_suppressed() -> None:
    result = BoundaryExtractor(
        _quad_tri_coupling_mesh(properties=(10, 20, 20)),
        suppress_quad_tri_interfaces=True,
    ).extract()

    assert len(result.external_faces) == 12
    assert not result.suppressed_quad_tri_interfaces


def test_quad_tri_coupling_with_mixed_triangle_properties_is_not_suppressed() -> None:
    result = BoundaryExtractor(
        _quad_tri_coupling_mesh(properties=(10, 10, 20)),
        suppress_quad_tri_interfaces=True,
    ).extract()

    assert len(result.external_faces) == 12
    assert not result.suppressed_quad_tri_interfaces


def test_quad_tri_suppression_rejects_triangles_sharing_quad_perimeter_edge() -> None:
    result = BoundaryExtractor(
        _quad_tri_coupling_mesh(valid_diagonal=False),
        suppress_quad_tri_interfaces=True,
    ).extract()

    assert not result.suppressed_quad_tri_interfaces
