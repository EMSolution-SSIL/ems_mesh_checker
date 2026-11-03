from __future__ import annotations

import meshio
import numpy as np

from ems_mesh_checker.checks import check_mesh


def _tetra_with_surface(surface_nodes: list[int]) -> meshio.Mesh:
    return meshio.Mesh(
        points=np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        ]),
        cells=[("tetra", np.array([[0, 1, 2, 3]])), ("triangle", np.array([surface_nodes]))],
        point_data={"id": np.array([101, 102, 103, 104])},
        cell_data={"element_id": [np.array([201]), np.array([301])],
                   "property_id": [np.array([3]), np.array([9])]},
    )


def test_explicit_surface_outward_and_inward_normals() -> None:
    outward = check_mesh(_tetra_with_surface([0, 2, 1]))
    inward = check_mesh(_tetra_with_surface([0, 1, 2]))
    inward_expected = check_mesh(
        _tetra_with_surface([0, 1, 2]), surface_normal_direction="inward",
    )
    outward_unexpected = check_mesh(
        _tetra_with_surface([0, 2, 1]), surface_normal_direction="inward",
    )
    assert outward.checks["normal_orientation"].status == "PASS"
    assert [item.code for item in inward.findings if item.code == "SURFACE_INWARD_NORMAL"] == [
        "SURFACE_INWARD_NORMAL",
    ]
    assert inward.checks["normal_orientation"].status == "FINDINGS"
    assert inward_expected.checks["normal_orientation"].status == "PASS"
    assert [item.code for item in outward_unexpected.findings if item.code == "SURFACE_OUTWARD_NORMAL"] == [
        "SURFACE_OUTWARD_NORMAL",
    ]


def test_surface_property_direction_override() -> None:
    mesh = _tetra_with_surface([0, 1, 2])
    report = check_mesh(mesh, surface_normal_directions={9: "inward"})
    assert report.checks["normal_orientation"].status == "PASS"


def test_cross_property_surface_overlay_does_not_block_normal_check() -> None:
    mesh = _tetra_with_surface([0, 1, 2])
    mesh.cells[1] = meshio.CellBlock("triangle", np.array([[0, 1, 2], [0, 1, 2]]))
    mesh.cell_data["element_id"][1] = np.array([301, 302])
    mesh.cell_data["property_id"][1] = np.array([9, 10])
    report = check_mesh(mesh, surface_normal_direction="inward")
    assert report.checks["normal_orientation"].status == "PASS"
    assert report.checks["free_faces"].status != "NOT_RUN"
    assert [(item.code, item.severity) for item in report.findings] == [
        ("OVERLAPPING_SURFACE_PROPERTIES", "INFO"),
    ]
    assert report.findings[0].related_element_ids == (301,)


def test_standalone_surface_shared_edge_orientation() -> None:
    mesh = meshio.Mesh(
        points=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        cells=[("triangle", np.array([[0, 1, 2], [1, 2, 3]]))],
        point_data={"id": np.arange(11, 15)},
        cell_data={"element_id": [np.array([21, 22])], "property_id": [np.array([5, 5])]},
    )
    report = check_mesh(mesh)
    assert any(item.code == "SURFACE_ORIENTATION_CONFLICT" for item in report.findings)


def test_property_interface_requires_owner_side_to_decide_outward() -> None:
    mesh = _tetra_with_surface([0, 2, 1])
    mesh.points = np.vstack((mesh.points, [[0.0, 0.0, -1.0]]))
    mesh.point_data["id"] = np.array([101, 102, 103, 104, 105])
    mesh.cells[0] = meshio.CellBlock("tetra", np.array([[0, 1, 2, 3], [0, 2, 1, 4]]))
    mesh.cell_data["element_id"][0] = np.array([201, 202])
    mesh.cell_data["property_id"][0] = np.array([3, 4])
    ambiguous = check_mesh(mesh)
    selected = check_mesh(mesh, surface_owner_properties={9: 3})
    assert any(item.code == "NORMAL_OWNER_AMBIGUOUS" for item in ambiguous.findings)
    assert selected.checks["normal_orientation"].status == "PASS"


def test_unmatched_surface_is_a_candidate_not_an_error() -> None:
    mesh = _tetra_with_surface([0, 2, 1])
    mesh.points = np.vstack((mesh.points, [[2.0, 0.0, 0.0]]))
    mesh.point_data["id"] = np.array([101, 102, 103, 104, 105])
    mesh.cells[1] = meshio.CellBlock("triangle", np.array([[0, 2, 4]]))
    report = check_mesh(mesh)
    assert any(item.code == "SURFACE_WITHOUT_VOLUME_FACE" for item in report.findings)


def test_closed_tetra_surface_detects_one_reversed_face() -> None:
    mesh = _tetra_with_surface([0, 2, 1])
    outward_faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    mesh.cells[1] = meshio.CellBlock("triangle", outward_faces)
    mesh.cell_data["element_id"][1] = np.arange(301, 305)
    mesh.cell_data["property_id"][1] = np.full(4, 9)
    outward = check_mesh(mesh, geometry_backend="off")
    assert outward.checks["normal_orientation"].status == "PASS"

    reversed_mesh = _tetra_with_surface([0, 2, 1])
    flipped_faces = outward_faces.copy()
    flipped_faces[3] = flipped_faces[3, ::-1]
    reversed_mesh.cells[1] = meshio.CellBlock("triangle", flipped_faces)
    reversed_mesh.cell_data["element_id"][1] = np.arange(301, 305)
    reversed_mesh.cell_data["property_id"][1] = np.full(4, 9)
    reversed_report = check_mesh(reversed_mesh, geometry_backend="off")
    inward = [item for item in reversed_report.findings if item.code == "SURFACE_INWARD_NORMAL"]
    assert [item.element_id for item in inward] == [304]
    assert any(item.code == "SURFACE_ORIENTATION_CONFLICT" for item in reversed_report.findings)

    inward_mesh = _tetra_with_surface([0, 2, 1])
    inward_mesh.cells[1] = meshio.CellBlock("triangle", outward_faces[:, ::-1])
    inward_mesh.cell_data["element_id"][1] = np.arange(301, 305)
    inward_mesh.cell_data["property_id"][1] = np.full(4, 9)
    inward_report = check_mesh(
        inward_mesh, geometry_backend="off", surface_normal_direction="inward",
    )
    assert inward_report.checks["normal_orientation"].status == "PASS"
