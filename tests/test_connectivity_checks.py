from __future__ import annotations

import json

import meshio
import numpy as np

from ems_mesh_checker.checks import check_mesh


def _adjacent_tetra(*, disconnected: bool = False) -> meshio.Mesh:
    points = [
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
    ]
    if disconnected:
        points.extend(points[:3])
    other = [5, 7, 6, 4] if disconnected else [0, 2, 1, 4]
    return meshio.Mesh(
        points=np.asarray(points),
        cells=[("tetra", np.array([[0, 1, 2, 3], other]))],
        point_data={"id": np.arange(101, 101 + len(points))},
        cell_data={"element_id": [np.array([201, 202])], "property_id": [np.array([3, 4])]},
    )


def test_shared_face_is_canceled_even_across_properties() -> None:
    report = check_mesh(_adjacent_tetra())
    assert report.topology["free_face_count"] == 6
    assert report.topology["coincident_unshared_face_group_count"] == 0
    assert report.checks["connectivity"].status == "PASS"
    assert {face["property_id"] for face in report.raw_free_faces} == {3, 4}


def test_double_nodes_leave_two_free_faces_and_a_connection_candidate() -> None:
    report = check_mesh(_adjacent_tetra(disconnected=True), geometry_backend="off")
    assert report.topology["free_face_count"] == 8
    assert report.topology["coincident_node_group_count"] == 3
    assert report.topology["coincident_unshared_face_group_count"] == 1
    assert report.checks["connectivity"].status == "FINDINGS"
    assert len(report.topology["free_face_patches"]) == 2
    candidates = [item for item in report.findings if item.code == "COINCIDENT_UNSHARED_FACES"]
    assert len(candidates) == 1
    assert {candidates[0].element_id, *candidates[0].related_element_ids} == {201, 202}
    assert report.checks["area_volume_degeneracy"].status == "NOT_RUN"


def test_free_face_artifact_preserves_original_node_and_element_ids(tmp_path) -> None:
    report = check_mesh(_adjacent_tetra())
    data = json.loads(report.write_free_faces_json(tmp_path / "free_faces.json").read_text(encoding="utf-8"))
    assert len(data["faces"]) == 6
    assert set(data["faces"][0]["node_ids"]).issubset(set(range(101, 106)))
    assert data["faces"][0]["element_id"] in {201, 202}


def test_nonmanifold_face_is_reported_with_owners() -> None:
    mesh = _adjacent_tetra()
    mesh.points = np.vstack((mesh.points, [[0.0, 0.0, 2.0]]))
    mesh.point_data["id"] = np.arange(101, 107)
    mesh.cells[0] = meshio.CellBlock("tetra", np.vstack((mesh.cells[0].data, [0, 1, 2, 5])))
    mesh.cell_data["element_id"][0] = np.array([201, 202, 203])
    mesh.cell_data["property_id"][0] = np.array([3, 4, 5])
    report = check_mesh(mesh)
    assert report.topology["nonmanifold_face_count"] == 1
    assert any(item.code == "NONMANIFOLD_FACE" for item in report.findings)


def test_explicit_tolerance_groups_nearby_nodes_without_all_pairs() -> None:
    mesh = _adjacent_tetra(disconnected=True)
    mesh.points[5:8] += np.array([0.0, 0.0, 1.0e-5])
    exact = check_mesh(mesh)
    approximate = check_mesh(mesh, coordinate_tolerance=2.0e-5)
    assert exact.topology["coincident_unshared_face_group_count"] == 0
    assert approximate.topology["coincident_unshared_face_group_count"] == 1


def _cubical_shell(*, vented: bool = False) -> meshio.Mesh:
    def node(i: int, j: int, k: int) -> int:
        return i + 4 * j + 16 * k

    points = np.array([[i, j, k] for k in range(4) for j in range(4) for i in range(4)], dtype=float)
    cells = []
    for k in range(3):
        for j in range(3):
            for i in range(3):
                if (i, j, k) == (1, 1, 1) or (vented and (i, j, k) == (1, 1, 0)):
                    continue
                cells.append([
                    node(i, j, k), node(i + 1, j, k), node(i + 1, j + 1, k), node(i, j + 1, k),
                    node(i, j, k + 1), node(i + 1, j, k + 1),
                    node(i + 1, j + 1, k + 1), node(i, j + 1, k + 1),
                ])
    return meshio.Mesh(
        points=points,
        cells=[("hexahedron", np.asarray(cells))],
        point_data={"id": np.arange(101, 165)},
        cell_data={"element_id": [np.arange(201, 201 + len(cells))],
                   "property_id": [np.full(len(cells), 3)]},
    )


def test_enclosed_closed_shell_is_a_candidate_not_a_hole_error() -> None:
    report = check_mesh(_cubical_shell(), geometry_backend="off")
    assert report.topology["free_face_count"] == 60
    assert len(report.topology["free_face_patches"]) == 2
    assert all(patch["closed"] for patch in report.topology["free_face_patches"])
    candidates = report.topology["enclosed_free_face_shell_candidates"]
    assert len(candidates) == 1
    assert candidates[0]["bounds"]["minimum"] == [1.0, 1.0, 1.0]
    assert candidates[0]["bounds"]["maximum"] == [2.0, 2.0, 2.0]
    assert all(face["patch_index"] in {1, 2} for face in report.raw_free_faces)
    findings = [item for item in report.findings if item.code == "ENCLOSED_FREE_FACE_SHELL_CANDIDATE"]
    assert len(findings) == 1
    assert findings[0].severity == "INFO"
    assert not any(item.severity == "ERROR" for item in report.findings)


def test_vented_cavity_is_not_reported_as_enclosed_shell() -> None:
    report = check_mesh(_cubical_shell(vented=True), geometry_backend="off")
    assert len(report.topology["free_face_patches"]) == 1
    assert report.topology["enclosed_free_face_shell_candidates"] == []
