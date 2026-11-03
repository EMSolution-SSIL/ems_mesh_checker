from __future__ import annotations

from types import SimpleNamespace
import sys

import meshio
import numpy as np
import pytest

from ems_mesh_checker.morphing import (
    MorphingError, apply_rectangle_case, create_rectangle_manifest, mesh_fingerprint,
)


def _grid() -> meshio.Mesh:
    values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    points = np.array([(x, y) for y in values for x in values], dtype=float)
    cells: list[list[int]] = []
    properties: list[int] = []
    for j in range(4):
        for i in range(4):
            a = j * 5 + i
            b = a + 1
            c = a + 5
            d = c + 1
            central = 1 <= i <= 2 and 1 <= j <= 2
            cells.extend(([a, b, d], [a, d, c]))
            properties.extend((2 if central else 1,) * 2)
    return meshio.Mesh(
        points, [("triangle", np.asarray(cells))],
        point_data={"id": np.arange(101, 126)},
        cell_data={"element_id": [np.arange(201, 233)],
                   "property_id": [np.asarray(properties)]},
    )


def _triangle_backend():
    def analyze(_kind, coords):
        vectors = coords[:, :, :2]
        first = vectors[:, 1] - vectors[:, 0]
        second = vectors[:, 2] - vectors[:, 0]
        cross = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
        scale = np.maximum.reduce([
            np.linalg.norm(vectors[:, 1] - vectors[:, 0], axis=1),
            np.linalg.norm(vectors[:, 2] - vectors[:, 1], axis=1),
            np.linalg.norm(vectors[:, 0] - vectors[:, 2], axis=1),
        ])
        return {"measure": np.abs(cross) / 2, "min_jacobian": cross,
                "max_jacobian": cross, "scale": scale}
    return SimpleNamespace(
        __name__="analytical_triangle_test", __version__="1",
        SUPPORTED_CELL_TYPES=("triangle",), analyze_block=analyze,
    )


def test_rectangle_manifest_tracks_original_ids_and_detects_mesh_change() -> None:
    mesh = _grid()
    manifest = create_rectangle_manifest(mesh, 2, deformable_properties=(1,))
    assert manifest["mesh_sha256"] == mesh_fingerprint(mesh)
    assert manifest["minimum"] == [-1.0, -1.0]
    assert manifest["maximum"] == [1.0, 1.0]
    assert len(manifest["boundary_node_ids"]) == 8
    assert len(manifest["target_node_ids"]) == 9
    assert len(manifest["boundary_edges"]) == 8
    altered = _grid()
    altered.points[0, 0] += 0.01
    assert mesh_fingerprint(altered) != manifest["mesh_sha256"]
    with pytest.raises(MorphingError, match="does not match"):
        apply_rectangle_case(altered, manifest)
    tampered = dict(manifest)
    tampered["target_node_ids"] = tampered["target_node_ids"][:-1]
    with pytest.raises(MorphingError, match="manifest was edited"):
        apply_rectangle_case(mesh, tampered)


def test_rectangle_rejects_disconnected_property_and_wrong_surroundings() -> None:
    mesh = _grid()
    with pytest.raises(MorphingError, match="do not exist"):
        create_rectangle_manifest(mesh, 2, deformable_properties=(99,))
    with pytest.raises(MorphingError, match="distinct"):
        create_rectangle_manifest(mesh, 2, deformable_properties=(2,))
    mesh.points[7, 1] += 0.2
    with pytest.raises(MorphingError, match="rectangle sides"):
        create_rectangle_manifest(mesh, 2, deformable_properties=(1,))


def test_three_cases_share_base_and_preserve_ids() -> None:
    pytest.importorskip("ems_mesh_morpher")
    mesh = _grid()
    manifest = create_rectangle_manifest(mesh, 2, deformable_properties=(1,))
    backend = _triangle_backend()
    original = mesh.points.copy()
    outputs = [
        apply_rectangle_case(mesh, manifest, thickness=t, anchor="center", geometry_backend=backend)
        for t in (1.8, 2.0, 2.2)
    ]
    assert np.array_equal(mesh.points, original)
    assert np.array_equal(outputs[1].mesh.points, original)
    assert np.allclose(outputs[0].mesh.points[12, :2], [0, 0])
    for item, thickness in zip(outputs, (1.8, 2.0, 2.2), strict=True):
        assert item.report["status"] == "PASS"
        assert item.report["max_target_error"] < 1e-12
        assert item.report["thickness"] == thickness
        assert np.array_equal(item.mesh.point_data["id"], mesh.point_data["id"])
        assert np.array_equal(item.mesh.cells[0].data, mesh.cells[0].data)
        assert np.array_equal(item.mesh.cell_data["element_id"][0], mesh.cell_data["element_id"][0])
        assert np.array_equal(item.mesh.cell_data["property_id"][0], mesh.cell_data["property_id"][0])
        assert np.array_equal(item.mesh.points[[0, 4, 20, 24]], original[[0, 4, 20, 24]])


def test_quality_gate_rejects_collapsed_surroundings() -> None:
    pytest.importorskip("ems_mesh_morpher")
    mesh = _grid()
    manifest = create_rectangle_manifest(mesh, 2, deformable_properties=(1,))
    with pytest.raises(MorphingError, match="degenerate|quality|orientation flipped"):
        apply_rectangle_case(mesh, manifest, translation=(1.0, 0.0), geometry_backend=_triangle_backend())


def test_missing_morpher_fails_closed(monkeypatch) -> None:
    mesh = _grid()
    manifest = create_rectangle_manifest(mesh, 2, deformable_properties=(1,))
    monkeypatch.setitem(sys.modules, "ems_mesh_morpher", None)
    monkeypatch.setitem(sys.modules, "ems_mesh_morpher.config", None)
    monkeypatch.setitem(sys.modules, "ems_mesh_morpher.mesh_motion", None)
    with pytest.raises(MorphingError, match="install ems_mesh_morpher"):
        apply_rectangle_case(mesh, manifest, thickness=2.1, geometry_backend=_triangle_backend())
