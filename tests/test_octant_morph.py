"""Small public fixtures for positive-octant geometry edits."""

from __future__ import annotations

import meshio
import json
import numpy as np
import pytest
import sys

from ems_mesh_checker.morphing import (
    MorphingError, apply_octant_scale_case, apply_octant_shell_case,
    create_octant_scale_manifest, create_octant_shell_manifest,
)

from test_parametric_morph import _hexa_backend


def _octant_box_shell() -> meshio.Mesh:
    coordinates = (0.0, 0.5, 1.0, 1.1, 2.0)
    count = len(coordinates)
    points = np.asarray([(x, y, z) for z in coordinates for y in coordinates for x in coordinates])

    def node(i: int, j: int, k: int) -> int:
        return (k * count + j) * count + i

    cells, properties = [], []
    for k in range(count - 1):
        for j in range(count - 1):
            for i in range(count - 1):
                cells.append([
                    node(i, j, k), node(i + 1, j, k), node(i + 1, j + 1, k), node(i, j + 1, k),
                    node(i, j, k + 1), node(i + 1, j, k + 1),
                    node(i + 1, j + 1, k + 1), node(i, j + 1, k + 1),
                ])
                # The shell occupies 1 <= max(x,y,z) <= 1.1.
                lower = max(coordinates[i], coordinates[j], coordinates[k])
                properties.append(3 if lower >= 1.1 else 1 if lower >= 1.0 else 2)
    return meshio.Mesh(
        points, [("hexahedron", np.asarray(cells))],
        point_data={"id": np.arange(1001, 1001 + len(points))},
        cell_data={"element_id": [np.arange(2001, 2001 + len(cells))],
                   "property_id": [np.asarray(properties)]},
    )


def _tetra_box_shell() -> meshio.Mesh:
    hexa = _octant_box_shell()
    pattern = ((0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
               (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6))
    rows = np.asarray([row[list(tetra)] for row in hexa.cells[0].data for tetra in pattern])
    properties = np.repeat(hexa.cell_data["property_id"][0], len(pattern))
    return meshio.Mesh(
        hexa.points.copy(), [("tetra", rows)],
        point_data={"id": hexa.point_data["id"].copy()},
        cell_data={"element_id": [np.arange(3001, 3001 + len(rows))],
                   "property_id": [properties]},
    )


@pytest.mark.parametrize("mesh_factory", [_octant_box_shell, _tetra_box_shell])
def test_octant_shell_recognizes_hexa_and_tetra(mesh_factory) -> None:
    manifest = create_octant_shell_manifest(
        mesh_factory(), 1, inner_property=2, outer_property=3,
        deformable_properties=(2, 3),
    )
    assert manifest["inner_radius"] == 1.0
    assert manifest["outer_radius"] == 1.1


def test_octant_shell_recognizes_interfaces_and_rejects_wrong_neighbors() -> None:
    mesh = _octant_box_shell()
    manifest = create_octant_shell_manifest(
        mesh, 1, inner_property=2, outer_property=3, deformable_properties=(2, 3),
    )
    assert manifest["shape"] == "octant_shell_3d"
    assert manifest["inner_radius"] == 1.0
    assert manifest["outer_radius"] == 1.1
    assert manifest["far_radius"] == 2.0
    with pytest.raises(MorphingError, match="neighboring Properties"):
        create_octant_shell_manifest(
            mesh, 1, inner_property=2, outer_property=3, deformable_properties=(2,),
        )
    with pytest.raises(MorphingError, match="fixed external boundary"):
        create_octant_scale_manifest(mesh, 3, deformable_properties=(1, 2))


@pytest.mark.parametrize("parameter,value", [
    ("inner_radius", 0.995), ("inner_radius", 1.005),
    ("outer_radius", 1.095), ("outer_radius", 1.105),
])
def test_octant_shell_moves_only_selected_face_and_preserves_symmetry(parameter: str, value: float) -> None:
    pytest.importorskip("ems_mesh_morpher")
    mesh = _octant_box_shell()
    original = mesh.points.copy()
    manifest = create_octant_shell_manifest(
        mesh, 1, inner_property=2, outer_property=3, deformable_properties=(2, 3),
    )
    result = apply_octant_shell_case(mesh, manifest, geometry_backend=_hexa_backend(), **{parameter: value})
    old_radius = np.max(original, axis=1)
    new_radius = np.max(result.mesh.points, axis=1)
    assert result.report["status"] == "PASS"
    assert result.report["interpolation"] == "analytic_radial"
    assert result.report["max_target_error"] == 0
    assert np.allclose(new_radius[np.isclose(old_radius, 1.0)], value if parameter == "inner_radius" else 1.0)
    assert np.allclose(new_radius[np.isclose(old_radius, 1.1)], value if parameter == "outer_radius" else 1.1)
    assert np.array_equal(result.mesh.points[np.isclose(old_radius, 2.0)], original[np.isclose(old_radius, 2.0)])
    assert np.array_equal(result.mesh.point_data["id"], mesh.point_data["id"])
    assert np.array_equal(mesh.points, original)
    for axis in range(3):
        assert np.array_equal(result.mesh.points[np.isclose(original[:, axis], 0), axis],
                              original[np.isclose(original[:, axis], 0), axis])


def test_octant_shell_rejects_quality_failure() -> None:
    pytest.importorskip("ems_mesh_morpher")
    mesh = _octant_box_shell()
    manifest = create_octant_shell_manifest(
        mesh, 1, inner_property=2, outer_property=3, deformable_properties=(2, 3),
    )
    with pytest.raises(MorphingError, match="inverted elements"):
        apply_octant_shell_case(mesh, manifest, inner_radius=1.05, geometry_backend=_hexa_backend())


def test_octant_scale_preserves_symmetry_and_rejects_zero_scale() -> None:
    pytest.importorskip("ems_mesh_morpher")
    mesh = _octant_box_shell()
    manifest = create_octant_scale_manifest(mesh, 2, deformable_properties=(1, 3))
    result = apply_octant_scale_case(mesh, manifest, scale=0.98, geometry_backend=_hexa_backend())
    assert result.report["status"] == "PASS"
    assert result.report["max_target_error"] == 0
    assert np.allclose(result.mesh.points[np.isclose(np.max(mesh.points, axis=1), 1.)].max(axis=0), .98)
    # A neighboring Property may have external nodes on a symmetry plane.
    # They must slide within that plane rather than remain fully fixed.
    symmetry_neighbor = np.flatnonzero(np.all(np.isclose(mesh.points, [0.0, 1.1, 0.5]), axis=1))[0]
    assert result.mesh.points[symmetry_neighbor, 0] == 0.0
    assert not np.allclose(result.mesh.points[symmetry_neighbor], mesh.points[symmetry_neighbor])
    for axis in range(3):
        on_plane = np.isclose(mesh.points[:, axis], 0.0)
        assert np.all(result.mesh.points[on_plane, axis] == 0.0)
    with pytest.raises(MorphingError, match="positive"):
        apply_octant_scale_case(mesh, manifest, scale=0, geometry_backend=_hexa_backend())


def test_octant_axis_displacement_changes_extents_without_leaving_symmetry() -> None:
    pytest.importorskip("ems_mesh_morpher")
    mesh = _octant_box_shell()
    manifest = create_octant_scale_manifest(mesh, 2, deformable_properties=(1, 3))
    offset = np.asarray((0.02, -0.01, 0.0))
    result = apply_octant_scale_case(
        mesh, manifest, displacement=tuple(offset), geometry_backend=_hexa_backend(),
    )
    indices = np.asarray(manifest["target_node_ids"], dtype=int) - 1001
    expected = mesh.points[indices] * (1.0 + offset)
    assert np.allclose(result.mesh.points[indices], expected)
    assert result.report["parameters"]["new_maximum"] == pytest.approx([1.02, 0.99, 1.0])
    assert result.report["max_target_error"] == 0
    for axis in range(3):
        on_plane = np.isclose(mesh.points[:, axis], 0.0)
        assert np.all(result.mesh.points[on_plane, axis] == 0.0)
    for bad in ((0, 0, 0), (0.1, 0.2), (float("nan"), 0, 0), (-1.0, 0, 0)):
        with pytest.raises(MorphingError, match="displacement"):
            apply_octant_scale_case(mesh, manifest, displacement=bad, geometry_backend=_hexa_backend())
    with pytest.raises(MorphingError, match="exactly one"):
        apply_octant_scale_case(mesh, manifest, scale=1.01, displacement=(0.01, 0, 0),
                                geometry_backend=_hexa_backend())


def test_octant_shell_displacement_matches_absolute_face_position() -> None:
    pytest.importorskip("ems_mesh_morpher")
    mesh = _octant_box_shell()
    manifest = create_octant_shell_manifest(
        mesh, 1, inner_property=2, outer_property=3, deformable_properties=(2, 3),
    )
    moved = apply_octant_shell_case(mesh, manifest, inner_displacement=-0.005,
                                    geometry_backend=_hexa_backend())
    absolute = apply_octant_shell_case(mesh, manifest, inner_radius=0.995,
                                       geometry_backend=_hexa_backend())
    assert np.allclose(moved.mesh.points, absolute.mesh.points)
    assert moved.report["parameters"]["inner_displacement"] == -0.005
    both = apply_octant_shell_case(mesh, manifest, inner_displacement=-0.005,
                                   outer_displacement=0.005, geometry_backend=_hexa_backend())
    assert both.report["parameters"]["inner_radius"] == pytest.approx(0.995)
    assert both.report["parameters"]["outer_radius"] == pytest.approx(1.105)
    with pytest.raises(MorphingError, match="not both"):
        apply_octant_shell_case(mesh, manifest, inner_radius=0.995, inner_displacement=-0.005,
                                geometry_backend=_hexa_backend())


def test_octant_shell_cli_neutral_round_trip(tmp_path, monkeypatch) -> None:
    pytest.importorskip("ems_mesh_morpher")
    converter = pytest.importorskip("ems_file_format_converter.io")
    from ems_mesh_checker.morph_cli import main

    monkeypatch.setitem(sys.modules, "ems_mesh_checker_native", _hexa_backend())
    mesh = _octant_box_shell()
    source = tmp_path / "base.neu"
    converter.write_mesh(source, mesh)
    manifest = tmp_path / "box_manifest.json"
    assert main([
        "manifest", str(source), str(manifest), "--shape", "octant_shell",
        "--property", "1", "--inner-property", "2", "--outer-property", "3",
        "--deformable", "2", "--deformable", "3",
    ]) == 0
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([
        {"name": "inner_smaller", "inner_radius": 0.995},
        {"name": "outer_larger", "outer_displacement": 0.005},
    ]), encoding="utf-8")
    output = tmp_path / "out"
    assert main(["apply", str(source), str(manifest), str(cases), str(output)]) == 0
    restored = converter.read_mesh(output / "inner_smaller.neu")
    assert np.array_equal(restored.point_data["id"], mesh.point_data["id"])
    assert np.array_equal(restored.cell_data["element_id"][0], mesh.cell_data["element_id"][0])
    assert np.array_equal(restored.cell_data["property_id"][0], mesh.cell_data["property_id"][0])
    assert (output / "outer_larger.neu").exists()
    assert json.loads((output / "summary.json").read_text(encoding="utf-8"))["failed_count"] == 0


def test_octant_displacement_cli_neutral_round_trip(tmp_path, monkeypatch) -> None:
    pytest.importorskip("ems_mesh_morpher")
    converter = pytest.importorskip("ems_file_format_converter.io")
    from ems_mesh_checker.morph_cli import main

    monkeypatch.setitem(sys.modules, "ems_mesh_checker_native", _hexa_backend())
    source = tmp_path / "base.neu"
    converter.write_mesh(source, _octant_box_shell())
    manifest = tmp_path / "inner_manifest.json"
    assert main([
        "manifest", str(source), str(manifest), "--shape", "octant_scale",
        "--property", "2", "--deformable", "1", "--deformable", "3",
    ]) == 0
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([
        {"name": "x_larger", "displacement": [0.02, 0, 0]},
        {"name": "invalid", "displacement": [-1.0, 0, 0]},
    ]), encoding="utf-8")
    output = tmp_path / "out"
    assert main(["apply", str(source), str(manifest), str(cases), str(output)]) == 1
    restored = converter.read_mesh(output / "x_larger.neu")
    assert max(restored.points[:, 0]) == pytest.approx(2.0)  # fixed far boundary
    assert not (output / "invalid.neu").exists()
    assert json.loads((output / "summary.json").read_text(encoding="utf-8"))["failed_count"] == 1


def test_octant_scale_cli_missing_parameter_is_a_failed_case(tmp_path, monkeypatch) -> None:
    pytest.importorskip("ems_mesh_morpher")
    converter = pytest.importorskip("ems_file_format_converter.io")
    from ems_mesh_checker.morph_cli import main

    monkeypatch.setitem(sys.modules, "ems_mesh_checker_native", _hexa_backend())
    source = tmp_path / "base.neu"
    converter.write_mesh(source, _octant_box_shell())
    manifest = tmp_path / "inner_manifest.json"
    assert main([
        "manifest", str(source), str(manifest), "--shape", "octant_scale",
        "--property", "2", "--deformable", "1", "--deformable", "3",
    ]) == 0
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([{"name": "missing_scale"}]), encoding="utf-8")
    output = tmp_path / "out"
    assert main(["apply", str(source), str(manifest), str(cases), str(output)]) == 1
    assert not (output / "missing_scale.neu").exists()
    assert json.loads((output / "missing_scale.json").read_text(encoding="utf-8"))["status"] == "FAILED"
