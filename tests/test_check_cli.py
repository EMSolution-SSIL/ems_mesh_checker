from __future__ import annotations

import json
import sys
from types import ModuleType

import meshio
import numpy as np

from ems_mesh_checker.check_cli import main


def test_cli_writes_report_without_native_backend(tmp_path, monkeypatch) -> None:
    source = tmp_path / "input.neu"
    source.write_text("placeholder", encoding="utf-8")
    mesh = meshio.Mesh(
        points=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        cells=[("triangle", np.array([[0, 1, 2]]))],
        point_data={"id": np.array([1, 2, 3])},
        cell_data={"element_id": [np.array([4])], "property_id": [np.array([5])]},
    )
    converter = ModuleType("ems_file_format_converter")
    converter.read_mesh = lambda *args, **kwargs: mesh
    monkeypatch.setitem(sys.modules, "ems_file_format_converter", converter)
    assert main([
        str(source), "--unit", "mm", "--progress-interval", "0",
        "--geometry-backend", "off",
    ]) == 0
    data = json.loads((tmp_path / "input_check.json").read_text(encoding="utf-8"))
    assert data["dimensions"]["properties"][0]["property_id"] == 5
    assert data["checks"]["area_volume_degeneracy"]["status"] == "NOT_RUN"


def test_cli_fail_on_error_but_still_writes_report(tmp_path, monkeypatch) -> None:
    source = tmp_path / "input.neu"
    source.write_text("placeholder", encoding="utf-8")
    mesh = meshio.Mesh(
        points=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        cells=[("triangle", np.array([[0, 0, 2]]))],
    )
    converter = ModuleType("ems_file_format_converter")
    converter.read_mesh = lambda *args, **kwargs: mesh
    monkeypatch.setitem(sys.modules, "ems_file_format_converter", converter)
    assert main([str(source), "--fail-on", "error"]) == 1
    assert (tmp_path / "input_check.json").is_file()


def test_cli_accepts_inward_surface_normals(tmp_path, monkeypatch) -> None:
    source = tmp_path / "input.neu"
    source.write_text("placeholder", encoding="utf-8")
    mesh = meshio.Mesh(
        points=np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        ]),
        cells=[("tetra", np.array([[0, 1, 2, 3]])), ("triangle", np.array([[0, 1, 2]]))],
        point_data={"id": np.array([1, 2, 3, 4])},
        cell_data={"element_id": [np.array([5]), np.array([6])],
                   "property_id": [np.array([3]), np.array([9])]},
    )
    converter = ModuleType("ems_file_format_converter")
    converter.read_mesh = lambda *args, **kwargs: mesh
    monkeypatch.setitem(sys.modules, "ems_file_format_converter", converter)

    assert main([str(source), "--surface-normal", "9:inward"]) == 0
    data = json.loads((tmp_path / "input_check.json").read_text(encoding="utf-8"))
    assert data["checks"]["normal_orientation"]["status"] == "PASS"
    assert "Property overrides" in data["checks"]["normal_orientation"]["reason"]


def test_required_geometry_backend_failure_returns_two_with_report(tmp_path, monkeypatch) -> None:
    source = tmp_path / "input.neu"
    source.write_text("placeholder", encoding="utf-8")
    mesh = meshio.Mesh(
        points=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        cells=[("triangle", np.array([[0, 1, 2]]))],
        point_data={"id": np.array([1, 2, 3])},
        cell_data={"element_id": [np.array([4])], "property_id": [np.array([5])]},
    )
    converter = ModuleType("ems_file_format_converter")
    converter.read_mesh = lambda *args, **kwargs: mesh
    monkeypatch.setitem(sys.modules, "ems_file_format_converter", converter)

    def missing(name: str):
        raise ModuleNotFoundError("No module named 'ems_mesh_checker_native'", name=name)

    monkeypatch.setattr("ems_mesh_checker.checks.geometry.importlib.import_module", missing)
    assert main([str(source), "--geometry-backend", "required"]) == 2
    data = json.loads((tmp_path / "input_check.json").read_text(encoding="utf-8"))
    assert data["checks"]["area_volume_degeneracy"]["status"] == "FAILED"


def test_cli_design_dimension_limits_fail_without_overwriting_spec(tmp_path, monkeypatch) -> None:
    source = tmp_path / "input.neu"
    source.write_text("placeholder", encoding="utf-8")
    mesh = meshio.Mesh(
        points=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        cells=[("triangle", np.array([[0, 1, 2]]))],
        point_data={"id": np.array([1, 2, 3])},
        cell_data={"element_id": [np.array([4])], "property_id": [np.array([5])]},
    )
    converter = ModuleType("ems_file_format_converter")
    converter.read_mesh = lambda *args, **kwargs: mesh
    monkeypatch.setitem(sys.modules, "ems_file_format_converter", converter)
    spec_path = tmp_path / "design.json"
    spec = {"absolute_tolerance": 0.01, "properties": {"5": {"size": [2.0, 1.0]}}}
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    assert main([
        str(source), "--dimension-spec", str(spec_path),
        "--geometry-backend", "off", "--fail-on", "error",
    ]) == 1
    report = json.loads((tmp_path / "input_check.json").read_text(encoding="utf-8"))
    assert report["checks"]["design_dimensions"]["status"] == "FINDINGS"
    assert any(item["code"] == "DIMENSION_OUT_OF_RANGE" for item in report["findings"])
    assert json.loads(spec_path.read_text(encoding="utf-8")) == spec
