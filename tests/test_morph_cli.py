from __future__ import annotations

import json
import os
import sys
from pathlib import Path
import shutil

import ezdxf
import numpy as np
import pytest

from ems_mesh_checker.morph_cli import main

from test_geometry_follow import _grid, _triangle_backend
from test_parametric_morph import _annular_hexa, _hexa_backend


def test_cli_keeps_failed_case_mesh_absent(tmp_path, monkeypatch) -> None:
    pytest.importorskip("ems_mesh_morpher")
    converter = pytest.importorskip("ems_file_format_converter.io")
    monkeypatch.setitem(sys.modules, "ems_mesh_checker_native", _triangle_backend())
    mesh = _grid()
    mesh.points = np.column_stack((mesh.points, np.zeros(len(mesh.points))))
    source = tmp_path / "base.neu"
    converter.write_mesh(source, mesh)
    manifest_path = tmp_path / "manifest.json"
    assert main(["manifest", str(source), str(manifest_path), "--property", "2", "--deformable", "1"]) == 0
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["property_id"] == 2
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps([
        {"name": "thin", "thickness": 1.8},
        {"name": "bad", "thickness": 0.0},
        {"name": "thick", "thickness": 2.2},
    ]), encoding="utf-8")
    output = tmp_path / "results"
    assert main(["apply", str(source), str(manifest_path), str(cases_path), str(output)]) == 1
    assert not (output / "bad.neu").exists()
    assert json.loads((output / "bad.json").read_text(encoding="utf-8"))["status"] == "FAILED"
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["case_count"] == 3 and summary["failed_count"] == 1
    assert (output / "summary.csv").is_file()
    for name in ("thin", "thick"):
        assert json.loads((output / f"{name}.json").read_text(encoding="utf-8"))["status"] == "PASS"
        restored = converter.read_mesh(output / f"{name}.neu")
        assert len(restored.points) == len(mesh.points)
        assert restored.point_data["id"].tolist() == mesh.point_data["id"].tolist()
        assert restored.cell_data["element_id"][0].tolist() == mesh.cell_data["element_id"][0].tolist()

    sketch = tmp_path / "editable.dxf"
    assert main(["export-dxf", str(manifest_path), str(sketch)]) == 0
    document = ezdxf.readfile(sketch)
    right = next(item for item in document.modelspace() if item.dxf.layer == "EMS_MORPH_RIGHT")
    top = next(item for item in document.modelspace() if item.dxf.layer == "EMS_MORPH_TOP")
    bottom = next(item for item in document.modelspace() if item.dxf.layer == "EMS_MORPH_BOTTOM")
    right.dxf.start = (1.1, -1., 0.)
    right.dxf.end = (1.1, 1., 0.)
    top.dxf.start = (1.1, 1., 0.)
    bottom.dxf.end = (1.1, -1., 0.)
    document.save()
    dxf_output = tmp_path / "dxf_results"
    assert main(["apply-dxf", str(source), str(manifest_path), str(sketch), str(dxf_output)]) == 0
    assert (dxf_output / "edited.neu").is_file()


def test_cli_cylinder_neutral_round_trip(tmp_path, monkeypatch) -> None:
    pytest.importorskip("ems_mesh_morpher")
    converter = pytest.importorskip("ems_file_format_converter.io")
    monkeypatch.setitem(sys.modules, "ems_mesh_checker_native", _hexa_backend())
    source = tmp_path / "cylinder.neu"
    converter.write_mesh(source, _annular_hexa())
    manifest = tmp_path / "cylinder_manifest.json"
    assert main([
        "manifest", str(source), str(manifest), "--shape", "cylinder",
        "--property", "2", "--deformable", "1", "--axis-point", "0", "0", "0",
        "--axis-direction", "0", "0", "1",
    ]) == 0
    cases = tmp_path / "cylinder_cases.json"
    cases.write_text(json.dumps([{"name": "thicker", "inner_radius": 0.9}]), encoding="utf-8")
    output = tmp_path / "cylinder_results"
    assert main(["apply", str(source), str(manifest), str(cases), str(output)]) == 0
    assert (output / "thicker.neu").is_file()
    assert json.loads((output / "summary.json").read_text(encoding="utf-8"))["failed_count"] == 0


def test_cli_gmsh_evaluated_geo_case(tmp_path, monkeypatch) -> None:
    pytest.importorskip("ems_mesh_morpher")
    converter = pytest.importorskip("ems_file_format_converter.io")
    candidate = os.environ.get("GMSH_EXECUTABLE") or shutil.which("gmsh") or "C:/EMSolution/gmsh-4.13.1-Windows64/gmsh.exe"
    if not Path(candidate).is_file():
        pytest.skip("Gmsh executable is not available")
    monkeypatch.setitem(sys.modules, "ems_mesh_checker_native", _triangle_backend())
    mesh = _grid()
    mesh.points = np.column_stack((mesh.points, np.zeros(len(mesh.points))))
    source = tmp_path / "base.neu"
    converter.write_mesh(source, mesh)
    manifest = tmp_path / "manifest.json"
    assert main(["manifest", str(source), str(manifest), "--property", "2", "--deformable", "1"]) == 0
    geometry = tmp_path / "editable.geo"
    assert main(["export-geo", str(manifest), str(geometry)]) == 0
    script = geometry.read_text(encoding="utf-8")
    geometry.write_text(script.replace("Point(2) = {1,", "Point(2) = {1.1,").replace(
        "Point(3) = {1,", "Point(3) = {1.1,",
    ), encoding="utf-8")
    output = tmp_path / "from_geo"
    assert main([
        "apply-geo", str(source), str(manifest), str(geometry), str(output),
        "--gmsh-executable", candidate,
    ]) == 0
    assert (output / "edited.neu").is_file()
