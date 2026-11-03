"""Slow end-to-end DXF exports for the AGC regression fixtures."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest

from ems_mesh_checker import BoundaryExtractor
from ems_mesh_checker.export import dxf_to_geo, write_boundary_result_dxf


def _fixture_path(*parts: str) -> Path:
    return Path(__file__).resolve().parents[2] / "ems_mesh_checker" / "data" / Path(*parts)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("fixture", "expected_entity_type"),
    [
        (("2D", "AGC", "pre_geom2D.neu"), "LINE"),
        (("2D", "AGC", "post_geom.neu"), "3DFACE"),
    ],
)
def test_agc_boundary_result_exports_to_readable_dxf(
    tmp_path: Path, fixture: tuple[str, ...], expected_entity_type: str
) -> None:
    ezdxf = pytest.importorskip("ezdxf")
    converter = pytest.importorskip("ems_file_format_converter")
    mesh = converter.read_mesh(_fixture_path(*fixture))
    result = BoundaryExtractor(mesh).extract()

    output = write_boundary_result_dxf(result, tmp_path / "boundaries.dxf")
    entities = list(ezdxf.readfile(output).modelspace())
    expected_count = len(result.external_faces) + sum(
        len(faces) for faces in result.property_interfaces.values()
    )

    assert len(entities) == expected_count
    assert {entity.dxftype() for entity in entities} == {expected_entity_type}


@pytest.mark.slow
def test_gmsh_parses_agc_3dface_dxf_when_configured(tmp_path: Path) -> None:
    gmsh_executable = os.environ.get("GMSH_EXECUTABLE")
    if not gmsh_executable:
        pytest.skip("set GMSH_EXECUTABLE to enable the local Gmsh import regression")

    converter = pytest.importorskip("ems_file_format_converter")
    mesh = converter.read_mesh(_fixture_path("2D", "AGC", "post_geom.neu"))
    dxf_path = write_boundary_result_dxf(BoundaryExtractor(mesh).extract(), tmp_path / "boundaries.dxf")
    output = dxf_to_geo(dxf_path, tmp_path / "boundaries.geo")
    completed = subprocess.run(
        [gmsh_executable, str(output), "-parse_and_exit", "-nopopup"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Error" not in completed.stdout + completed.stderr
