"""Regression coverage using AGC meshes supplied with this workspace."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from ems_mesh_checker.topology.face_table import FaceTable


def _fixture_path(*parts: str) -> Path:
    return Path(__file__).resolve().parents[2] / "ems_mesh_checker" / "data" / Path(*parts)


def _read_mesh(path: Path):
    converter = pytest.importorskip("ems_file_format_converter")
    if not path.is_file():
        pytest.skip(f"AGC fixture is not available: {path}")
    return converter.read_mesh(path)


def test_agc_2d_pre_geometry_selects_quad_area_elements() -> None:
    table = FaceTable.from_mesh(_read_mesh(_fixture_path("2D", "AGC", "pre_geom2D.neu")))

    assert table.source_dimension == 2
    assert table.num_source_elements == 6630
    assert len(table) == 13423
    assert Counter(record.owner_count for record in table.iter_records()) == {1: 326, 2: 13097}
    assert {owner.cell_type for record in table.iter_records() for owner in record.owners} == {"quad"}


def test_agc_2d_extruded_geometry_enumerates_hexahedron_faces() -> None:
    table = FaceTable.from_mesh(_read_mesh(_fixture_path("2D", "AGC", "post_geom.neu")))

    assert table.source_dimension == 3
    assert table.num_source_elements == 6630
    assert len(table) == 26683
    assert Counter(record.owner_count for record in table.iter_records()) == {1: 13586, 2: 13097}
    assert {owner.cell_type for record in table.iter_records() for owner in record.owners} == {"hexahedron"}


@pytest.mark.slow
def test_large_agc_3d_fixture_remains_readable_by_converter() -> None:
    mesh = _read_mesh(_fixture_path("3D", "AGC", "post_geom.neu"))

    assert len(mesh.points) == 439794
    assert sum(len(block.data) for block in mesh.cells) == 422136
