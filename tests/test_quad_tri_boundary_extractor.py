from __future__ import annotations

from pathlib import Path

import pytest

from ems_mesh_checker import BoundaryExtractor


converter = pytest.importorskip("ems_file_format_converter")


def _fixture() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "3D" / "QuadTri" / "post_geom.neu"


def test_quad_tri_fixture_suppresses_only_same_property_coupling_faces() -> None:
    mesh = converter.read_mesh(_fixture())

    original = BoundaryExtractor(mesh).extract()
    suppressed = BoundaryExtractor(
        mesh,
        suppress_quad_tri_interfaces=True,
    ).extract()

    assert len(original.external_faces) == 6816
    assert len(suppressed.suppressed_quad_tri_interfaces) == 48
    assert len(suppressed.external_faces) == 6816 - 3 * 48
    assert {interface.property_ids for interface in suppressed.suppressed_quad_tri_interfaces} == {
        (1,),
    }
    assert not suppressed.nonmanifold_faces
    assert not suppressed.unclassified_faces
