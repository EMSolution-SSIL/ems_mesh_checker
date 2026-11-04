"""Boundary-classification regressions using the AGC fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from ems_mesh_checker.boundary import BoundaryExtractor


def _post_geometry_fixture() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "ems_mesh_checker"
        / "data"
        / "2D"
        / "AGC"
        / "post_geom.neu"
    )


def test_agc_extruded_geometry_classifies_every_shared_hexahedron_face() -> None:
    sample = _post_geometry_fixture()
    if not sample.is_file():
        pytest.skip("private AGC fixture is not available")
    converter = pytest.importorskip("ems_file_format_converter")
    mesh = converter.read_mesh(sample)

    result = BoundaryExtractor(mesh).extract()

    assert len(result.external_faces) == 13586
    assert not result.nonmanifold_faces
    assert not result.unclassified_faces
    assert len(result.internal_faces) + sum(
        len(faces) for faces in result.property_interfaces.values()
    ) == 13097
