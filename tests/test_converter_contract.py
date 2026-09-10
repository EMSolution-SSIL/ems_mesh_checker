"""Regression tests for the converter-to-checker mesh contract."""

from __future__ import annotations

from pathlib import Path

import pytest

from ems_mesh_checker import inspect_mesh_contract


def _sample(*parts: str) -> Path:
    return Path(__file__).resolve().parents[1].joinpath("data", "3D", *parts)


@pytest.mark.parametrize(
    "sample",
    [
        _sample("IronCoil", "post_geom.atl"),
        _sample("QuadTri", "post_geom.neu"),
    ],
)
def test_public_sample_mesh_contract(sample: Path) -> None:
    converter = pytest.importorskip("ems_file_format_converter")

    mesh = converter.read_mesh(sample)
    summary = inspect_mesh_contract(mesh)

    assert summary.num_points > 0
    assert summary.num_cells > 0
    assert summary.has_point_ids
    assert summary.has_element_ids
    assert summary.has_property_ids
