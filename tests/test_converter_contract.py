"""Phase 0 regression for the converter-to-checker mesh contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ems_mesh_checker import inspect_mesh_contract


def _converter_sample() -> Path:
    return Path(__file__).resolve().parents[2] / "ems_file_format_converter" / "sample" / "mesh_sample.neu"


def _agc_femap_10_3_sample() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "2D" / "AGC" / "pre_geom2D.neu"


def test_femap_neutral_mesh_contract() -> None:
    converter = pytest.importorskip("ems_file_format_converter")
    sample = _converter_sample()
    if not sample.is_file():
        pytest.skip(f"converter sample is not available: {sample}")

    mesh = converter.read_mesh(sample)
    summary = inspect_mesh_contract(mesh)

    assert summary.num_points > 0
    assert summary.num_cells > 0
    assert summary.has_point_ids
    assert summary.has_element_ids
    assert summary.has_property_ids


def test_agc_femap_10_3_coordinates_are_not_replaced_by_trailing_status_fields() -> None:
    converter = pytest.importorskip("ems_file_format_converter")
    sample = _agc_femap_10_3_sample()
    if not sample.is_file():
        pytest.skip(f"AGC Femap 10.3 sample is not available: {sample}")

    mesh = converter.read_mesh(sample)

    np.testing.assert_allclose(mesh.points[0], [0.0, 100.0, 0.0])
    np.testing.assert_allclose(mesh.points[2], [134.0, 150.0, 0.0])
    assert np.ptp(mesh.points[:, 0]) > 0.0
    assert np.ptp(mesh.points[:, 1]) > 0.0
