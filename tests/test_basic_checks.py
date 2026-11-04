from __future__ import annotations

import json

import meshio
import numpy as np
import pytest

from ems_mesh_checker.checks import check_mesh
from ems_mesh_checker.checks.report import MeshCheckError


def _two_regions() -> meshio.Mesh:
    return meshio.Mesh(
        points=np.array([
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
            [5.0, 0.0], [6.0, 0.0], [5.0, 1.0],
            [100.0, 100.0],
        ]),
        cells=[("triangle", np.array([[0, 1, 2], [3, 4, 5]]))],
        point_data={"id": np.arange(101, 108)},
        cell_data={"element_id": [np.array([201, 202])], "property_id": [np.array([7, 7])]},
    )


def test_dimensions_use_referenced_nodes_and_separate_same_property_components() -> None:
    mesh = _two_regions()
    report = check_mesh(mesh, geometry_backend="off")

    assert report.dimensions["model"]["minimum"] == [0.0, 0.0]
    assert report.dimensions["model"]["maximum"] == [6.0, 1.0]
    assert report.dimensions["properties"][0]["bounds"]["size"] == [6.0, 1.0]
    assert [item["bounds"]["size"] for item in report.dimensions["components"]] == [
        [1.0, 1.0], [1.0, 1.0],
    ]
    assert report.dimensions["model"]["maximum_node_ids"][0] == 105
    assert [(item.code, item.node_ids) for item in report.findings] == [("UNUSED_NODE", (107,))]
    assert report.checks["area_volume_degeneracy"].status == "NOT_RUN"
    assert np.array_equal(mesh.points, _two_regions().points)


def test_bad_ids_coordinates_and_references_are_auditable() -> None:
    mesh = _two_regions()
    mesh.point_data["id"][1] = 101
    mesh.cell_data["element_id"][0][1] = 201
    mesh.points[2, 0] = np.nan
    mesh.cells[0] = meshio.CellBlock("triangle", np.array([[0, 1, 2], [3, 4, 99]]))

    report = check_mesh(mesh, dimension_spec={"model": {"size": [1.0, 1.0]}})
    codes = {item.code for item in report.findings}
    assert {"DUPLICATE_NODE_ID", "DUPLICATE_ELEMENT_ID", "NONFINITE_COORDINATE", "INVALID_NODE_REFERENCE"} <= codes
    assert report.dimensions["model"]["maximum"] == [1.0, 0.0]
    assert report.checks["input_integrity"].status == "FINDINGS"
    assert report.checks["design_dimensions"].status == "NOT_RUN"
    assert "NaN" not in json.dumps(report.to_dict(), allow_nan=False)


def test_repeated_node_zero_length_edge_and_duplicate_element() -> None:
    mesh = meshio.Mesh(
        points=np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 1.0]]),
        cells=[("triangle", np.array([[0, 1, 2], [0, 1, 2], [0, 0, 2]]))],
        point_data={"id": np.array([11, 12, 13])},
        cell_data={"element_id": [np.array([21, 22, 23])], "property_id": [np.array([3, 3, 3])]},
    )
    report = check_mesh(mesh)
    codes = {item.code for item in report.findings}
    assert {"DUPLICATE_ELEMENT", "REPEATED_ELEMENT_NODE", "ZERO_LENGTH_EDGE"} <= codes
    assert report.checks["area_volume_degeneracy"].status == "NOT_RUN"
    assert report.checks["free_faces"].status == "NOT_RUN"
    assert report.checks["normal_orientation"].status == "NOT_RUN"


def test_missing_metadata_and_unsupported_cells_are_not_reported_as_checked() -> None:
    mesh = meshio.Mesh(
        points=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]]),
        cells=[("triangle6", np.array([[0, 1, 2, 3, 4, 5]]))],
    )
    report = check_mesh(mesh)
    assert report.checks["supported_elements"].status == "UNSUPPORTED"
    assert report.checks["property_dimensions"].status == "NOT_RUN"
    assert report.dimensions["model"]["node_count"] == 6
    assert report.dimensions["components"] == []
    assert report.checks["area_volume_degeneracy"].status == "NOT_RUN"


def test_json_report_is_utf8_and_keeps_original_ids(tmp_path) -> None:
    report = check_mesh(
        _two_regions(), source="example.neu", coordinate_unit="mm", geometry_backend="off",
    )
    path = report.write_json(tmp_path / "report.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["summary"]["coordinate_unit"] == "mm"
    assert data["findings"][0]["node_ids"] == [107]
    assert data["checks"]["area_volume_degeneracy"]["status"] == "NOT_RUN"


def test_optional_design_dimensions_accept_exact_values_and_ranges() -> None:
    report = check_mesh(_two_regions(), geometry_backend="off", dimension_spec={
        "absolute_tolerance": 0.1,
        "model": {"size": [6.0, 1.0]},
        "properties": {"7": {"size": {"minimum": [5.5, 0.9], "maximum": [6.5, 1.1]}}},
    })
    assert report.checks["design_dimensions"].status == "PASS"
    assert len(report.dimensions["design_comparison"]["comparisons"]) == 4
    assert all(item["within_limits"] for item in report.dimensions["design_comparison"]["comparisons"])


def test_design_dimension_one_sided_range_stays_json_serializable() -> None:
    report = check_mesh(_two_regions(), geometry_backend="off", dimension_spec={
        "model": {"size": {"minimum": [5.0, 0.5]}},
    })
    assert report.checks["design_dimensions"].status == "PASS"
    assert all(item["allowed_maximum"] is None for item in report.dimensions["design_comparison"]["comparisons"])
    json.dumps(report.to_dict(), allow_nan=False)


def test_design_dimension_error_identifies_property_axis_and_extreme_nodes() -> None:
    report = check_mesh(_two_regions(), geometry_backend="off", dimension_spec={
        "absolute_tolerance": 0.01,
        "properties": {"7": {"size": [5.0, 1.0]}, "8": {"size": [1.0, 1.0]}},
    })
    assert report.checks["design_dimensions"].status == "FINDINGS"
    mismatch = next(item for item in report.findings if item.code == "DIMENSION_OUT_OF_RANGE")
    assert mismatch.property_id == 7
    assert mismatch.measured_value == 6.0
    assert mismatch.node_ids == (101, 105)
    assert any(item.code == "DESIGN_PROPERTY_MISSING" and item.property_id == 8 for item in report.findings)


@pytest.mark.parametrize("spec", [
    {},
    {"properties": {}},
    {"model": {"size": [1.0, 2.0, 3.0]}},
    {"model": {"size": {"minimum": [2.0, 2.0], "maximum": [1.0, 3.0]}}},
    {"model": {"size": [6.0, 1.0]}, "relative_tolerance": -1.0},
])
def test_bad_design_dimension_spec_is_rejected(spec) -> None:
    with pytest.raises(MeshCheckError):
        check_mesh(_two_regions(), geometry_backend="off", dimension_spec=spec)
