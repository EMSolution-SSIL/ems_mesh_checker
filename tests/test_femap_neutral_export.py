from pathlib import Path

import numpy as np
import pytest

from ems_mesh_checker import CircularArcPrimitive, LinePrimitive
from ems_mesh_checker.export.femap_neutral import (
    FemapNeutralGeometryExportError,
    _write_representative_curve_groups_femap_neutral,
)
from ems_mesh_checker.export.representative_geo import RepresentativeCurveGroup


def _block(lines: list[str], identifier: int) -> list[str]:
    for index in range(len(lines) - 1):
        if lines[index].strip() == "-1" and lines[index + 1].strip() == str(identifier):
            start = index + 2
            end = next(
                current
                for current in range(start, len(lines))
                if lines[current].strip() == "-1"
            )
            return lines[start:end]
    raise AssertionError(f"data block {identifier} was not found")


def test_femap_neutral_writes_shared_line_and_true_arc_geometry(tmp_path: Path) -> None:
    line = LinePrimitive(np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    reversed_line = LinePrimitive(line.end.copy(), line.start.copy())
    arc = CircularArcPrimitive(
        start=np.array([1.0, 0.0, 0.0]),
        center=np.array([0.0, 0.0, 0.0]),
        end=np.array([0.0, 1.0, 0.0]),
        radius=1.0,
        angle_degrees=90.0,
        max_radial_error=0.0,
        source_edge_count=4,
    )
    groups = (
        RepresentativeCurveGroup("PROPERTY_A", (line, arc)),
        RepresentativeCurveGroup("PROPERTY_B", (reversed_line,)),
    )

    output = _write_representative_curve_groups_femap_neutral(
        groups,
        tmp_path / "representative.neu",
    )
    raw = output.read_bytes()
    text = raw.decode("ascii")
    lines = text.splitlines()

    assert b"\r\n" in raw
    assert _block(lines, 100) == [
        "ems_mesh_checker representative curve geometry",
        "10.3,",
    ]
    assert _block(lines, 413) == [
        "1,24,",
        "REPRESENTATIVE_POINTS",
        "2,120,",
        "PROPERTY_A",
        "3,120,",
        "SHARED_PROPERTIES",
    ]

    point_rows = _block(lines, 570)
    assert len(point_rows) == 8
    coordinates = [
        tuple(float(value) for value in point_rows[index].split(",")[:3])
        for index in range(1, len(point_rows), 2)
    ]
    assert coordinates[0] == pytest.approx((0.0, 0.0, 0.0))
    assert coordinates[1] == pytest.approx((1.0, 0.0, 0.0))
    assert coordinates[2] == pytest.approx((2**-0.5, 2**-0.5, 0.0))
    assert coordinates[3] == pytest.approx((0.0, 1.0, 0.0))

    curve_rows = _block(lines, 571)
    assert len(curve_rows) == 12
    assert curve_rows[0].split(",")[:4] == ["1", "120", "0", "3"]
    assert curve_rows[5] == "1,2,0,0,0,"
    assert curve_rows[6].split(",")[:4] == ["2", "120", "1", "2"]
    assert curve_rows[11] == "1,2,3,4,0,"
    assert max(map(len, lines)) <= 255


def test_femap_neutral_rejects_unoriented_semicircle(tmp_path: Path) -> None:
    arc = CircularArcPrimitive(
        start=np.array([1.0, 0.0, 0.0]),
        center=np.array([0.0, 0.0, 0.0]),
        end=np.array([-1.0, 0.0, 0.0]),
        radius=1.0,
        angle_degrees=180.0,
        max_radial_error=0.0,
        source_edge_count=8,
    )

    with pytest.raises(FemapNeutralGeometryExportError, match="180-degree"):
        _write_representative_curve_groups_femap_neutral(
            (RepresentativeCurveGroup("PROPERTY_A", (arc,)),),
            tmp_path / "semicircle.neu",
        )


def test_femap_neutral_keeps_both_oriented_semicircles(tmp_path: Path) -> None:
    upper = CircularArcPrimitive(
        start=np.array([1.0, 0.0, 0.0]),
        center=np.array([0.0, 0.0, 0.0]),
        end=np.array([-1.0, 0.0, 0.0]),
        radius=1.0,
        angle_degrees=180.0,
        max_radial_error=0.0,
        source_edge_count=8,
        plane_normal=np.array([0.0, 0.0, 1.0]),
    )
    lower = CircularArcPrimitive(
        start=upper.start.copy(),
        center=upper.center.copy(),
        end=upper.end.copy(),
        radius=1.0,
        angle_degrees=180.0,
        max_radial_error=0.0,
        source_edge_count=8,
        plane_normal=np.array([0.0, 0.0, -1.0]),
    )

    output = _write_representative_curve_groups_femap_neutral(
        (RepresentativeCurveGroup("PROPERTY_A", (upper, lower)),),
        tmp_path / "circle.neu",
    )
    lines = output.read_text(encoding="ascii").splitlines()

    assert len(_block(lines, 571)) == 12
    point_rows = _block(lines, 570)
    coordinates = [
        tuple(float(value) for value in point_rows[index].split(",")[:3])
        for index in range(1, len(point_rows), 2)
    ]
    assert any(coordinate == pytest.approx((0.0, 1.0, 0.0)) for coordinate in coordinates)
    assert any(coordinate == pytest.approx((0.0, -1.0, 0.0)) for coordinate in coordinates)
