from __future__ import annotations

import numpy as np
import pytest

from ems_mesh_checker import (
    CircularArcPrimitive,
    CurveSimplificationConfig,
    LinePrimitive,
    PlanarLoop,
    simplify_planar_loop,
)


def _loop(points: np.ndarray) -> PlanarLoop:
    return PlanarLoop(np.asarray(points, dtype=float), signed_area=1.0, is_hole=False)


def test_quarter_circle_samples_become_one_arc_and_two_lines() -> None:
    angles = np.radians(np.arange(0.0, 91.0, 18.0))
    arc = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(len(angles))))
    points = np.vstack(([0.0, 0.0, 0.0], arc))

    result = simplify_planar_loop(_loop(points), normal=np.array([0.0, 0.0, 1.0]))

    assert result.original_edge_count == 7
    assert result.arc_count == 1
    assert result.line_count == 2
    circular_arc = next(item for item in result.primitives if isinstance(item, CircularArcPrimitive))
    np.testing.assert_allclose(circular_arc.center, [0.0, 0.0, 0.0], atol=1.0e-12)
    assert circular_arc.radius == pytest.approx(1.0)
    assert circular_arc.angle_degrees == pytest.approx(90.0)
    assert circular_arc.source_edge_count == 5
    assert np.linalg.norm(circular_arc.start - circular_arc.center) == pytest.approx(
        np.linalg.norm(circular_arc.end - circular_arc.center)
    )


def test_square_is_not_misidentified_as_a_circle() -> None:
    square = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    )

    result = simplify_planar_loop(_loop(square), normal=np.array([0.0, 0.0, 1.0]))

    assert result.arc_count == 0
    assert result.line_count == 4
    assert all(isinstance(item, LinePrimitive) for item in result.primitives)


def test_full_circle_is_split_into_gmsh_safe_arcs() -> None:
    angles = np.radians(np.arange(0.0, 360.0, 15.0))
    points = np.column_stack((2.0 * np.cos(angles), 2.0 * np.sin(angles), np.zeros(len(angles))))

    result = simplify_planar_loop(_loop(points), normal=np.array([0.0, 0.0, 1.0]))

    assert result.line_count == 0
    assert result.arc_count >= 3
    assert sum(item.source_edge_count for item in result.primitives) == len(points)
    assert all(
        isinstance(item, CircularArcPrimitive) and item.angle_degrees < 180.0
        for item in result.primitives
    )


def test_arc_recognition_can_be_disabled_for_exact_polyline_output() -> None:
    angles = np.radians(np.arange(0.0, 91.0, 18.0))
    points = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(len(angles))))
    result = simplify_planar_loop(
        _loop(points),
        normal=np.array([0.0, 0.0, 1.0]),
        config=CurveSimplificationConfig(enabled=False),
    )

    assert result.arc_count == 0
    assert result.line_count == len(points)


def test_arc_recognition_requires_at_least_four_points() -> None:
    with pytest.raises(ValueError, match="min_arc_points"):
        CurveSimplificationConfig(min_arc_points=3)
