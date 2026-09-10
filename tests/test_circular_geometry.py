from math import cos, radians, sin

import numpy as np
import pytest

from ems_mesh_checker.geometry.circular_geometry import (
    CircularGeometryNormalizer,
    LabeledPrimitive,
)
from ems_mesh_checker.geometry.curve_simplification import CircularArcPrimitive


def _arc(
    start_degrees: float,
    end_degrees: float,
    *,
    center_offset: tuple[float, float],
) -> CircularArcPrimitive:
    radius = 0.016
    start = radians(start_degrees)
    end = radians(end_degrees)
    center = np.array([center_offset[0], center_offset[1], 0.0])
    return CircularArcPrimitive(
        start=np.array([radius * cos(start), radius * sin(start), 0.0]),
        center=center,
        end=np.array([radius * cos(end), radius * sin(end), 0.0]),
        radius=float(np.linalg.norm(np.array([radius * cos(start), radius * sin(start), 0.0]) - center)),
        angle_degrees=end_degrees - start_degrees,
        max_radial_error=5.0e-7,
        source_edge_count=max(1, int(round((end_degrees - start_degrees) / 22.5))),
        plane_normal=np.array([0.0, 0.0, 1.0]),
    )


def test_overlapping_noisy_arc_runs_become_one_semicircle() -> None:
    arcs = [
        ("SHARED_PROPERTIES", _arc(22.5, 180.0, center_offset=(-0.2e-6, 0.2e-6))),
        ("PROPERTY_P122", _arc(0.0, 157.5, center_offset=(0.2e-6, 0.2e-6))),
    ]
    offsets = (
        (-0.4e-6, 0.6e-6),
        (-0.6e-6, 0.8e-6),
        (2.4e-6, -0.3e-6),
        (-2.4e-6, -0.2e-6),
        (0.6e-6, 0.8e-6),
        (0.4e-6, 0.6e-6),
    )
    for index, offset in enumerate(offsets):
        arcs.append(("PROPERTY_P102", _arc(22.5 * (index + 1), 22.5 * (index + 2), center_offset=offset)))

    normalizer = CircularGeometryNormalizer(
        (arc for _, arc in arcs),
        point_tolerance=2.0e-8,
    )
    consolidated = normalizer.consolidate_labeled(
        LabeledPrimitive(arc, frozenset((label,))) for label, arc in arcs
    )

    assert len(consolidated) == 1
    result = consolidated[0]
    assert isinstance(result.primitive, CircularArcPrimitive)
    assert result.primitive.angle_degrees == pytest.approx(180.0)
    assert result.primitive.center == pytest.approx((0.0, 0.0, 0.0), abs=1.0e-12)
    assert result.primitive.radius == pytest.approx(0.016, abs=1.0e-12)
    assert result.labels == frozenset(("SHARED_PROPERTIES", "PROPERTY_P102", "PROPERTY_P122"))


def test_nearby_points_on_a_canonical_circle_share_one_vertex() -> None:
    arc = _arc(0.0, 90.0, center_offset=(0.3e-6, -0.2e-6))
    normalizer = CircularGeometryNormalizer((arc,), point_tolerance=2.0e-8)
    angle = radians(22.5)
    first = np.array([0.016 * cos(angle), 0.016 * sin(angle), 0.0])
    second_angle = angle + radians(0.001)
    second = np.array([0.016 * cos(second_angle), 0.016 * sin(second_angle), 0.0])
    radial_offset = 2.0e-5 * np.array([cos(angle), sin(angle), 0.0])

    assert normalizer.circular_points_coincident(first, second)
    assert not normalizer.circular_points_coincident(first, first + radial_offset)
