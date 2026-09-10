"""Reconstruct extruded circular side surfaces from planar strip regions."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sin
from typing import Iterable

import numpy as np

from .curve_simplification import (
    CircularArcPrimitive,
    CurvePrimitive,
    CurveSimplificationConfig,
    LinePrimitive,
    simplify_planar_loop,
)
from .planar_regions import PlanarRegion, PlanarRegionResult


@dataclass(frozen=True)
class RuledSurfaceConfig:
    """Controls conservative matching of cap arcs and extruded side strips."""

    enabled: bool = True
    axis_alignment_angle_degrees: float = 1.0
    match_absolute_tolerance: float = 1.0e-10
    match_relative_tolerance: float = 1.0e-5
    minimum_side_regions: int = 3
    max_single_strip_arc_angle_degrees: float = 45.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.axis_alignment_angle_degrees < 90.0:
            raise ValueError("axis_alignment_angle_degrees must be in [0, 90)")
        if self.match_absolute_tolerance < 0.0 or self.match_relative_tolerance < 0.0:
            raise ValueError("surface matching tolerances must be non-negative")
        if self.minimum_side_regions < 1:
            raise ValueError("minimum_side_regions must be positive")
        if not 0.0 < self.max_single_strip_arc_angle_degrees < 180.0:
            raise ValueError("max_single_strip_arc_angle_degrees must be in (0, 180)")


@dataclass(frozen=True)
class RuledSurfacePatch:
    """One four-curve surface replacing a chain of planar side strips."""

    patch_index: int
    source_region_indices: tuple[int, ...]
    cap_region_indices: tuple[int, int]
    primitives: tuple[CurvePrimitive, CurvePrimitive, CurvePrimitive, CurvePrimitive]
    radius: float
    extrusion_vector: np.ndarray
    max_match_error: float


@dataclass(frozen=True)
class RuledSurfaceResult:
    """Detected curved patches and the planar regions they replace."""

    patches: tuple[RuledSurfacePatch, ...]
    replaced_region_indices: tuple[int, ...]
    diagnostics: tuple[str, ...]


@dataclass(frozen=True)
class _ArcRecord:
    region_index: int | None
    normal: np.ndarray
    arc: CircularArcPrimitive


@dataclass(frozen=True)
class _PatchCandidate:
    first: _ArcRecord
    second: _ArcRecord
    source_regions: tuple[int, ...]
    primitives: tuple[CurvePrimitive, CurvePrimitive, CurvePrimitive, CurvePrimitive]
    extrusion_vector: np.ndarray
    max_match_error: float


@dataclass(frozen=True)
class _SideStripRecord:
    region_index: int
    axial_low: float
    axial_high: float


def reconstruct_ruled_surfaces(
    result: PlanarRegionResult,
    *,
    curve_config: CurveSimplificationConfig | None = None,
    config: RuledSurfaceConfig | None = None,
    arc_templates: Iterable[CircularArcPrimitive] | None = None,
) -> RuledSurfaceResult:
    """Pair translated cap arcs and replace their planar side-strip chain.

    A patch is accepted only when every source strip is a quadrilateral whose
    vertices lie on the same cylinder at the two cap levels. The strip count
    must equal the sampled edge count of both arcs, preventing partial or
    ambiguous reconstruction.
    """

    settings = config or RuledSurfaceConfig()
    if not settings.enabled:
        return RuledSurfaceResult((), (), ("ruled_surface_reconstruction_disabled",))
    curve_settings = curve_config or CurveSimplificationConfig()
    if not curve_settings.enabled:
        return RuledSurfaceResult((), (), ("arc_simplification_disabled",))
    if not result.regions:
        return RuledSurfaceResult((), (), ())

    point_groups = [region.outer_loop.points for region in result.regions]
    point_groups.extend(hole.points for region in result.regions for hole in region.hole_loops)
    all_points = np.vstack(point_groups)
    model_extent = float(np.linalg.norm(np.ptp(all_points, axis=0)))
    tolerance = max(
        settings.match_absolute_tolerance,
        model_extent * settings.match_relative_tolerance,
    )
    arcs = (
        _arc_records(result, curve_settings)
        if arc_templates is None
        else _arc_template_records(tuple(arc_templates))
    )
    candidates: list[_PatchCandidate] = []
    for first_index, first in enumerate(arcs):
        for second in arcs[first_index + 1 :]:
            candidates.extend(
                _match_arc_pair(first, second, result.regions, tolerance, settings)
            )
    candidates.extend(
        _single_strip_candidates(arcs, result.regions, tolerance, settings)
    )

    selected: list[_PatchCandidate] = []
    occupied_regions: set[int] = set()
    for candidate in sorted(
        candidates,
        key=lambda item: (-len(item.source_regions), item.max_match_error, item.source_regions),
    ):
        if occupied_regions.isdisjoint(candidate.source_regions):
            selected.append(candidate)
            occupied_regions.update(candidate.source_regions)

    patches = tuple(
        RuledSurfacePatch(
            patch_index=index,
            source_region_indices=candidate.source_regions,
            cap_region_indices=(
                -1 if candidate.first.region_index is None else candidate.first.region_index,
                -1 if candidate.second.region_index is None else candidate.second.region_index,
            ),
            primitives=candidate.primitives,
            radius=candidate.first.arc.radius,
            extrusion_vector=candidate.extrusion_vector,
            max_match_error=candidate.max_match_error,
        )
        for index, candidate in enumerate(
            sorted(
                selected,
                key=lambda item: (
                    -1 if item.first.region_index is None else item.first.region_index,
                    item.source_regions,
                ),
            )
        )
    )
    diagnostics = ()
    if arcs and not patches:
        diagnostics = (f"unpaired_circular_arcs={len(arcs)}",)
    return RuledSurfaceResult(patches, tuple(sorted(occupied_regions)), diagnostics)


def _single_strip_candidates(
    arcs: tuple[_ArcRecord, ...],
    regions: tuple[PlanarRegion, ...],
    tolerance: float,
    config: RuledSurfaceConfig,
) -> tuple[_PatchCandidate, ...]:
    """Recover isolated cylindrical quads from a known circular-arc family.

    Full-chain matching remains preferred. This fallback handles a side strip
    whose neighboring Property seams prevent the complete sampled arc chain
    from appearing in one region set.
    """

    candidates: list[_PatchCandidate] = []
    for region in regions:
        if region.hole_loops:
            continue
        points = np.asarray(region.outer_loop.points, dtype=float)
        if len(points) != 4:
            continue
        best: _PatchCandidate | None = None
        for offset in (0, 1):
            ordered = np.roll(points, -offset, axis=0)
            lower_start, lower_end, upper_end, upper_start = ordered
            first_translation = upper_start - lower_start
            second_translation = upper_end - lower_end
            translation_error = float(
                np.linalg.norm(first_translation - second_translation)
            )
            length = float(np.linalg.norm(first_translation))
            if length <= tolerance or translation_error > tolerance:
                continue
            axis = first_translation / length
            if abs(float(np.dot(region.normal, axis))) > sin(
                radians(config.axis_alignment_angle_degrees)
            ):
                continue
            for record in arcs:
                if abs(float(np.dot(axis, record.normal))) < cos(
                    radians(config.axis_alignment_angle_degrees)
                ):
                    continue
                candidate = _single_strip_candidate(
                    record,
                    region,
                    lower_start,
                    lower_end,
                    upper_start,
                    upper_end,
                    first_translation,
                    axis,
                    translation_error,
                    tolerance,
                    config,
                )
                if candidate is not None and (
                    best is None or candidate.max_match_error < best.max_match_error
                ):
                    best = candidate
        if best is not None:
            candidates.append(best)
    return tuple(candidates)


def _single_strip_candidate(
    record: _ArcRecord,
    region: PlanarRegion,
    lower_start: np.ndarray,
    lower_end: np.ndarray,
    upper_start: np.ndarray,
    upper_end: np.ndarray,
    translation: np.ndarray,
    axis: np.ndarray,
    translation_error: float,
    tolerance: float,
    config: RuledSurfaceConfig,
) -> _PatchCandidate | None:
    template = record.arc
    lower_center = template.center + axis * float(
        np.dot(lower_start - template.center, axis)
    )
    endpoint_radii = np.array(
        [
            np.linalg.norm(lower_start - lower_center),
            np.linalg.norm(lower_end - lower_center),
            np.linalg.norm(upper_start - (lower_center + translation)),
            np.linalg.norm(upper_end - (lower_center + translation)),
        ]
    )
    radial_error = float(np.max(np.abs(endpoint_radii - template.radius)))
    recovery_tolerance = max(
        tolerance,
        2.0 * template.max_radial_error,
        template.radius * 1.0e-4,
    )
    if radial_error > recovery_tolerance:
        return None

    family_template = CircularArcPrimitive(
        start=template.start + (lower_center - template.center),
        center=lower_center,
        end=template.end + (lower_center - template.center),
        radius=template.radius,
        angle_degrees=template.angle_degrees,
        max_radial_error=template.max_radial_error,
        source_edge_count=1,
    )
    lower_arc = _arc_with_exact_endpoints(
        family_template,
        lower_start,
        lower_end,
        axis,
    )
    if lower_arc is None or (
        lower_arc.angle_degrees > config.max_single_strip_arc_angle_degrees
    ):
        return None
    lower_arc = CircularArcPrimitive(
        start=lower_arc.start,
        center=lower_arc.center,
        end=lower_arc.end,
        radius=lower_arc.radius,
        angle_degrees=lower_arc.angle_degrees,
        max_radial_error=max(lower_arc.max_radial_error, radial_error),
        source_edge_count=1,
    )
    upper_arc = _translated_arc(lower_arc, translation)
    primitives: tuple[CurvePrimitive, CurvePrimitive, CurvePrimitive, CurvePrimitive] = (
        lower_arc,
        LinePrimitive(lower_arc.end.copy(), upper_arc.end.copy()),
        _reverse_arc(upper_arc),
        LinePrimitive(upper_arc.start.copy(), lower_arc.start.copy()),
    )
    return _PatchCandidate(
        first=_ArcRecord(None, axis.copy(), lower_arc),
        second=_ArcRecord(None, axis.copy(), upper_arc),
        source_regions=(region.region_index,),
        primitives=primitives,
        extrusion_vector=translation.copy(),
        max_match_error=max(radial_error, translation_error),
    )


def _arc_records(
    result: PlanarRegionResult,
    curve_config: CurveSimplificationConfig,
) -> tuple[_ArcRecord, ...]:
    records: list[_ArcRecord] = []
    for region in result.regions:
        for loop in (region.outer_loop, *region.hole_loops):
            simplified = simplify_planar_loop(loop, normal=region.normal, config=curve_config)
            records.extend(
                _ArcRecord(region.region_index, region.normal, primitive)
                for primitive in simplified.primitives
                if isinstance(primitive, CircularArcPrimitive)
            )
    return tuple(records)


def _arc_template_records(
    arcs: tuple[CircularArcPrimitive, ...],
) -> tuple[_ArcRecord, ...]:
    records: list[_ArcRecord] = []
    for arc in arcs:
        first = arc.start - arc.center
        second = arc.end - arc.center
        normal = np.cross(first, second)
        length = float(np.linalg.norm(normal))
        if length > 0.0:
            records.append(_ArcRecord(None, normal / length, arc))
    return tuple(records)


def _match_arc_pair(
    first: _ArcRecord,
    second: _ArcRecord,
    regions: tuple[PlanarRegion, ...],
    tolerance: float,
    config: RuledSurfaceConfig,
) -> tuple[_PatchCandidate, ...]:
    if first.region_index is not None and first.region_index == second.region_index:
        return ()
    arc_a, arc_b = first.arc, second.arc
    radius_error = abs(arc_a.radius - arc_b.radius)
    if radius_error > tolerance:
        return ()
    translation = arc_b.center - arc_a.center
    length = float(np.linalg.norm(translation))
    if length <= tolerance:
        return ()
    axis = translation / length
    alignment = cos(radians(config.axis_alignment_angle_degrees))
    if abs(float(np.dot(axis, first.normal))) < alignment:
        return ()
    if abs(float(np.dot(axis, second.normal))) < alignment:
        return ()

    direct_error = max(
        float(np.linalg.norm(arc_b.start - (arc_a.start + translation))),
        float(np.linalg.norm(arc_b.end - (arc_a.end + translation))),
    )
    reverse_error = max(
        float(np.linalg.norm(arc_b.end - (arc_a.start + translation))),
        float(np.linalg.norm(arc_b.start - (arc_a.end + translation))),
    )
    endpoint_error = min(direct_error, reverse_error)
    if endpoint_error > tolerance:
        return ()
    side_region_groups = _cylindrical_side_region_groups(
        regions,
        excluded={
            index for index in (first.region_index, second.region_index) if index is not None
        },
        arc=arc_a,
        second_center=arc_b.center,
        axis=axis,
        length=length,
        tolerance=tolerance,
        alignment_angle=config.axis_alignment_angle_degrees,
    )
    expected_strips = arc_a.source_edge_count
    if arc_b.source_edge_count != expected_strips:
        return ()
    complete_groups = tuple(
        group
        for group in side_region_groups
        if len(group) >= config.minimum_side_regions and len(group) == expected_strips
    )
    if not _groups_span_extrusion(complete_groups, length, tolerance):
        return ()

    candidates: list[_PatchCandidate] = []
    for group in complete_groups:
        low = group[0].axial_low
        high = group[0].axial_high
        lower_arc = _snap_arc_to_side_group(
            _translated_arc(arc_a, axis * low),
            group,
            regions,
            axis,
            tolerance,
        )
        upper_arc = _snap_arc_to_side_group(
            _translated_arc(arc_a, axis * high),
            group,
            regions,
            axis,
            tolerance,
        )
        if lower_arc is None or upper_arc is None:
            continue
        primitives: tuple[CurvePrimitive, CurvePrimitive, CurvePrimitive, CurvePrimitive] = (
            lower_arc,
            LinePrimitive(lower_arc.end.copy(), upper_arc.end.copy()),
            _reverse_arc(upper_arc),
            LinePrimitive(upper_arc.start.copy(), lower_arc.start.copy()),
        )
        candidates.append(
            _PatchCandidate(
                first=first,
                second=second,
                source_regions=tuple(item.region_index for item in group),
                primitives=primitives,
                extrusion_vector=axis * (high - low),
                max_match_error=max(radius_error, endpoint_error),
            )
        )
    return tuple(candidates)


def _cylindrical_side_region_groups(
    regions: tuple[PlanarRegion, ...],
    *,
    excluded: set[int],
    arc: CircularArcPrimitive,
    second_center: np.ndarray,
    axis: np.ndarray,
    length: float,
    tolerance: float,
    alignment_angle: float,
) -> tuple[tuple[_SideStripRecord, ...], ...]:
    selected: list[_SideStripRecord] = []
    perpendicular_limit = sin(radians(alignment_angle))
    for region in regions:
        if region.region_index in excluded or region.hole_loops:
            continue
        points = np.asarray(region.outer_loop.points, dtype=float)
        if len(points) != 4:
            continue
        if abs(float(np.dot(region.normal, axis))) > perpendicular_limit:
            continue
        axial = (points - arc.center) @ axis
        if float(np.min(axial)) < -tolerance or float(np.max(axial)) > length + tolerance:
            continue
        axial_levels = _two_axial_levels(axial, tolerance)
        if axial_levels is None:
            continue
        axial_low, axial_high, low, high = axial_levels
        radial_vectors = (points - arc.center) - axial[:, None] * axis
        radii = np.linalg.norm(radial_vectors, axis=1)
        if float(np.max(np.abs(radii - arc.radius))) > tolerance:
            continue
        low_points = points[low]
        high_points = points[high]
        translation = axis * (axial_high - axial_low)
        if not _translated_pair_matches(low_points, high_points, translation, tolerance):
            continue
        if not all(
            _inside_minor_arc(vector / np.linalg.norm(vector), arc, tolerance)
            for vector in radial_vectors
        ):
            continue
        selected.append(
            _SideStripRecord(
                region_index=region.region_index,
                axial_low=axial_low,
                axial_high=axial_high,
            )
        )
    return _group_side_strips(selected, tolerance)


def _two_axial_levels(
    axial: np.ndarray,
    tolerance: float,
) -> tuple[float, float, np.ndarray, np.ndarray] | None:
    low_value = float(np.min(axial))
    high_value = float(np.max(axial))
    if high_value - low_value <= tolerance:
        return None
    low = np.abs(axial - low_value) <= tolerance
    high = np.abs(axial - high_value) <= tolerance
    if int(np.sum(low)) != 2 or int(np.sum(high)) != 2 or np.any(~(low | high)):
        return None
    return float(np.mean(axial[low])), float(np.mean(axial[high])), low, high


def _group_side_strips(
    strips: list[_SideStripRecord],
    tolerance: float,
) -> tuple[tuple[_SideStripRecord, ...], ...]:
    groups: list[list[_SideStripRecord]] = []
    for strip in sorted(strips, key=lambda item: (item.axial_low, item.axial_high, item.region_index)):
        matching = next(
            (
                group
                for group in groups
                if abs(group[0].axial_low - strip.axial_low) <= tolerance
                and abs(group[0].axial_high - strip.axial_high) <= tolerance
            ),
            None,
        )
        if matching is None:
            groups.append([strip])
        else:
            matching.append(strip)
    return tuple(
        tuple(sorted(group, key=lambda item: item.region_index))
        for group in sorted(groups, key=lambda item: (item[0].axial_low, item[0].axial_high))
    )


def _groups_span_extrusion(
    groups: tuple[tuple[_SideStripRecord, ...], ...],
    length: float,
    tolerance: float,
) -> bool:
    if not groups:
        return False
    current = 0.0
    for group in groups:
        low = group[0].axial_low
        high = group[0].axial_high
        if abs(low - current) > tolerance or high <= low + tolerance:
            return False
        current = high
    return abs(current - length) <= tolerance


def _translated_pair_matches(
    low_points: np.ndarray,
    high_points: np.ndarray,
    translation: np.ndarray,
    tolerance: float,
) -> bool:
    remaining = list(range(len(high_points)))
    for point in low_points:
        errors = [float(np.linalg.norm(high_points[index] - (point + translation))) for index in remaining]
        best = int(np.argmin(errors))
        if errors[best] > tolerance:
            return False
        del remaining[best]
    return True


def _translated_arc(arc: CircularArcPrimitive, translation: np.ndarray) -> CircularArcPrimitive:
    return CircularArcPrimitive(
        start=arc.start + translation,
        center=arc.center + translation,
        end=arc.end + translation,
        radius=arc.radius,
        angle_degrees=arc.angle_degrees,
        max_radial_error=arc.max_radial_error,
        source_edge_count=arc.source_edge_count,
    )


def _snap_arc_to_side_group(
    template: CircularArcPrimitive,
    group: tuple[_SideStripRecord, ...],
    regions: tuple[PlanarRegion, ...],
    axis: np.ndarray,
    tolerance: float,
) -> CircularArcPrimitive | None:
    by_index = {region.region_index: region for region in regions}
    points = np.vstack(
        [by_index[item.region_index].outer_loop.points for item in group]
    )
    start = points[int(np.argmin(np.linalg.norm(points - template.start, axis=1)))]
    end = points[int(np.argmin(np.linalg.norm(points - template.end, axis=1)))]
    recovery_tolerance = max(
        tolerance,
        2.0 * template.max_radial_error,
        template.radius * 1.0e-4,
    )
    if (
        np.linalg.norm(start - template.start) > recovery_tolerance
        or np.linalg.norm(end - template.end) > recovery_tolerance
    ):
        return None
    return _arc_with_exact_endpoints(template, start, end, axis)


def _arc_with_exact_endpoints(
    template: CircularArcPrimitive,
    start: np.ndarray,
    end: np.ndarray,
    normal: np.ndarray,
) -> CircularArcPrimitive | None:
    chord = end - start
    chord_length = float(np.linalg.norm(chord))
    perpendicular = np.cross(normal, chord)
    perpendicular_length = float(np.linalg.norm(perpendicular))
    if chord_length <= 0.0 or perpendicular_length <= 0.0:
        return None
    perpendicular /= perpendicular_length
    midpoint = 0.5 * (start + end)
    center = midpoint + float(np.dot(template.center - midpoint, perpendicular)) * perpendicular
    first = start - center
    second = end - center
    radius = float(np.linalg.norm(first))
    if radius <= 0.0:
        return None
    angle = float(
        np.degrees(
            np.arccos(
                np.clip(np.dot(first, second) / (radius * radius), -1.0, 1.0)
            )
        )
    )
    if not 0.0 < angle < 180.0:
        return None
    return CircularArcPrimitive(
        start=start.copy(),
        center=center,
        end=end.copy(),
        radius=radius,
        angle_degrees=angle,
        max_radial_error=template.max_radial_error,
        source_edge_count=template.source_edge_count,
    )


def _inside_minor_arc(
    radial: np.ndarray,
    arc: CircularArcPrimitive,
    tolerance: float,
) -> bool:
    start = (arc.start - arc.center) / arc.radius
    end = (arc.end - arc.center) / arc.radius
    span = radians(arc.angle_degrees)
    first = _angle_between(start, radial)
    second = _angle_between(radial, end)
    angular_tolerance = max(tolerance / max(arc.radius, tolerance), 1.0e-8)
    return first <= span + angular_tolerance and second <= span + angular_tolerance and abs(
        first + second - span
    ) <= angular_tolerance


def _angle_between(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.arccos(np.clip(np.dot(first, second), -1.0, 1.0)))


def _reverse_arc(arc: CircularArcPrimitive) -> CircularArcPrimitive:
    return CircularArcPrimitive(
        start=arc.end.copy(),
        center=arc.center.copy(),
        end=arc.start.copy(),
        radius=arc.radius,
        angle_degrees=arc.angle_degrees,
        max_radial_error=arc.max_radial_error,
        source_edge_count=arc.source_edge_count,
    )
