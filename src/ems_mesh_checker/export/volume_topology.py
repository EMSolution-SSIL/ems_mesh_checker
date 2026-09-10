"""Validate representative surfaces as closed, orientable Property shells."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class VolumeTopologyConfig:
    """Controls conservative ``Surface Loop`` and ``Volume`` creation."""

    enabled: bool = True
    point_absolute_tolerance: float = 1.0e-10
    point_relative_tolerance: float = 1.0e-8

    def __post_init__(self) -> None:
        if self.point_absolute_tolerance < 0.0 or self.point_relative_tolerance < 0.0:
            raise ValueError("point matching tolerances must be non-negative")


@dataclass(frozen=True)
class SurfaceTopologyRecord:
    """One Property-facing use of a canonical Gmsh surface.

    ``curve_references`` are oriented as seen from the Property. The canonical
    Gmsh surface can have the same or opposite orientation, recorded by
    ``orientation_to_source``.
    """

    property_id: int
    surface_id: int
    orientation_to_source: int
    curve_references: tuple[int, ...]
    property_label: str | None = None

    def __post_init__(self) -> None:
        if self.orientation_to_source not in {-1, 1}:
            raise ValueError("orientation_to_source must be -1 or 1")
        if not self.curve_references:
            raise ValueError("a topology surface must contain boundary curves")
        if any(reference == 0 for reference in self.curve_references):
            raise ValueError("curve references must be non-zero")


@dataclass(frozen=True)
class ClosedSurfaceShell:
    """One connected, orientable, watertight Property shell."""

    property_id: int
    shell_index: int
    surface_references: tuple[int, ...]
    curve_count: int
    property_label: str | None = None


@dataclass(frozen=True)
class OpenSurfaceComponent:
    """A connected surface component deliberately rejected for Volume output."""

    property_id: int
    component_index: int
    surface_ids: tuple[int, ...]
    boundary_curve_ids: tuple[int, ...]
    non_manifold_curve_ids: tuple[int, ...]
    reason: str
    property_label: str | None = None


@dataclass(frozen=True)
class VolumeTopologyResult:
    """Closed shells plus auditable components that could not become volumes."""

    shells: tuple[ClosedSurfaceShell, ...]
    open_components: tuple[OpenSurfaceComponent, ...]
    diagnostics: tuple[str, ...]


def build_volume_topology(
    records: Iterable[SurfaceTopologyRecord],
    *,
    config: VolumeTopologyConfig | None = None,
) -> VolumeTopologyResult:
    """Group Property surfaces and accept only degree-two orientable shells.

    Separate connected components become separate shells, so one Property may
    legitimately produce multiple Volumes. A component is rejected if a curve
    belongs to fewer or more than two surfaces, or if its orientation
    constraints are inconsistent.
    """

    settings = config or VolumeTopologyConfig()
    if not settings.enabled:
        return VolumeTopologyResult((), (), ("volume_topology_disabled",))

    by_property: dict[tuple[int, str], list[SurfaceTopologyRecord]] = defaultdict(list)
    for record in records:
        label = record.property_label or f"PROPERTY_P{record.property_id}"
        by_property[(int(record.property_id), label)].append(record)

    shells: list[ClosedSurfaceShell] = []
    rejected: list[OpenSurfaceComponent] = []
    diagnostics: list[str] = []
    for property_id, property_label in sorted(by_property):
        property_records = sorted(
            by_property[(property_id, property_label)], key=lambda item: item.surface_id
        )
        components = _surface_components(property_records)
        property_shell_index = 0
        for component_index, component in enumerate(components):
            component_records = [property_records[index] for index in component]
            shell, failure = _build_component_shell(
                property_id,
                property_shell_index,
                component_index,
                component_records,
                property_label=property_label,
            )
            if shell is not None:
                shells.append(shell)
                property_shell_index += 1
            else:
                assert failure is not None
                rejected.append(failure)
                diagnostics.append(
                    f"property={property_id} component={component_index} rejected: {failure.reason}"
                )

    return VolumeTopologyResult(tuple(shells), tuple(rejected), tuple(diagnostics))


def _surface_components(records: list[SurfaceTopologyRecord]) -> tuple[tuple[int, ...], ...]:
    curve_owners: dict[int, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        for curve_id in {abs(reference) for reference in record.curve_references}:
            curve_owners[curve_id].append(index)

    adjacency: dict[int, set[int]] = defaultdict(set)
    for owners in curve_owners.values():
        for owner in owners:
            adjacency[owner].update(other for other in owners if other != owner)

    remaining = set(range(len(records)))
    components: list[tuple[int, ...]] = []
    while remaining:
        seed = min(remaining)
        queue = deque([seed])
        component: set[int] = set()
        while queue:
            current = queue.popleft()
            if current in component:
                continue
            component.add(current)
            queue.extend(sorted(adjacency[current] - component))
        remaining.difference_update(component)
        components.append(tuple(sorted(component)))
    return tuple(components)


def _build_component_shell(
    property_id: int,
    shell_index: int,
    component_index: int,
    records: list[SurfaceTopologyRecord],
    *,
    property_label: str | None = None,
) -> tuple[ClosedSurfaceShell | None, OpenSurfaceComponent | None]:
    incidence: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for record_index, record in enumerate(records):
        seen_on_surface: set[int] = set()
        for reference in record.curve_references:
            curve_id = abs(reference)
            if curve_id in seen_on_surface:
                return None, _failure(
                    property_id,
                    component_index,
                    records,
                    (),
                    (curve_id,),
                    "a surface uses the same curve more than once",
                    property_label=property_label,
                )
            seen_on_surface.add(curve_id)
            incidence[curve_id].append((record_index, 1 if reference > 0 else -1))

    boundary = tuple(sorted(curve_id for curve_id, uses in incidence.items() if len(uses) == 1))
    non_manifold = tuple(sorted(curve_id for curve_id, uses in incidence.items() if len(uses) > 2))
    duplicate_surface_ids = len({record.surface_id for record in records}) != len(records)
    if boundary or non_manifold or duplicate_surface_ids:
        parts = []
        if boundary:
            parts.append(f"{len(boundary)} boundary curves")
        if non_manifold:
            parts.append(f"{len(non_manifold)} non-manifold curves")
        if duplicate_surface_ids:
            parts.append("a canonical surface is repeated")
        return None, _failure(
            property_id,
            component_index,
            records,
            boundary,
            non_manifold,
            ", ".join(parts),
            property_label=property_label,
        )

    flips: dict[int, int] = {}
    for seed in range(len(records)):
        if seed in flips:
            continue
        flips[seed] = 1
        queue = deque([seed])
        while queue:
            current = queue.popleft()
            current_signs = {
                abs(reference): 1 if reference > 0 else -1
                for reference in records[current].curve_references
            }
            for curve_id, current_sign in current_signs.items():
                uses = incidence[curve_id]
                if len(uses) != 2:
                    continue
                other, other_sign = uses[0] if uses[1][0] == current else uses[1]
                required = -flips[current] * current_sign * other_sign
                if other in flips and flips[other] != required:
                    return None, _failure(
                        property_id,
                        component_index,
                        records,
                        (),
                        (),
                        "surface orientations are inconsistent",
                        property_label=property_label,
                    )
                if other not in flips:
                    flips[other] = required
                    queue.append(other)

    surface_references = tuple(
        flips[index] * record.orientation_to_source * record.surface_id
        for index, record in enumerate(records)
    )
    return (
        ClosedSurfaceShell(
            property_id=property_id,
            shell_index=shell_index,
            surface_references=surface_references,
            curve_count=len(incidence),
            property_label=property_label,
        ),
        None,
    )


def _failure(
    property_id: int,
    component_index: int,
    records: list[SurfaceTopologyRecord],
    boundary: tuple[int, ...],
    non_manifold: tuple[int, ...],
    reason: str,
    *,
    property_label: str | None = None,
) -> OpenSurfaceComponent:
    return OpenSurfaceComponent(
        property_id=property_id,
        component_index=component_index,
        surface_ids=tuple(record.surface_id for record in records),
        boundary_curve_ids=boundary,
        non_manifold_curve_ids=non_manifold,
        reason=reason,
        property_label=property_label,
    )
