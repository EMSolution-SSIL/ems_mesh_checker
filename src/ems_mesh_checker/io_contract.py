"""Validation of the ``meshio.Mesh`` contract consumed by the checker.

File readers belong to :mod:`ems_file_format_converter`. Keeping this module
reader-agnostic makes the topology code usable with every meshio-compatible source.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class MeshContractError(ValueError):
    """Raised when a mesh cannot safely be consumed by the checker."""


@dataclass(frozen=True)
class MeshContractSummary:
    """A deterministic description of a validated mesh input."""

    num_points: int
    num_cells: int
    cell_types: tuple[tuple[str, int], ...]
    has_point_ids: bool
    has_element_ids: bool
    has_property_ids: bool


def _as_one_dimensional_integer_array(value: Any, *, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise MeshContractError(f"{label} must be a one-dimensional array")
    if not np.issubdtype(array.dtype, np.integer):
        raise MeshContractError(f"{label} must contain integer values")
    return array


def _cell_data_blocks(mesh: Any, name: str) -> list[Any] | None:
    values = getattr(mesh, "cell_data", {}).get(name)
    if values is None:
        return None
    if len(values) != len(mesh.cells):
        raise MeshContractError(
            f'cell_data["{name}"] has {len(values)} blocks; expected {len(mesh.cells)}'
        )
    return values


def _validate_cell_data(mesh: Any, name: str, *, required: bool) -> bool:
    blocks = _cell_data_blocks(mesh, name)
    if blocks is None:
        if required:
            raise MeshContractError(f'cell_data["{name}"] is required')
        return False

    for block_index, (cell_block, values) in enumerate(zip(mesh.cells, blocks, strict=True)):
        array = _as_one_dimensional_integer_array(
            values, label=f'cell_data["{name}"][{block_index}]'
        )
        if len(array) != len(cell_block.data):
            raise MeshContractError(
                f'cell_data["{name}"][{block_index}] has {len(array)} values; '
                f"expected {len(cell_block.data)}"
            )
    return True


def inspect_mesh_contract(
    mesh: Any,
    *,
    require_point_ids: bool = True,
    require_element_ids: bool = True,
    require_property_ids: bool = True,
) -> MeshContractSummary:
    """Validate the meshio contract required by boundary extraction.

    ``property_id`` may be optional in future extraction modes, but Phase 0 requires
    it so converter regressions cannot silently remove property metadata.
    """

    if not hasattr(mesh, "points") or not hasattr(mesh, "cells"):
        raise MeshContractError("mesh must provide points and cells")

    points = np.asarray(mesh.points)
    if points.ndim != 2 or points.shape[1] not in (2, 3):
        raise MeshContractError("points must have shape (n, 2) or (n, 3)")

    point_ids = getattr(mesh, "point_data", {}).get("id")
    if point_ids is None:
        if require_point_ids:
            raise MeshContractError('point_data["id"] is required')
        has_point_ids = False
    else:
        point_id_array = _as_one_dimensional_integer_array(point_ids, label='point_data["id"]')
        if len(point_id_array) != len(points):
            raise MeshContractError(
                f'point_data["id"] has {len(point_id_array)} values; expected {len(points)}'
            )
        if len(np.unique(point_id_array)) != len(point_id_array):
            raise MeshContractError('point_data["id"] must contain unique original node IDs')
        has_point_ids = True

    cell_type_counts: dict[str, int] = {}
    for block_index, cell_block in enumerate(mesh.cells):
        connectivity = np.asarray(cell_block.data)
        if connectivity.ndim != 2:
            raise MeshContractError(f"cell block {block_index} connectivity must be two-dimensional")
        if not np.issubdtype(connectivity.dtype, np.integer):
            raise MeshContractError(f"cell block {block_index} connectivity must contain integers")
        if connectivity.size and (connectivity.min() < 0 or connectivity.max() >= len(points)):
            raise MeshContractError(
                f"cell block {block_index} connectivity contains a point index outside [0, {len(points)})"
            )
        cell_type = str(cell_block.type)
        cell_type_counts[cell_type] = cell_type_counts.get(cell_type, 0) + len(connectivity)

    has_element_ids = _validate_cell_data(mesh, "element_id", required=require_element_ids)
    has_property_ids = _validate_cell_data(mesh, "property_id", required=require_property_ids)

    cell_types = tuple(sorted(cell_type_counts.items()))
    return MeshContractSummary(
        num_points=len(points),
        num_cells=sum(count for _, count in cell_types),
        cell_types=cell_types,
        has_point_ids=has_point_ids,
        has_element_ids=has_element_ids,
        has_property_ids=has_property_ids,
    )
