"""Build a deterministic face table from a meshio-compatible mesh."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Iterator, Mapping

import numpy as np

from .element_topology import get_element_topology
from .face import FaceOwner, FaceRecord


class TopologyError(ValueError):
    """Raised when connectivity or metadata cannot form a topology table."""


class FaceTable:
    """Map canonical corner-node keys to all owning mesh elements.

    If a mesh mixes dimensions, only the highest supported dimension is enumerated.
    Thus a 3-D mesh ignores lower-dimensional boundary elements, while a 2-D mesh
    ignores line elements and enumerates triangle/quad edges.
    """

    def __init__(
        self,
        records: dict[tuple[int, ...], FaceRecord],
        *,
        source_dimension: int,
        num_source_elements: int,
    ) -> None:
        self._records = records
        self.source_dimension = source_dimension
        self.num_source_elements = num_source_elements

    @property
    def records(self) -> Mapping[tuple[int, ...], FaceRecord]:
        """Read-only mapping of canonical keys to face records."""

        return MappingProxyType(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def get(self, node_indices: tuple[int, ...] | list[int]) -> FaceRecord | None:
        """Look up an entity without requiring its input node order."""

        return self._records.get(tuple(sorted(node_indices)))

    def iter_records(self) -> Iterator[FaceRecord]:
        """Iterate face records in a stable canonical-key order."""

        for key in sorted(self._records):
            yield self._records[key]

    @classmethod
    def from_mesh(cls, mesh: Any, *, dimension: int | None = None) -> "FaceTable":
        """Enumerate supported boundary entities and their metadata owners."""

        if not hasattr(mesh, "points") or not hasattr(mesh, "cells"):
            raise TopologyError("mesh must provide points and cells")

        points = np.asarray(mesh.points)
        if points.ndim != 2:
            raise TopologyError("mesh points must be a two-dimensional array")

        supported_dimensions = {
            topology.dimension
            for block in mesh.cells
            if (topology := get_element_topology(str(block.type))) is not None
        }
        if not supported_dimensions:
            raise TopologyError("mesh has no supported linear element types")
        source_dimension = max(supported_dimensions) if dimension is None else dimension
        if source_dimension not in supported_dimensions:
            raise TopologyError(f"mesh has no supported {source_dimension}-D element types")

        element_ids = _aligned_cell_data(mesh, "element_id")
        property_ids = _aligned_cell_data(mesh, "property_id")
        records: dict[tuple[int, ...], FaceRecord] = {}
        num_source_elements = 0

        for block_index, block in enumerate(mesh.cells):
            topology = get_element_topology(str(block.type))
            if topology is None or topology.dimension != source_dimension:
                continue

            connectivity = np.asarray(block.data)
            if connectivity.ndim != 2 or connectivity.shape[1] != topology.node_count:
                raise TopologyError(
                    f"cell block {block_index} ({block.type}) must have shape "
                    f"(n, {topology.node_count})"
                )
            if not np.issubdtype(connectivity.dtype, np.integer):
                raise TopologyError(f"cell block {block_index} connectivity must contain integers")
            if connectivity.size and (connectivity.min() < 0 or connectivity.max() >= len(points)):
                raise TopologyError(f"cell block {block_index} connectivity contains an invalid point index")

            block_element_ids = element_ids[block_index]
            block_property_ids = property_ids[block_index]
            num_source_elements += len(connectivity)
            for local_element_index, element_nodes in enumerate(connectivity):
                for local_face_index, local_nodes in enumerate(topology.boundary_entities):
                    oriented_nodes = tuple(int(element_nodes[index]) for index in local_nodes)
                    corner_key = tuple(sorted(oriented_nodes))
                    owner = FaceOwner(
                        cell_block_index=block_index,
                        local_element_index=local_element_index,
                        element_id=_metadata_value(block_element_ids, local_element_index, "element_id"),
                        property_id=_metadata_value(block_property_ids, local_element_index, "property_id"),
                        cell_type=str(block.type),
                        local_face_index=local_face_index,
                        oriented_nodes=oriented_nodes,
                    )
                    record = records.setdefault(
                        corner_key,
                        FaceRecord(corner_key=corner_key, full_node_key=corner_key),
                    )
                    record.owners.append(owner)

        for record in records.values():
            record.owners.sort(
                key=lambda owner: (
                    owner.cell_block_index,
                    owner.local_element_index,
                    owner.local_face_index,
                )
            )
        return cls(
            records,
            source_dimension=source_dimension,
            num_source_elements=num_source_elements,
        )


def _aligned_cell_data(mesh: Any, name: str) -> list[np.ndarray | None]:
    values = getattr(mesh, "cell_data", {}).get(name)
    if values is None:
        return [None] * len(mesh.cells)
    if len(values) != len(mesh.cells):
        raise TopologyError(
            f'cell_data["{name}"] has {len(values)} blocks; expected {len(mesh.cells)}'
        )

    aligned: list[np.ndarray] = []
    for block_index, (block, raw_values) in enumerate(zip(mesh.cells, values, strict=True)):
        array = np.asarray(raw_values)
        if array.ndim != 1 or len(array) != len(block.data):
            raise TopologyError(
                f'cell_data["{name}"][{block_index}] must have one value per cell'
            )
        if not np.issubdtype(array.dtype, np.integer):
            raise TopologyError(f'cell_data["{name}"][{block_index}] must contain integers')
        aligned.append(array)
    return aligned


def _metadata_value(values: np.ndarray | None, index: int, name: str) -> int | None:
    if values is None:
        return None
    value = values[index]
    if not isinstance(value, np.integer) and not isinstance(value, int):
        raise TopologyError(f'cell_data["{name}"] must contain integers')
    return int(value)
