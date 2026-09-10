from __future__ import annotations

import pytest

from ems_mesh_checker.topology.element_topology import get_element_topology


@pytest.mark.parametrize(
    ("cell_type", "dimension", "node_count", "num_boundary_entities"),
    [
        ("tetra", 3, 4, 4),
        ("hexahedron", 3, 8, 6),
        ("wedge", 3, 6, 5),
        ("pyramid", 3, 5, 5),
        ("triangle", 2, 3, 3),
        ("quad", 2, 4, 4),
    ],
)
def test_linear_element_topology_definitions(
    cell_type: str, dimension: int, node_count: int, num_boundary_entities: int
) -> None:
    topology = get_element_topology(cell_type)

    assert topology is not None
    assert topology.dimension == dimension
    assert topology.node_count == node_count
    assert len(topology.boundary_entities) == num_boundary_entities
    assert all(0 <= node < node_count for entity in topology.boundary_entities for node in entity)


def test_unsupported_element_topology_is_not_silently_mapped() -> None:
    assert get_element_topology("tetra10") is None
