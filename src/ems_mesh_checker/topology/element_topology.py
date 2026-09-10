"""Local boundary-entity definitions for supported linear meshio cells."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ElementTopology:
    """Topology expressed using local node indices in meshio ordering."""

    cell_type: str
    dimension: int
    node_count: int
    boundary_entities: tuple[tuple[int, ...], ...]


# The ordering is retained on FaceOwner for later orientation normalization. Face
# identity itself is established from a sorted key and is independent of this order.
ELEMENT_TOPOLOGIES: dict[str, ElementTopology] = {
    "tetra": ElementTopology(
        cell_type="tetra",
        dimension=3,
        node_count=4,
        boundary_entities=((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)),
    ),
    "hexahedron": ElementTopology(
        cell_type="hexahedron",
        dimension=3,
        node_count=8,
        boundary_entities=(
            (0, 3, 2, 1),
            (4, 5, 6, 7),
            (0, 1, 5, 4),
            (1, 2, 6, 5),
            (2, 3, 7, 6),
            (3, 0, 4, 7),
        ),
    ),
    "wedge": ElementTopology(
        cell_type="wedge",
        dimension=3,
        node_count=6,
        boundary_entities=(
            (0, 2, 1),
            (3, 4, 5),
            (0, 1, 4, 3),
            (1, 2, 5, 4),
            (2, 0, 3, 5),
        ),
    ),
    "pyramid": ElementTopology(
        cell_type="pyramid",
        dimension=3,
        node_count=5,
        boundary_entities=((0, 3, 2, 1), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)),
    ),
    "triangle": ElementTopology(
        cell_type="triangle",
        dimension=2,
        node_count=3,
        boundary_entities=((0, 1), (1, 2), (2, 0)),
    ),
    "quad": ElementTopology(
        cell_type="quad",
        dimension=2,
        node_count=4,
        boundary_entities=((0, 1), (1, 2), (2, 3), (3, 0)),
    ),
}


def get_element_topology(cell_type: str) -> ElementTopology | None:
    """Return the supported linear topology for a meshio cell type, if any."""

    return ELEMENT_TOPOLOGIES.get(cell_type)
