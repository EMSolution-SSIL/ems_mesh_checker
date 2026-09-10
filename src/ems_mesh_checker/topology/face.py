"""Face ownership records used as the canonical topology representation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FaceOwner:
    """One volume or area element owning a boundary entity."""

    cell_block_index: int
    local_element_index: int
    element_id: int | None
    property_id: int | None
    cell_type: str
    local_face_index: int
    oriented_nodes: tuple[int, ...]


@dataclass
class FaceRecord:
    """All owners of a geometrically identical boundary entity."""

    corner_key: tuple[int, ...]
    full_node_key: tuple[int, ...]
    owners: list[FaceOwner] = field(default_factory=list)

    @property
    def owner_count(self) -> int:
        return len(self.owners)
