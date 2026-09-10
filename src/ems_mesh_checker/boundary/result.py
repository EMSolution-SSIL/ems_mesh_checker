"""Immutable result objects returned by property-boundary extraction."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from ems_mesh_checker.topology.face import FaceOwner, FaceRecord
from ems_mesh_checker.topology.face_table import FaceTable


PropertyPair = tuple[int, int]


@dataclass(frozen=True)
class BoundaryFace:
    """One selected boundary entity with canonical orientation and ownership."""

    record: FaceRecord
    owner: FaceOwner
    neighbor: FaceOwner | None
    oriented_nodes: tuple[int, ...]

    @property
    def property_pair(self) -> PropertyPair | None:
        if self.neighbor is None:
            return None
        assert self.owner.property_id is not None
        assert self.neighbor.property_id is not None
        return tuple(sorted((self.owner.property_id, self.neighbor.property_id)))


@dataclass(frozen=True)
class QuadTriInterface:
    """One suppressed same-Property hexahedron/tetrahedron coupling surface."""

    quad_face: FaceRecord
    triangle_faces: tuple[FaceRecord, FaceRecord]

    @property
    def face_count(self) -> int:
        return 3

    @property
    def property_ids(self) -> tuple[int, ...]:
        values = {
            owner.property_id
            for record in (self.quad_face, *self.triangle_faces)
            for owner in record.owners
            if owner.property_id is not None
        }
        return tuple(sorted(values))


@dataclass(frozen=True)
class BoundaryResult:
    """Classification of every supported boundary entity in a mesh."""

    mesh: Any
    face_table: FaceTable
    external_faces: tuple[BoundaryFace, ...]
    internal_faces: tuple[FaceRecord, ...]
    property_interfaces: Mapping[PropertyPair, tuple[BoundaryFace, ...]]
    interface_patches: Mapping[PropertyPair, tuple[tuple[BoundaryFace, ...], ...]]
    nonmanifold_faces: tuple[FaceRecord, ...]
    unclassified_faces: tuple[FaceRecord, ...]
    suppressed_quad_tri_interfaces: tuple[QuadTriInterface, ...]
    warnings: tuple[str, ...]

    @classmethod
    def create(
        cls,
        *,
        mesh: Any,
        face_table: FaceTable,
        external_faces: list[BoundaryFace],
        internal_faces: list[FaceRecord],
        property_interfaces: dict[PropertyPair, list[BoundaryFace]],
        interface_patches: dict[PropertyPair, tuple[tuple[BoundaryFace, ...], ...]],
        nonmanifold_faces: list[FaceRecord],
        unclassified_faces: list[FaceRecord],
        warnings: list[str],
        suppressed_quad_tri_interfaces: list[QuadTriInterface] | None = None,
    ) -> "BoundaryResult":
        ordered_interfaces = {
            pair: tuple(sorted(faces, key=_face_key))
            for pair, faces in sorted(property_interfaces.items())
        }
        ordered_patches = {
            pair: tuple(tuple(patch) for patch in patches)
            for pair, patches in sorted(interface_patches.items())
        }
        return cls(
            mesh=mesh,
            face_table=face_table,
            external_faces=tuple(sorted(external_faces, key=_face_key)),
            internal_faces=tuple(sorted(internal_faces, key=lambda record: record.corner_key)),
            property_interfaces=MappingProxyType(ordered_interfaces),
            interface_patches=MappingProxyType(ordered_patches),
            nonmanifold_faces=tuple(sorted(nonmanifold_faces, key=lambda record: record.corner_key)),
            unclassified_faces=tuple(sorted(unclassified_faces, key=lambda record: record.corner_key)),
            suppressed_quad_tri_interfaces=tuple(
                sorted(
                    suppressed_quad_tri_interfaces or (),
                    key=lambda interface: interface.quad_face.corner_key,
                )
            ),
            warnings=tuple(warnings),
        )

    @property
    def external_surface(self) -> Any:
        """Return the complete external surface as a new ``meshio.Mesh``."""

        from .surface_mesh import build_surface_mesh

        return build_surface_mesh(self.mesh, self.external_faces)

    def get_interface(self, property_a: int, property_b: int) -> Any:
        """Return all faces of a property pair as a new ``meshio.Mesh``."""

        from .surface_mesh import build_surface_mesh

        pair = tuple(sorted((int(property_a), int(property_b))))
        return build_surface_mesh(self.mesh, self.property_interfaces.get(pair, ()))

    def get_interface_patches(self, property_a: int, property_b: int) -> tuple[Any, ...]:
        """Return one surface mesh per connected patch of a property pair."""

        from .surface_mesh import build_surface_mesh

        pair = tuple(sorted((int(property_a), int(property_b))))
        return tuple(
            build_surface_mesh(self.mesh, patch)
            for patch in self.interface_patches.get(pair, ())
        )


def _face_key(face: BoundaryFace) -> tuple[int, ...]:
    return face.record.corner_key
