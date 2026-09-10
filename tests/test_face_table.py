from __future__ import annotations

import numpy as np

import meshio

from ems_mesh_checker.topology.face_table import FaceTable


def _two_tetra_mesh() -> meshio.Mesh:
    return meshio.Mesh(
        points=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ]
        ),
        cells=[("tetra", np.array([[0, 1, 2, 3], [0, 2, 1, 4]], dtype=int))],
        cell_data={
            "element_id": [np.array([1001, 1002], dtype=int)],
            "property_id": [np.array([10, 20], dtype=int)],
        },
    )


def test_face_table_groups_shared_tetra_face_and_preserves_owner_metadata() -> None:
    table = FaceTable.from_mesh(_two_tetra_mesh())

    assert table.source_dimension == 3
    assert table.num_source_elements == 2
    assert len(table) == 7
    shared = table.get((0, 1, 2))
    assert shared is not None
    assert shared.owner_count == 2
    assert [owner.element_id for owner in shared.owners] == [1001, 1002]
    assert [owner.property_id for owner in shared.owners] == [10, 20]


def test_face_table_uses_highest_dimension_and_ignores_line_cells() -> None:
    mesh = meshio.Mesh(
        points=np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
        ),
        cells=[
            ("line", np.array([[0, 1]], dtype=int)),
            ("quad", np.array([[0, 1, 2, 3]], dtype=int)),
        ],
    )

    table = FaceTable.from_mesh(mesh)

    assert table.source_dimension == 2
    assert table.num_source_elements == 1
    assert len(table) == 4
    assert all(owner.cell_type == "quad" for record in table.iter_records() for owner in record.owners)
