from __future__ import annotations

import numpy as np
import pytest

import meshio

from ems_mesh_checker import MeshContractError, inspect_mesh_contract


def _valid_mesh() -> meshio.Mesh:
    return meshio.Mesh(
        points=np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ),
        cells=[("tetra", np.array([[0, 1, 2, 3]], dtype=int))],
        point_data={"id": np.array([101, 102, 103, 104], dtype=int)},
        cell_data={
            "element_id": [np.array([201], dtype=int)],
            "property_id": [np.array([301], dtype=int)],
        },
    )


def test_inspect_mesh_contract_returns_deterministic_summary() -> None:
    summary = inspect_mesh_contract(_valid_mesh())

    assert summary.num_points == 4
    assert summary.num_cells == 1
    assert summary.cell_types == (("tetra", 1),)
    assert summary.has_point_ids
    assert summary.has_element_ids
    assert summary.has_property_ids


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda mesh: mesh.point_data.update({"id": np.array([1, 1, 2, 3])}), "unique"),
        (lambda mesh: mesh.cell_data.pop("element_id"), "element_id"),
        (lambda mesh: mesh.cell_data.update({"property_id": [np.array([], dtype=int)]}), "expected 1"),
        (
            lambda mesh: mesh.cells.__setitem__(
                0, meshio.CellBlock("tetra", np.array([[0, 1, 2, 4]]))
            ),
            "outside",
        ),
    ],
)
def test_inspect_mesh_contract_rejects_invalid_metadata_or_connectivity(mutate, message: str) -> None:
    mesh = _valid_mesh()
    mutate(mesh)

    with pytest.raises(MeshContractError, match=message):
        inspect_mesh_contract(mesh)
