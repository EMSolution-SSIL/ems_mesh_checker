from __future__ import annotations

import numpy as np
import pytest

import meshio

from ems_mesh_checker import BoundaryExtractor
from ems_mesh_checker.export import write_boundary_result_dxf, write_dxf


def _surface_mesh() -> meshio.Mesh:
    return meshio.Mesh(
        points=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        ),
        cells=[
            ("line", np.array([[0, 1]], dtype=int)),
            ("triangle", np.array([[0, 1, 2]], dtype=int)),
            ("quad", np.array([[0, 1, 3, 2]], dtype=int)),
        ],
        cell_data={
            "owner_property_id": [np.array([10]), np.array([1]), np.array([1])],
            "neighbor_property_id": [np.array([-1]), np.array([2]), np.array([2])],
        },
    )


def test_write_dxf_emits_lines_faces_and_property_layers(tmp_path) -> None:
    ezdxf = pytest.importorskip("ezdxf")
    output = write_dxf(_surface_mesh(), tmp_path / "surface.dxf")

    document = ezdxf.readfile(output)
    entities = list(document.modelspace())
    assert [entity.dxftype() for entity in entities] == ["LINE", "3DFACE", "3DFACE"]
    assert {entity.dxf.layer for entity in entities} == {"EXTERNAL_P10", "INTERFACE_P1_P2"}


def test_write_boundary_result_dxf_keeps_external_and_interface_layers(tmp_path) -> None:
    ezdxf = pytest.importorskip("ezdxf")
    mesh = meshio.Mesh(
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
        cell_data={"property_id": [np.array([10, 20], dtype=int)]},
    )
    result = BoundaryExtractor(mesh).extract()
    output = write_boundary_result_dxf(result, tmp_path / "boundaries.dxf")

    document = ezdxf.readfile(output)
    entities = list(document.modelspace())
    assert len(entities) == 7
    assert {entity.dxftype() for entity in entities} == {"3DFACE"}
    assert {entity.dxf.layer for entity in entities} == {"EXTERNAL", "INTERFACE_P10_P20"}
