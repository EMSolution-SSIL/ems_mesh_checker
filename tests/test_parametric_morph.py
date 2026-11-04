from __future__ import annotations

from types import SimpleNamespace

import meshio
import numpy as np
import pytest

from ems_mesh_checker.morphing import (
    MorphingError, apply_cylinder_case, apply_l_prism_case,
    create_cylinder_manifest, create_l_prism_manifest,
)

from test_geometry_follow import _triangle_backend


def _l_grid() -> meshio.Mesh:
    values = np.arange(-1.0, 5.0)
    points = np.asarray([(x, y) for y in values for x in values], dtype=float)
    cells, properties = [], []
    for j in range(5):
        for i in range(5):
            a = j * 6 + i
            selected = (i == 1 and 1 <= j <= 3) or (j == 1 and 1 <= i <= 3)
            cells.extend(([a, a + 1, a + 7], [a, a + 7, a + 6]))
            properties.extend((2 if selected else 1,) * 2)
    return meshio.Mesh(
        points, [("triangle", np.asarray(cells))],
        point_data={"id": np.arange(101, 137)},
        cell_data={"element_id": [np.arange(201, 251)],
                   "property_id": [np.asarray(properties)]},
    )


def _hexa_backend():
    def analyze(_kind, coords):
        edge_x = coords[:, 1] - coords[:, 0]
        edge_y = coords[:, 3] - coords[:, 0]
        edge_z = coords[:, 4] - coords[:, 0]
        determinant = np.einsum("ij,ij->i", np.cross(edge_x, edge_y), edge_z)
        scale = np.maximum.reduce([
            np.linalg.norm(edge_x, axis=1), np.linalg.norm(edge_y, axis=1),
            np.linalg.norm(edge_z, axis=1),
        ])
        return {"measure": np.abs(determinant), "min_jacobian": determinant,
                "max_jacobian": determinant, "scale": scale}
    return SimpleNamespace(
        __name__="analytical_hexa_test", __version__="1",
        SUPPORTED_CELL_TYPES=("hexahedron",), analyze_block=analyze,
    )


def _extrude(mesh: meshio.Mesh) -> meshio.Mesh:
    planar = np.asarray(mesh.points)
    n = len(planar)
    points = np.vstack([np.column_stack((planar, np.full(n, z))) for z in (-1., 0., 1., 2.)])
    # Build a regular hexahedral grid, not from triangulated 2-D cells.
    cells, properties = [], []
    for k in range(3):
        for j in range(5):
            for i in range(5):
                a = k * n + j * 6 + i
                b = a + n
                cells.append([a, a + 1, a + 7, a + 6,
                              b, b + 1, b + 7, b + 6])
                selected = k == 1 and ((i == 1 and 1 <= j <= 3) or (j == 1 and 1 <= i <= 3))
                properties.append(2 if selected else 1)
    return meshio.Mesh(
        points, [("hexahedron", np.asarray(cells))],
        point_data={"id": np.arange(101, 101 + len(points))},
        cell_data={"element_id": [np.arange(201, 201 + len(cells))],
                   "property_id": [np.asarray(properties)]},
    )


def _annular_hexa() -> meshio.Mesh:
    radii = (0.5, 1.0, 2.0, 3.0)
    zs = (-1.0, 0.0, 1.0, 2.0)
    sectors = 12
    points = []
    for z in zs:
        for radius in radii:
            for theta_index in range(sectors):
                theta = 2 * np.pi * theta_index / sectors
                points.append((radius * np.cos(theta), radius * np.sin(theta), z))
    def node(k, r, t):
        return (k * len(radii) + r) * sectors + t % sectors
    cells, properties = [], []
    for k in range(3):
        for r in range(3):
            for t in range(sectors):
                cells.append([node(k, r, t), node(k, r + 1, t),
                              node(k, r + 1, t + 1), node(k, r, t + 1),
                              node(k + 1, r, t), node(k + 1, r + 1, t),
                              node(k + 1, r + 1, t + 1), node(k + 1, r, t + 1)])
                properties.append(2 if k == 1 and r == 1 else 1)
    return meshio.Mesh(
        np.asarray(points), [("hexahedron", np.asarray(cells))],
        point_data={"id": np.arange(1001, 1001 + len(points))},
        cell_data={"element_id": [np.arange(2001, 2001 + len(cells))],
                   "property_id": [np.asarray(properties)]},
    )


def test_l_shape_parameter_edit_and_invalid_notch() -> None:
    pytest.importorskip("ems_mesh_morpher")
    mesh = _l_grid()
    manifest = create_l_prism_manifest(mesh, 2, deformable_properties=(1,), notch=(1., 1.))
    assert manifest["shape"] == "l_shape_2d"
    original = mesh.points.copy()
    result = apply_l_prism_case(
        mesh, manifest, length_x=3.2, length_y=2.8,
        arm_width_x=1.1, arm_width_y=0.9,
        geometry_backend=_triangle_backend(),
    )
    assert result.report["status"] == "PASS"
    assert result.report["max_target_error"] < 1e-12
    assert np.array_equal(mesh.points, original)
    with pytest.raises(MorphingError, match="positive widths"):
        apply_l_prism_case(mesh, manifest, arm_width_x=4., geometry_backend=_triangle_backend())


def test_extruded_l_shape_keeps_axial_layers() -> None:
    pytest.importorskip("ems_mesh_morpher")
    mesh = _extrude(_l_grid())
    manifest = create_l_prism_manifest(mesh, 2, deformable_properties=(1,), notch=(1., 1.))
    assert manifest["shape"] == "l_prism_3d"
    result = apply_l_prism_case(mesh, manifest, arm_width_x=1.1, geometry_backend=_hexa_backend())
    assert result.report["status"] == "PASS"
    assert np.array_equal(result.mesh.points[:, 2], mesh.points[:, 2])


def test_cylinder_radii_and_axial_ends() -> None:
    pytest.importorskip("ems_mesh_morpher")
    mesh = _annular_hexa()
    manifest = create_cylinder_manifest(
        mesh, 2, deformable_properties=(1,),
        axis_point=(0., 0., 0.), axis_direction=(0., 0., 1.),
    )
    assert manifest["inner_radius"] == pytest.approx(1.)
    assert manifest["outer_radius"] == pytest.approx(2.)
    result = apply_cylinder_case(
        mesh, manifest, inner_radius=1.1, outer_radius=2.,
        geometry_backend=_hexa_backend(),
    )
    assert result.report["status"] == "PASS"
    assert np.array_equal(result.mesh.points[:, 2], mesh.points[:, 2])
    with pytest.raises(MorphingError, match="inner < outer"):
        apply_cylinder_case(mesh, manifest, inner_radius=2.1, outer_radius=2., geometry_backend=_hexa_backend())


def test_partial_annulus_is_not_misidentified_as_complete_cylinder() -> None:
    mesh = _annular_hexa()
    properties = mesh.cell_data["property_id"][0]
    # Remove one angular sector from the target Property without altering the mesh.
    properties[48] = 1
    with pytest.raises(MorphingError, match="ring is open"):
        create_cylinder_manifest(
            mesh, 2, deformable_properties=(1,),
            axis_point=(0., 0., 0.), axis_direction=(0., 0., 1.),
        )
