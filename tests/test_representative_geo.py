from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import re
import subprocess

import numpy as np
import pytest

import meshio

from ems_mesh_checker import (
    BoundaryExtractor,
    extract_exterior_feature_edges,
    extract_property_group_feature_edges,
    extract_property_feature_edges,
    extract_planar_regions,
    write_planar_regions_dxf,
    write_planar_regions_geo,
)


pytest.importorskip("pyvista")


def _cube_mesh() -> meshio.Mesh:
    return meshio.Mesh(
        points=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ]
        ),
        cells=[("hexahedron", np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=int))],
        cell_data={"property_id": [np.array([1], dtype=int)]},
    )


def _two_disconnected_cubes_mesh() -> meshio.Mesh:
    first = _cube_mesh()
    points = np.vstack((first.points, first.points + np.array([2.0, 0.0, 0.0])))
    cells = np.vstack((first.cells[0].data, first.cells[0].data + len(first.points)))
    return meshio.Mesh(
        points=points,
        cells=[("hexahedron", cells)],
        cell_data={"property_id": [np.array([1, 1], dtype=int)]},
    )


def _two_adjacent_property_cubes_mesh() -> meshio.Mesh:
    return meshio.Mesh(
        points=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
                [2.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
                [2.0, 0.0, 1.0],
                [2.0, 1.0, 1.0],
            ]
        ),
        cells=[
            (
                "hexahedron",
                np.array(
                    [
                        [0, 1, 2, 3, 4, 5, 6, 7],
                        [1, 8, 9, 2, 5, 10, 11, 6],
                    ],
                    dtype=int,
                ),
            )
        ],
        cell_data={"property_id": [np.array([10, 20], dtype=int)]},
    )


def _cube_regions():
    boundaries = BoundaryExtractor(_cube_mesh()).extract()
    return extract_planar_regions(extract_exterior_feature_edges(boundaries))


def _property_cube_regions():
    boundaries = BoundaryExtractor(_cube_mesh()).extract()
    return extract_planar_regions(extract_property_feature_edges(boundaries, 1))


def _quarter_disk_regions():
    angles = np.radians(np.arange(0.0, 91.0, 18.0))
    arc = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(len(angles))))
    points = np.vstack(([0.0, 0.0, 0.0], arc))
    triangles = np.array(
        [[0, index, index + 1] for index in range(1, len(points) - 1)],
        dtype=int,
    )
    mesh = meshio.Mesh(
        points=points,
        cells=[("triangle", triangles)],
        cell_data={"property_id": [np.ones(len(triangles), dtype=int)]},
    )
    boundaries = BoundaryExtractor(mesh).extract()
    return extract_planar_regions(extract_exterior_feature_edges(boundaries))


def _quarter_disk_regions_in_yz_plane():
    angles = np.radians(np.arange(0.0, 91.0, 18.0))
    arc = np.column_stack(
        (
            np.full(len(angles), 2.0),
            np.cos(angles),
            np.sin(angles),
        )
    )
    points = np.vstack(([2.0, 0.0, 0.0], arc))
    triangles = np.array(
        [[0, index, index + 1] for index in range(1, len(points) - 1)],
        dtype=int,
    )
    mesh = meshio.Mesh(
        points=points,
        cells=[("triangle", triangles)],
        cell_data={"property_id": [np.ones(len(triangles), dtype=int)]},
    )
    boundaries = BoundaryExtractor(mesh).extract()
    return extract_planar_regions(extract_exterior_feature_edges(boundaries))


def test_cube_geo_contains_six_representative_surfaces_not_tessellated_faces(tmp_path: Path) -> None:
    output = write_planar_regions_geo(_cube_regions(), tmp_path / "cube.geo", mesh_size=0.25)
    text = output.read_text(encoding="utf-8")

    assert len(re.findall(r"^Plane Surface\(", text, flags=re.MULTILINE)) == 6
    assert len(re.findall(r"^Surface\(", text, flags=re.MULTILINE)) == 0
    assert len(re.findall(r"^Curve Loop\(", text, flags=re.MULTILINE)) == 6
    assert 'Physical Surface("EXTERIOR")' in text
    assert "representative_mesh_size = 0.25;" in text


def test_property_cube_reuses_edges_and_becomes_one_closed_volume(tmp_path: Path) -> None:
    output = write_planar_regions_geo(
        _property_cube_regions(),
        tmp_path / "property_cube.geo",
        mesh_size=0.25,
    )
    text = output.read_text(encoding="utf-8")

    assert len(re.findall(r"^Point\(", text, flags=re.MULTILINE)) == 8
    assert len(re.findall(r"^Line\(", text, flags=re.MULTILINE)) == 12
    assert len(re.findall(r"^Surface Loop\(", text, flags=re.MULTILINE)) == 1
    assert len(re.findall(r"^Volume\(", text, flags=re.MULTILINE)) == 1
    assert 'Physical Volume("PROPERTY_P1") = {1};' in text
    assert "closed_shell=0 surfaces=6 curves=12" in text


def test_open_property_shell_is_reported_and_not_exported_as_volume(tmp_path: Path) -> None:
    complete = _property_cube_regions()
    incomplete = replace(complete, regions=complete.regions[:-1])
    output = write_planar_regions_geo(incomplete, tmp_path / "open_cube.geo")
    text = output.read_text(encoding="utf-8")

    assert "OPEN_SHELL PROPERTY_P1" in text
    assert "boundary_curves=" in text
    assert len(re.findall(r"^Volume\(", text, flags=re.MULTILINE)) == 0
    assert "Physical Volume" not in text


def test_disconnected_regions_of_one_property_become_separate_volumes(tmp_path: Path) -> None:
    boundaries = BoundaryExtractor(_two_disconnected_cubes_mesh()).extract()
    regions = extract_planar_regions(extract_property_feature_edges(boundaries, 1))
    output = write_planar_regions_geo(regions, tmp_path / "two_cubes.geo")
    text = output.read_text(encoding="utf-8")

    assert len(re.findall(r"^Surface Loop\(", text, flags=re.MULTILINE)) == 2
    assert len(re.findall(r"^Volume\(", text, flags=re.MULTILINE)) == 2
    assert 'Physical Volume("PROPERTY_P1") = {1, 2};' in text
    assert "OPEN_SHELL" not in text


def test_merged_properties_omit_interface_and_export_one_labelled_volume(tmp_path: Path) -> None:
    boundaries = BoundaryExtractor(_two_adjacent_property_cubes_mesh()).extract()
    features = extract_property_group_feature_edges(boundaries, (10, 20))
    regions = extract_planar_regions(features)

    output = write_planar_regions_geo(regions, tmp_path / "merged_properties.geo")
    text = output.read_text(encoding="utf-8")

    assert len(re.findall(r"^Plane Surface\(", text, flags=re.MULTILINE)) == 6
    assert len(re.findall(r"^Volume\(", text, flags=re.MULTILINE)) == 1
    assert 'Physical Volume("PROPERTY_GROUP_P10_P20") = {1};' in text
    assert "OPEN_SHELL" not in text


def test_quarter_disk_geo_uses_one_circle_and_two_lines(tmp_path: Path) -> None:
    output = write_planar_regions_geo(_quarter_disk_regions(), tmp_path / "quarter_disk.geo")
    text = output.read_text(encoding="utf-8")

    assert len(re.findall(r"^Circle\(", text, flags=re.MULTILINE)) == 1
    assert len(re.findall(r"^Line\(", text, flags=re.MULTILINE)) == 2
    assert "original_edges=7 lines=2 circles=1" in text
    assert "Geometry.NumSubEdges = 100;" in text


def test_circle_discretization_is_scaled_by_arc_angle(tmp_path: Path) -> None:
    output = write_planar_regions_geo(
        _quarter_disk_regions(),
        tmp_path / "quarter_disk_transfinite.geo",
        circle_elements_per_turn=48,
    )
    text = output.read_text(encoding="utf-8")

    assert "Circular discretization target: 48 elements per turn" in text
    assert re.search(r"^Transfinite Curve \{\d+\} = 13;$", text, flags=re.MULTILINE)


def test_circle_discretization_rejects_fewer_than_three_elements(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least 3"):
        write_planar_regions_geo(
            _quarter_disk_regions(),
            tmp_path / "invalid.geo",
            circle_elements_per_turn=2,
        )


def test_representative_dxf_uses_arbitrary_plane_arc_and_lines(tmp_path: Path) -> None:
    ezdxf = pytest.importorskip("ezdxf")
    output = write_planar_regions_dxf(
        _quarter_disk_regions_in_yz_plane(),
        tmp_path / "quarter_disk_yz.dxf",
    )
    entities = list(ezdxf.readfile(output).modelspace())

    assert [entity.dxftype() for entity in entities].count("ARC") == 1
    assert [entity.dxftype() for entity in entities].count("LINE") == 2
    arc = next(entity for entity in entities if entity.dxftype() == "ARC")
    actual_endpoints = (np.asarray(arc.start_point), np.asarray(arc.end_point))
    expected_endpoints = (np.array([2.0, 1.0, 0.0]), np.array([2.0, 0.0, 1.0]))
    assert all(
        min(np.linalg.norm(actual - expected) for actual in actual_endpoints) < 1.0e-9
        for expected in expected_endpoints
    )


def test_representative_dxf_deduplicates_shared_property_curves(tmp_path: Path) -> None:
    ezdxf = pytest.importorskip("ezdxf")
    exterior = _quarter_disk_regions()
    property_one = replace(
        exterior,
        feature_edges=replace(
            exterior.feature_edges,
            source_kind="property",
            property_id=1,
        ),
    )
    property_two = replace(
        exterior,
        feature_edges=replace(
            exterior.feature_edges,
            source_kind="property",
            property_id=2,
        ),
    )
    output = write_planar_regions_dxf(
        (property_one, property_two),
        tmp_path / "shared.dxf",
    )
    entities = list(ezdxf.readfile(output).modelspace())

    assert len(entities) == 3
    assert {entity.dxf.layer for entity in entities} == {"SHARED_PROPERTIES"}


def test_gmsh_parses_representative_cube_geo_when_configured(tmp_path: Path) -> None:
    executable = os.environ.get("GMSH_EXECUTABLE")
    if not executable:
        pytest.skip("set GMSH_EXECUTABLE to enable the local Gmsh parse regression")
    output = write_planar_regions_geo(_cube_regions(), tmp_path / "cube.geo")

    completed = subprocess.run(
        [executable, str(output), "-parse_and_exit", "-nopopup"],
        text=True,
        capture_output=True,
        check=False,
    )

    combined = f"{completed.stdout}\n{completed.stderr}"
    assert completed.returncode == 0, combined
    assert "Error" not in combined


def test_gmsh_meshes_representative_cube_surfaces_when_configured(tmp_path: Path) -> None:
    executable = os.environ.get("GMSH_EXECUTABLE")
    if not executable:
        pytest.skip("set GMSH_EXECUTABLE to enable the local Gmsh mesh regression")
    geo_path = write_planar_regions_geo(_cube_regions(), tmp_path / "cube.geo", mesh_size=0.25)
    mesh_path = tmp_path / "cube.msh"

    completed = subprocess.run(
        [executable, str(geo_path), "-2", "-format", "msh2", "-o", str(mesh_path), "-nopopup"],
        text=True,
        capture_output=True,
        check=False,
    )

    combined = f"{completed.stdout}\n{completed.stderr}"
    assert completed.returncode == 0, combined
    assert "Error" not in combined
    assert mesh_path.is_file() and mesh_path.stat().st_size > 0


def test_gmsh_meshes_property_cube_volume_when_configured(tmp_path: Path) -> None:
    executable = os.environ.get("GMSH_EXECUTABLE")
    if not executable:
        pytest.skip("set GMSH_EXECUTABLE to enable the local Gmsh volume regression")
    geo_path = write_planar_regions_geo(
        _property_cube_regions(),
        tmp_path / "property_cube.geo",
        mesh_size=0.25,
    )
    mesh_path = tmp_path / "property_cube.msh"

    completed = subprocess.run(
        [executable, str(geo_path), "-3", "-format", "msh4", "-o", str(mesh_path), "-nopopup"],
        text=True,
        capture_output=True,
        check=False,
    )

    combined = f"{completed.stdout}\n{completed.stderr}"
    assert completed.returncode == 0, combined
    assert "Error" not in combined
    assert mesh_path.is_file() and mesh_path.stat().st_size > 0
