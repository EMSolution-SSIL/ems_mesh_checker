from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest

import cadquery as cq
import meshio

from ems_mesh_checker import (
    BoundaryExtractor,
    extract_planar_regions,
    extract_property_feature_edges,
    write_planar_regions_step,
    write_step_assembly,
)
from ems_mesh_checker.export import StepExportError


def _regions(offsets=((0.0, 0.0, 0.0),)):
    base = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
         [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float
    )
    points = np.vstack([base + offset for offset in offsets])
    cells = np.array([[i + 8 * j for i in range(8)] for j in range(len(offsets))])
    mesh = meshio.Mesh(points, [("hexahedron", cells)],
                       cell_data={"property_id": [np.ones(len(offsets), dtype=int)]})
    boundaries = BoundaryExtractor(mesh).extract()
    return extract_planar_regions(extract_property_feature_edges(boundaries, 1))


def test_box_step_roundtrip(tmp_path):
    output = tmp_path / "box.step"
    exported = write_planar_regions_step(_regions(), output, unit="MM")
    solids = cq.importers.importStep(str(output), unit="MM").solids().vals()
    assert exported.solid_count == 1
    assert len(solids) == 1
    assert solids[0].isValid()
    assert solids[0].Volume() == pytest.approx(1.0)
    assert len(solids[0].Faces()) == 6


def test_disconnected_property_step_roundtrip(tmp_path):
    output = tmp_path / "boxes.step"
    exported = write_planar_regions_step(_regions(((0, 0, 0), (3, 0, 0))), output, unit="MM")
    solids = cq.importers.importStep(str(output), unit="MM").solids().vals()
    assert exported.solid_count == len(solids) == 2
    assert sum(solid.Volume() for solid in solids) == pytest.approx(2.0)


def test_invalid_unit_does_not_write_step(tmp_path):
    output = tmp_path / "box.step"
    with pytest.raises(ValueError, match="unit"):
        write_planar_regions_step(_regions(), output, unit="BANANAS")
    assert not output.exists()


def test_cli_requires_explicit_step_unit(monkeypatch, capsys):
    from ems_mesh_checker.cli import main

    monkeypatch.setattr(sys, "argv", ["ems-mesh-checker", "unused.neu", "--step"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "--step requires --step-unit" in capsys.readouterr().err


def test_combined_step_retains_named_independent_solids(tmp_path):
    first = write_planar_regions_step(_regions(), tmp_path / "p1.step", unit="MM")
    second = write_planar_regions_step(
        _regions(((3, 0, 0),)), tmp_path / "p2.step", unit="MM"
    )
    second = replace(second, property_label="PROPERTY_P2")
    output = tmp_path / "all.step"
    assembled = write_step_assembly([first, second], output, unit="MM")
    solids = cq.importers.importStep(str(output), unit="MM").solids().vals()
    assert assembled.property_count == 2
    assert assembled.solid_count == len(solids) == 2
    assert sum(solid.Volume() for solid in solids) == pytest.approx(2.0)
    step_text = output.read_text(encoding="latin-1")
    assert "PROPERTY_P1_C1" in step_text
    assert "PROPERTY_P2_C1" in step_text


def test_combined_step_rejects_mixed_units(tmp_path):
    first = write_planar_regions_step(_regions(), tmp_path / "p1.step", unit="MM")
    with pytest.raises(StepExportError, match="units"):
        write_step_assembly([first, replace(first, unit="M")], tmp_path / "all.step", unit="MM")
    assert not (tmp_path / "all.step").exists()


def test_open_shell_is_rejected_without_step_file(tmp_path):
    complete = _regions()
    incomplete = replace(complete, regions=complete.regions[:-1])
    output = tmp_path / "open.step"
    with pytest.raises(StepExportError, match="open or invalid CAD shell"):
        write_planar_regions_step(incomplete, output, unit="MM")
    assert not output.exists()


def test_through_hole_is_retained_in_step(tmp_path):
    def node(i, j, k):
        return 16 * k + 4 * j + i

    points = np.array([(i, j, k) for k in range(2) for j in range(4)
                       for i in range(4)], dtype=float)
    cells = []
    for j in range(3):
        for i in range(3):
            if (i, j) == (1, 1):
                continue
            cells.append([node(i, j, 0), node(i + 1, j, 0),
                          node(i + 1, j + 1, 0), node(i, j + 1, 0),
                          node(i, j, 1), node(i + 1, j, 1),
                          node(i + 1, j + 1, 1), node(i, j + 1, 1)])
    mesh = meshio.Mesh(points, [("hexahedron", np.asarray(cells))],
                       cell_data={"property_id": [np.ones(len(cells), dtype=int)]})
    regions = extract_planar_regions(
        extract_property_feature_edges(BoundaryExtractor(mesh).extract(), 1)
    )
    assert any(region.hole_loops for region in regions.regions)
    output = tmp_path / "hole.step"
    write_planar_regions_step(regions, output, unit="MM")
    solid = cq.importers.importStep(str(output), unit="MM").solids().vals()[0]
    assert solid.Volume() == pytest.approx(8.0)


def test_ironcoil_preserves_analytical_cylinder(tmp_path):
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Cylinder
    from ems_file_format_converter import read_mesh

    sample = Path(__file__).resolve().parents[1] / "data/3D/IronCoil/post_geom.atl"
    if not sample.is_file():
        pytest.skip("public IronCoil sample is unavailable")
    boundaries = BoundaryExtractor(read_mesh(sample)).extract()
    regions = extract_planar_regions(extract_property_feature_edges(boundaries, 3))
    output = tmp_path / "coil.step"
    exported = write_planar_regions_step(regions, output, unit="MM")
    solid = cq.importers.importStep(str(output), unit="MM").solids().vals()[0]
    assert exported.solid_count == 1
    assert solid.isValid()
    assert any(BRepAdaptor_Surface(face.wrapped).GetType() == GeomAbs_Cylinder
               for face in solid.Faces())
