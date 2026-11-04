from __future__ import annotations

import os
from pathlib import Path
import shutil

import pytest

from ems_mesh_checker.morphing import (
    MorphingError, create_cylinder_manifest, create_l_prism_manifest,
    create_rectangle_manifest,
    read_edited_geo, write_editable_geo,
)

from test_geometry_follow import _grid
from test_parametric_morph import _annular_hexa, _l_grid


def _gmsh() -> Path:
    candidate = os.environ.get("GMSH_EXECUTABLE") or shutil.which("gmsh") or "C:/EMSolution/gmsh-4.13.1-Windows64/gmsh.exe"
    path = Path(candidate)
    if not path.is_file():
        pytest.skip("set GMSH_EXECUTABLE to test evaluated GEO correspondence")
    return path


def test_gmsh_evaluates_rectangle_variables_and_rejects_renumbering(tmp_path) -> None:
    gmsh = _gmsh()
    manifest = create_rectangle_manifest(_grid(), 2, deformable_properties=(1,))
    path = tmp_path / "editable.geo"
    write_editable_geo(manifest, path)
    source = path.read_text(encoding="utf-8")
    source = "right_x = 1.2;\n" + source.replace("Point(2) = {1,", "Point(2) = {right_x,").replace(
        "Point(3) = {1,", "Point(3) = {right_x,",
    )
    path.write_text(source, encoding="utf-8")
    case = read_edited_geo(manifest, path, gmsh_executable=gmsh)
    assert case["thickness"] == pytest.approx(2.2)
    assert case["translation"] == [0., 0.]
    path.write_text(source.replace("Line(4)", "Line(9)"), encoding="utf-8")
    with pytest.raises(MorphingError, match="renumbered|rejected"):
        read_edited_geo(manifest, path, gmsh_executable=gmsh)


def test_gmsh_evaluates_cylinder_arcs(tmp_path) -> None:
    gmsh = _gmsh()
    manifest = create_cylinder_manifest(
        _annular_hexa(), 2, deformable_properties=(1,),
        axis_point=(0., 0., 0.), axis_direction=(0., 0., 1.),
    )
    path = tmp_path / "cylinder.geo"
    write_editable_geo(manifest, path)
    original = read_edited_geo(manifest, path, gmsh_executable=gmsh)
    assert original["inner_radius"] == pytest.approx(1.)
    assert original["outer_radius"] == pytest.approx(2.)
    source = path.read_text(encoding="utf-8")
    for identifier, before, after in ((6, "2, 0", "2.2, 0"),
                                      (7, "0, 2", "0, 2.2"),
                                      (8, "-2, 0", "-2.2, 0"),
                                      (9, "0, -2", "0, -2.2")):
        source = source.replace(f"Point({identifier}) = {{{before},", f"Point({identifier}) = {{{after},")
    path.write_text(source, encoding="utf-8")
    edited = read_edited_geo(manifest, path, gmsh_executable=gmsh)
    assert edited["outer_radius"] == pytest.approx(2.2)


def test_gmsh_evaluates_l_notch_width(tmp_path) -> None:
    gmsh = _gmsh()
    manifest = create_l_prism_manifest(_l_grid(), 2, deformable_properties=(1,), notch=(1., 1.))
    path = tmp_path / "l.geo"
    write_editable_geo(manifest, path)
    source = path.read_text(encoding="utf-8")
    source = source.replace("Point(4) = {1,", "Point(4) = {1.1,").replace(
        "Point(5) = {1,", "Point(5) = {1.1,",
    )
    path.write_text(source, encoding="utf-8")
    case = read_edited_geo(manifest, path, gmsh_executable=gmsh)
    assert case["arm_width_x"] == pytest.approx(1.1)
