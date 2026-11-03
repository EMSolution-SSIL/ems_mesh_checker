from __future__ import annotations

import ezdxf
import numpy as np
import pytest

from ems_mesh_checker.morphing import (
    MorphingError, create_cylinder_manifest, create_l_prism_manifest,
    create_rectangle_manifest, read_edited_dxf, write_editable_dxf,
)

from test_geometry_follow import _grid
from test_parametric_morph import _annular_hexa, _l_grid


def test_rectangle_edit_round_trip_and_extra_entity_rejection(tmp_path) -> None:
    manifest = create_rectangle_manifest(_grid(), 2, deformable_properties=(1,))
    path = tmp_path / "rectangle.dxf"
    write_editable_dxf(manifest, path)
    document = ezdxf.readfile(path)
    right = next(item for item in document.modelspace() if item.dxf.layer == "EMS_MORPH_RIGHT")
    top = next(item for item in document.modelspace() if item.dxf.layer == "EMS_MORPH_TOP")
    bottom = next(item for item in document.modelspace() if item.dxf.layer == "EMS_MORPH_BOTTOM")
    right.dxf.start = (1.2, -1., 0.)
    right.dxf.end = (1.2, 1., 0.)
    top.dxf.start = (1.2, 1., 0.)
    bottom.dxf.end = (1.2, -1., 0.)
    document.save()
    assert read_edited_dxf(manifest, path) == {
        "translation": [0., 0.], "thickness": 2.2, "anchor": "min",
    }
    document.modelspace().add_line((0, 0), (1, 1))
    document.save()
    with pytest.raises(MorphingError, match="exactly"):
        read_edited_dxf(manifest, path)


def test_l_edit_round_trip_and_broken_corner(tmp_path) -> None:
    manifest = create_l_prism_manifest(_l_grid(), 2, deformable_properties=(1,), notch=(1., 1.))
    path = tmp_path / "l.dxf"
    write_editable_dxf(manifest, path)
    result = read_edited_dxf(manifest, path)
    assert result["arm_width_x"] == 1.
    document = ezdxf.readfile(path)
    horizontal = next(item for item in document.modelspace() if item.dxf.layer == "EMS_MORPH_INNER_H")
    horizontal.dxf.end = (1.2, 1., 0.)
    document.save()
    with pytest.raises(MorphingError, match="concave corner"):
        read_edited_dxf(manifest, path)


def test_cylinder_edit_round_trip(tmp_path) -> None:
    manifest = create_cylinder_manifest(
        _annular_hexa(), 2, deformable_properties=(1,),
        axis_point=(0., 0., 0.), axis_direction=(0., 0., 1.),
    )
    path = tmp_path / "cylinder.dxf"
    write_editable_dxf(manifest, path)
    document = ezdxf.readfile(path)
    inner = next(item for item in document.modelspace() if item.dxf.layer == "EMS_MORPH_INNER")
    inner.dxf.radius = 1.1
    document.save()
    result = read_edited_dxf(manifest, path)
    assert result["inner_radius"] == pytest.approx(1.1)
    assert result["outer_radius"] == pytest.approx(2.)
