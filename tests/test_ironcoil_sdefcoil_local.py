"""Optional regression using the locally supplied SDEFCOIL mesh."""

from __future__ import annotations

from pathlib import Path

import pytest

from ems_mesh_checker.checks import check_mesh


SAMPLE = Path(__file__).resolve().parents[1] / "data/3D/IronCoil/post_geom_SDEFCOIL.neu"


def test_sdefcoil_inward_surface_normals() -> None:
    if not SAMPLE.is_file():
        pytest.skip("local SDEFCOIL sample is not distributed")
    converter = pytest.importorskip("ems_file_format_converter")

    mesh = converter.read_mesh(SAMPLE)
    surface_properties = {
        int(property_id)
        for index, block in enumerate(mesh.cells)
        if block.type in {"triangle", "quad"}
        for property_id in mesh.cell_data["property_id"][index]
    }
    assert surface_properties == {101, 102, 103, 104}
    outward_report = check_mesh(
        mesh,
        surface_owner_properties={102: 3, 103: 3, 104: 3},
    )
    report = check_mesh(
        mesh,
        surface_normal_direction="inward",
        surface_owner_properties={102: 3, 103: 3, 104: 3},
    )

    assert sum(item.code == "SURFACE_INWARD_NORMAL" for item in outward_report.findings) == 240
    assert report.checks["normal_orientation"].status == "PASS"
    assert report.checks["connectivity"].status == "PASS"
    assert not [item for item in report.findings if item.severity in {"ERROR", "WARNING"}]
    assert report.findings == []
