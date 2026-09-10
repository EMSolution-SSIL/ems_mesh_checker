"""Real-model regression for the IronCoil ATLAS feature-edge example."""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess

import numpy as np
import pytest

from ems_mesh_checker import (
    BoundaryExtractor,
    FeatureEdgeConfig,
    PlanarRegionConfig,
    extract_planar_regions,
    extract_property_feature_edges,
    write_planar_regions_geo,
)


pytest.importorskip("pyvista")


def _ironcoil_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "3D" / "IronCoil" / "post_geom.atl"


def test_ironcoil_atlas_matches_property_block_feature_edge_workflow(tmp_path: Path) -> None:
    converter = pytest.importorskip("ems_file_format_converter")
    source = _ironcoil_path()
    if not source.is_file():
        pytest.skip(f"IronCoil ATLAS fixture is not available: {source}")

    mesh = converter.read_mesh(source)
    assert len(mesh.points) == 3312
    assert [(block.type, len(block.data)) for block in mesh.cells] == [
        ("hexahedron", 2715),
        ("wedge", 45),
    ]
    property_ids = sorted(set(np.concatenate(mesh.cell_data["property_id"]).tolist()))
    assert property_ids == [1, 3, 4]

    boundaries = BoundaryExtractor(mesh).extract()
    expected = {
        1: (80, 12, 6),
        3: (92, 12, 18),
        4: (259, 30, 30),
    }
    planar_results = []
    for property_id in property_ids:
        features = extract_property_feature_edges(
            boundaries,
            property_id,
            config=FeatureEdgeConfig(feature_angle_degrees=30.0),
        )
        regions = extract_planar_regions(
            features,
            config=PlanarRegionConfig(coplanar_angle_degrees=1.0),
        )
        expected_edges, expected_curves, expected_regions = expected[property_id]
        assert features.edge_count == expected_edges
        assert len(features.curves) == expected_curves
        assert len(regions.regions) == expected_regions
        assert not regions.unsupported_regions
        planar_results.append(regions)

    mesh_size = float(np.linalg.norm(np.ptp(mesh.points, axis=0))) / 100.0
    geo_path = write_planar_regions_geo(
        planar_results,
        tmp_path / "ironcoil_feature_edges.geo",
        mesh_size=mesh_size,
    )
    geo_text = geo_path.read_text(encoding="utf-8")
    # Coincident Property interfaces are emitted once and shared by both volumes.
    assert len(re.findall(r"^Plane Surface\(", geo_text, flags=re.MULTILINE)) == 21
    assert len(re.findall(r"^Surface\(", geo_text, flags=re.MULTILINE)) == 3
    assert geo_text.count("Physical Surface(") == 3
    assert geo_text.count("Circle(") == 6
    assert geo_text.count("Line(") == 47
    assert len(re.findall(r"^Surface Loop\(", geo_text, flags=re.MULTILINE)) == 3
    assert len(re.findall(r"^Volume\(", geo_text, flags=re.MULTILINE)) == 3
    assert geo_text.count("Physical Volume(") == 3
    assert "OPEN_SHELL" not in geo_text

    gmsh = os.environ.get("GMSH_EXECUTABLE")
    if gmsh:
        msh_path = tmp_path / "ironcoil_feature_edges.msh"
        completed = subprocess.run(
            [gmsh, str(geo_path), "-3", "-format", "msh4", "-o", str(msh_path), "-nopopup"],
            text=True,
            capture_output=True,
            check=False,
        )
        combined = f"{completed.stdout}\n{completed.stderr}"
        assert completed.returncode == 0, combined
        assert "Error" not in combined
        assert msh_path.is_file() and msh_path.stat().st_size > 0
