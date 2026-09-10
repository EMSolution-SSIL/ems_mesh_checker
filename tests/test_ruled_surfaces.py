from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import re
import subprocess

import meshio
import numpy as np
import pytest

from ems_mesh_checker import (
    BoundaryExtractor,
    PlanarLoop,
    extract_exterior_feature_edges,
    extract_planar_regions,
    reconstruct_ruled_surfaces,
    RuledSurfaceConfig,
    write_planar_regions_geo,
)


def _quarter_cylinder_regions():
    angles = np.radians(np.arange(0.0, 91.0, 18.0))
    arc = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(len(angles))))
    bottom = np.vstack(([0.0, 0.0, 0.0], arc))
    top = bottom + np.array([0.0, 0.0, 2.0])
    points = np.vstack((bottom, top))
    offset = len(bottom)
    wedges = np.array(
        [
            [0, index, index + 1, offset, offset + index, offset + index + 1]
            for index in range(1, len(bottom) - 1)
        ],
        dtype=int,
    )
    mesh = meshio.Mesh(
        points=points,
        cells=[("wedge", wedges)],
        cell_data={"property_id": [np.ones(len(wedges), dtype=int)]},
    )
    boundaries = BoundaryExtractor(mesh).extract()
    return extract_planar_regions(extract_exterior_feature_edges(boundaries))


def _split_curved_strips_at_midplane():
    regions = _quarter_cylinder_regions()
    curved = set(reconstruct_ruled_surfaces(regions).replaced_region_indices)
    split_regions = []
    for region in regions.regions:
        if region.region_index not in curved:
            split_regions.append(region)
            continue
        points = region.outer_loop.points
        z_low = float(np.min(points[:, 2]))
        z_high = float(np.max(points[:, 2]))
        z_mid = 0.5 * (z_low + z_high)
        lower = points.copy()
        lower[np.isclose(lower[:, 2], z_high), 2] = z_mid
        upper = points.copy()
        upper[np.isclose(upper[:, 2], z_low), 2] = z_mid
        for patch_points in (lower, upper):
            split_regions.append(
                replace(
                    region,
                    outer_loop=PlanarLoop(
                        points=patch_points,
                        signed_area=0.5 * region.outer_loop.signed_area,
                        is_hole=False,
                    ),
                )
            )
    return replace(
        regions,
        regions=tuple(
            replace(region, region_index=index)
            for index, region in enumerate(split_regions)
        ),
    )


def test_quarter_cylinder_strips_become_one_four_curve_surface() -> None:
    regions = _quarter_cylinder_regions()

    result = reconstruct_ruled_surfaces(regions)

    assert len(result.patches) == 1
    patch = result.patches[0]
    assert len(patch.source_region_indices) == 5
    assert patch.radius == pytest.approx(1.0)
    np.testing.assert_allclose(abs(patch.extrusion_vector[2]), 2.0)
    assert len(result.replaced_region_indices) == 5


def test_ruled_surface_reconstruction_can_be_disabled() -> None:
    result = reconstruct_ruled_surfaces(
        _quarter_cylinder_regions(),
        config=RuledSurfaceConfig(enabled=False),
    )

    assert not result.patches
    assert not result.replaced_region_indices
    assert result.diagnostics == ("ruled_surface_reconstruction_disabled",)


def test_axially_segmented_cylinder_becomes_stacked_ruled_surfaces() -> None:
    result = reconstruct_ruled_surfaces(_split_curved_strips_at_midplane())

    assert len(result.patches) == 2
    assert sorted(len(patch.source_region_indices) for patch in result.patches) == [5, 5]
    assert len(result.replaced_region_indices) == 10
    assert sorted(abs(float(patch.extrusion_vector[2])) for patch in result.patches) == [1.0, 1.0]


def test_isolated_cylindrical_strip_uses_known_cap_circle_family() -> None:
    regions = _quarter_cylinder_regions()
    full = reconstruct_ruled_surfaces(regions)
    curved = set(full.replaced_region_indices)
    retained_curved = min(curved)
    reduced = replace(
        regions,
        regions=tuple(
            replace(region, region_index=index)
            for index, region in enumerate(
                region
                for region in regions.regions
                if region.region_index not in curved
                or region.region_index == retained_curved
            )
        ),
    )

    result = reconstruct_ruled_surfaces(reduced)

    assert len(result.patches) == 1
    assert len(result.patches[0].source_region_indices) == 1
    assert result.patches[0].primitives[0].source_edge_count == 1


def test_quarter_cylinder_geo_uses_one_ruled_surface(tmp_path: Path) -> None:
    regions = _quarter_cylinder_regions()
    geo_path = write_planar_regions_geo(regions, tmp_path / "quarter_cylinder.geo", mesh_size=0.2)
    text = geo_path.read_text(encoding="utf-8")

    assert len(re.findall(r"^Plane Surface\(", text, flags=re.MULTILINE)) == 4
    assert len(re.findall(r"^Surface\(", text, flags=re.MULTILINE)) == 1
    assert "ruled_patch=0" in text

    executable = os.environ.get("GMSH_EXECUTABLE")
    if executable:
        mesh_path = tmp_path / "quarter_cylinder.msh"
        completed = subprocess.run(
            [executable, str(geo_path), "-2", "-format", "msh4", "-o", str(mesh_path), "-nopopup"],
            text=True,
            capture_output=True,
            check=False,
        )
        combined = f"{completed.stdout}\n{completed.stderr}"
        assert completed.returncode == 0, combined
        assert "Error" not in combined
        assert mesh_path.is_file() and mesh_path.stat().st_size > 0
