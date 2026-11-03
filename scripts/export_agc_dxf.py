"""Export the AGC fixtures to DXF for visual boundary-extraction checks.

Run from the ``ems_mesh_checker`` directory after installing the project together
with its converter and DXF optional dependencies::

    python scripts/export_agc_dxf.py

The default exports the 2-D AGC input as LINE entities and its extruded 3-D mesh as
3DFACE entities. Use ``--include-large-3d`` only when a large DXF is intended.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import subprocess

from ems_file_format_converter import read_mesh
from ems_mesh_checker import BoundaryExtractor
from ems_mesh_checker.export import dxf_to_geo, write_boundary_result_dxf


@dataclass(frozen=True)
class ExportCase:
    name: str
    input_path: Path
    output_path: Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-large-3d",
        action="store_true",
        help="also export data/3D/AGC/post_geom.neu (the resulting DXF can be large)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_project_root() / "generated" / "AGC",
        help="directory where DXF files are written",
    )
    parser.add_argument(
        "--gmsh-executable",
        type=Path,
        help="optional gmsh.exe path; convert each DXF to GEO and parse the GEO",
    )
    args = parser.parse_args()

    for case in _cases(args.output_dir, include_large_3d=args.include_large_3d):
        mesh = read_mesh(case.input_path)
        result = BoundaryExtractor(mesh).extract()
        write_boundary_result_dxf(result, case.output_path)
        if args.gmsh_executable:
            geo_path = dxf_to_geo(case.output_path, case.output_path.with_suffix(".geo"))
            _verify_gmsh_import(args.gmsh_executable, geo_path)
        interface_faces = sum(len(faces) for faces in result.property_interfaces.values())
        patch_count = sum(len(patches) for patches in result.interface_patches.values())
        print(
            f"{case.name}: {case.output_path}\n"
            f"  external={len(result.external_faces)}, interface={interface_faces}, "
            f"pairs={len(result.property_interfaces)}, patches={patch_count}, "
            f"warnings={len(result.warnings)}"
        )


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _cases(output_dir: Path, *, include_large_3d: bool) -> tuple[ExportCase, ...]:
    root = _project_root()
    cases = [
        ExportCase(
            "AGC 2D pre-geometry boundary lines",
            root / "data" / "2D" / "AGC" / "pre_geom2D.neu",
            output_dir / "2D" / "pre_geom2D_boundaries.dxf",
        ),
        ExportCase(
            "AGC 2D extruded boundary faces",
            root / "data" / "2D" / "AGC" / "post_geom.neu",
            output_dir / "2D" / "post_geom_boundaries.dxf",
        ),
    ]
    if include_large_3d:
        cases.append(
            ExportCase(
                "AGC 3D boundary faces",
                root / "data" / "3D" / "AGC" / "post_geom.neu",
                output_dir / "3D" / "post_geom_boundaries.dxf",
            )
        )
    return tuple(cases)


def _verify_gmsh_import(gmsh_executable: Path, geo_path: Path) -> None:
    if not gmsh_executable.is_file():
        raise FileNotFoundError(f"Gmsh executable does not exist: {gmsh_executable}")
    completed = subprocess.run(
        [str(gmsh_executable), str(geo_path), "-parse_and_exit", "-nopopup"],
        text=True,
        capture_output=True,
        check=False,
    )
    output = f"{completed.stdout}\n{completed.stderr}"
    if completed.returncode != 0 or "Error" in output:
        raise RuntimeError(f"Gmsh could not parse {geo_path}:\n{output}")


if __name__ == "__main__":
    main()
