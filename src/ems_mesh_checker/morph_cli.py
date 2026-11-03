"""CLI for manifest-backed, parameter-driven mesh edits.

Only strictly recognized edits to an exported DXF sketch can be re-imported.
Arbitrary GEO/DXF cannot be mapped unambiguously to original node IDs.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from ems_mesh_checker.morphing import (
    MorphingError, apply_cylinder_case, apply_l_prism_case, apply_rectangle_case,
    apply_octant_scale_case, apply_octant_shell_case,
    create_cylinder_manifest, create_l_prism_manifest, create_rectangle_manifest,
    create_octant_scale_manifest, create_octant_shell_manifest,
    read_edited_dxf, read_edited_geo, write_editable_dxf, write_editable_geo,
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    if path.exists():
        raise MorphingError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _read_mesh(path: Path):
    try:
        from ems_file_format_converter.io import read_mesh
    except ImportError as error:
        raise MorphingError("ems_file_format_converter is required for mesh file input") from error
    return read_mesh(path)


def _element_signature(mesh: Any) -> dict[int, tuple[str, int, tuple[int, ...]]]:
    node_ids = np.asarray(mesh.point_data["id"], dtype=int)
    signatures: dict[int, tuple[str, int, tuple[int, ...]]] = {}
    for index, block in enumerate(mesh.cells):
        for row, eid, pid in zip(
            block.data, mesh.cell_data["element_id"][index],
            mesh.cell_data["property_id"][index], strict=True,
        ):
            identifier = int(eid)
            if identifier in signatures:
                raise MorphingError(f"duplicate output element ID {identifier}")
            signatures[identifier] = (
                str(block.type), int(pid), tuple(int(node_ids[node]) for node in row),
            )
    return signatures


def _write_verified_mesh(path: Path, mesh: Any) -> None:
    """Publish only after a converter round-trip preserves the mesh contract."""

    from ems_file_format_converter.io import write_mesh

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".morph-", suffix=path.suffix, dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        write_mesh(temporary, mesh)
        reread = _read_mesh(temporary)
        if set(map(int, mesh.point_data["id"])) != set(map(int, reread.point_data["id"])):
            raise MorphingError("mesh writer changed original node IDs")
        if _element_signature(mesh) != _element_signature(reread):
            raise MorphingError("mesh writer changed element IDs, Property IDs, or connectivity")
        original = {int(identifier): np.asarray(point, dtype=float) for identifier, point in zip(mesh.point_data["id"], mesh.points, strict=True)}
        for identifier, point in zip(reread.point_data["id"], reread.points, strict=True):
            source = original[int(identifier)]
            destination = np.asarray(point, dtype=float)
            if len(source) == 2 and len(destination) == 3:
                source = np.r_[source, 0.0]
            if not np.allclose(source, destination, rtol=0, atol=1e-10 * max(1., float(np.max(np.abs(source))))):
                raise MorphingError(f"mesh writer changed node {identifier} coordinates beyond tolerance")
        if path.exists():
            raise MorphingError(f"refusing to overwrite {path}")
        temporary.rename(path)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manifest-backed geometry-following mesh morphing")
    commands = parser.add_subparsers(dest="command", required=True)
    manifest = commands.add_parser("manifest", help="fit a supported Property shape")
    manifest.add_argument("mesh", type=Path)
    manifest.add_argument("output", type=Path)
    manifest.add_argument("--property", type=int, required=True)
    manifest.add_argument("--deformable", type=int, action="append", required=True, metavar="PROPERTY_ID")
    manifest.add_argument("--shape", choices=("rectangle", "l_prism", "cylinder", "octant_scale", "octant_shell"), default="rectangle")
    manifest.add_argument("--axis", choices=("x", "y"), default="x")
    manifest.add_argument("--notch", type=float, nargs=2, metavar=("X", "Y"))
    manifest.add_argument("--axis-point", type=float, nargs=3, metavar=("X", "Y", "Z"))
    manifest.add_argument("--axis-direction", type=float, nargs=3, metavar=("X", "Y", "Z"))
    manifest.add_argument("--inner-property", type=int)
    manifest.add_argument("--outer-property", type=int)
    manifest.add_argument("--unit", default="model coordinate unit")
    manifest.add_argument("--tolerance", type=float, default=1e-8)
    apply = commands.add_parser("apply", help="generate independent cases from the base mesh")
    apply.add_argument("mesh", type=Path)
    apply.add_argument("manifest", type=Path)
    apply.add_argument("cases", type=Path, help="JSON array of named shape parameters or displacements")
    apply.add_argument("output_dir", type=Path)
    apply.add_argument("--interpolation", choices=("weighted_laplace", "rbf", "analytic_radial"))
    apply.add_argument("--format", choices=("neu", "msh", "unv"), default="neu")
    export_dxf = commands.add_parser("export-dxf", help="write an editable, named DXF sketch")
    export_dxf.add_argument("manifest", type=Path)
    export_dxf.add_argument("output", type=Path)
    export_geo = commands.add_parser("export-geo", help="write an editable, numbered Gmsh GEO sketch")
    export_geo.add_argument("manifest", type=Path)
    export_geo.add_argument("output", type=Path)
    apply_dxf = commands.add_parser("apply-dxf", help="apply a strictly recognized DXF edit")
    apply_dxf.add_argument("mesh", type=Path)
    apply_dxf.add_argument("manifest", type=Path)
    apply_dxf.add_argument("edited_dxf", type=Path)
    apply_dxf.add_argument("output_dir", type=Path)
    apply_dxf.add_argument("--name", default="edited")
    apply_dxf.add_argument("--interpolation", choices=("weighted_laplace", "rbf", "analytic_radial"))
    apply_dxf.add_argument("--format", choices=("neu", "msh", "unv"), default="neu")
    apply_geo = commands.add_parser("apply-geo", help="evaluate tagged GEO entities using Gmsh")
    apply_geo.add_argument("mesh", type=Path)
    apply_geo.add_argument("manifest", type=Path)
    apply_geo.add_argument("edited_geo", type=Path)
    apply_geo.add_argument("output_dir", type=Path)
    apply_geo.add_argument("--name", default="edited")
    apply_geo.add_argument("--gmsh-executable", type=Path)
    apply_geo.add_argument("--interpolation", choices=("weighted_laplace", "rbf", "analytic_radial"))
    apply_geo.add_argument("--format", choices=("neu", "msh", "unv"), default="neu")
    return parser


def main(argv: list[str] | None = None) -> int:
    options = _parser().parse_args(argv)
    try:
        if options.command == "export-dxf":
            write_editable_dxf(_read_json(options.manifest), options.output)
            print(f"editable DXF: {options.output}")
            return 0
        if options.command == "export-geo":
            write_editable_geo(_read_json(options.manifest), options.output)
            print(f"editable GEO: {options.output}")
            return 0
        mesh = _read_mesh(options.mesh)
        if options.command == "manifest":
            common = {
                "deformable_properties": tuple(options.deformable),
                "coordinate_unit": options.unit, "tolerance": options.tolerance,
            }
            if options.shape == "rectangle":
                result = create_rectangle_manifest(mesh, options.property, axis=options.axis, **common)
            elif options.shape == "l_prism":
                if options.notch is None:
                    raise MorphingError("--shape l_prism requires --notch X Y")
                result = create_l_prism_manifest(mesh, options.property, notch=tuple(options.notch), **common)
            elif options.shape == "cylinder":
                if options.axis_point is None or options.axis_direction is None:
                    raise MorphingError("--shape cylinder requires --axis-point and --axis-direction")
                result = create_cylinder_manifest(
                    mesh, options.property,
                    axis_point=tuple(options.axis_point), axis_direction=tuple(options.axis_direction),
                    **common,
                )
            elif options.shape == "octant_scale":
                result = create_octant_scale_manifest(mesh, options.property, **common)
            else:
                if options.inner_property is None or options.outer_property is None:
                    raise MorphingError("--shape octant_shell requires --inner-property and --outer-property")
                result = create_octant_shell_manifest(
                    mesh, options.property, inner_property=options.inner_property,
                    outer_property=options.outer_property, **common,
                )
            result["base_mesh_path"] = str(options.mesh.resolve())
            _write_json(options.output, result)
            print(f"manifest: {options.output} ({len(result['boundary_node_ids'])} boundary nodes)")
            return 0
        manifest = _read_json(options.manifest)
        if options.command == "apply":
            cases = _read_json(options.cases)
        elif options.command == "apply-dxf":
            cases = [{"name": options.name, **read_edited_dxf(manifest, options.edited_dxf)}]
        else:
            cases = [{"name": options.name, **read_edited_geo(
                manifest, options.edited_geo, gmsh_executable=options.gmsh_executable,
            )}]
        if not isinstance(cases, list) or not cases:
            raise MorphingError("cases file must contain a nonempty JSON array")
        names = [case.get("name") for case in cases if isinstance(case, dict)]
        if len(names) != len(cases) or any(not isinstance(name, str) or not name.isidentifier() for name in names):
            raise MorphingError("every case needs a unique identifier-like name")
        if len(set(names)) != len(names):
            raise MorphingError("case names must be unique")
        if "summary" in names:
            raise MorphingError("case name 'summary' is reserved for survey output")
        summary_json = options.output_dir / "summary.json"
        summary_csv = options.output_dir / "summary.csv"
        if summary_json.exists() or summary_csv.exists():
            raise MorphingError("refusing to overwrite an existing survey summary")
        failures = 0
        summaries: list[dict[str, Any]] = []
        for case in cases:
            result = None
            name = case["name"]
            output_mesh = options.output_dir / f"{name}.{options.format}"
            output_report = options.output_dir / f"{name}.json"
            if output_mesh.exists() or output_report.exists():
                raise MorphingError(f"refusing to overwrite existing case {name}")
            try:
                if manifest.get("shape") == "rectangle_2d":
                    result = apply_rectangle_case(
                        mesh, manifest,
                        translation=tuple(case.get("translation", (0.0, 0.0))),
                        thickness=case.get("thickness"), anchor=case.get("anchor", "center"),
                        interpolation=options.interpolation or "weighted_laplace",
                    )
                elif manifest.get("shape") in {"l_shape_2d", "l_prism_3d"}:
                    result = apply_l_prism_case(
                        mesh, manifest,
                        translation=tuple(case.get("translation", (0.0, 0.0))),
                        length_x=case.get("length_x"), length_y=case.get("length_y"),
                        arm_width_x=case.get("arm_width_x"), arm_width_y=case.get("arm_width_y"),
                        interpolation=options.interpolation or "weighted_laplace",
                    )
                elif manifest.get("shape") == "coaxial_cylinder_3d":
                    result = apply_cylinder_case(
                        mesh, manifest, inner_radius=case.get("inner_radius"),
                        outer_radius=case.get("outer_radius"), interpolation=options.interpolation or "weighted_laplace",
                    )
                elif manifest.get("shape") == "octant_scale_3d":
                    result = apply_octant_scale_case(
                        mesh, manifest, scale=case.get("scale"), displacement=case.get("displacement"),
                        interpolation=options.interpolation or "weighted_laplace",
                    )
                elif manifest.get("shape") == "octant_shell_3d":
                    result = apply_octant_shell_case(
                        mesh, manifest, inner_radius=case.get("inner_radius"),
                        outer_radius=case.get("outer_radius"),
                        inner_displacement=case.get("inner_displacement"),
                        outer_displacement=case.get("outer_displacement"),
                        interpolation=options.interpolation or "analytic_radial",
                    )
                else:
                    raise MorphingError(f"unsupported manifest shape: {manifest.get('shape')}")
                _write_verified_mesh(output_mesh, result.mesh)
                report = dict(result.report)
                report["name"] = name
                report["output_mesh"] = str(output_mesh)
                print(f"PASS {name}: {output_mesh}")
            except (MorphingError, ValueError) as error:
                report = {
                    "schema_version": 1, "status": "FAILED", "name": name,
                    "base_mesh_sha256": manifest.get("mesh_sha256"),
                    "parameters": case, "reason": str(error),
                }
                failures += 1
                print(f"FAILED {name}: {error}")
            _write_json(output_report, report)
            summaries.append({
                "name": name, "status": report["status"],
                "output_mesh": report.get("output_mesh", ""),
                "reason": report.get("reason", ""),
                "max_displacement": report.get("max_displacement"),
                "max_target_error": report.get("max_target_error"),
            })
            # Release each large morphed mesh before the next survey case.
            del result
            gc.collect()
        _write_json(summary_json, {
            "schema_version": 1, "base_mesh_sha256": manifest.get("mesh_sha256"),
            "case_count": len(cases), "failed_count": failures, "cases": summaries,
        })
        with summary_csv.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=(
                "name", "status", "output_mesh", "reason", "max_displacement", "max_target_error",
            ))
            writer.writeheader()
            writer.writerows(summaries)
        return 1 if failures else 0
    except (MorphingError, ValueError, OSError, KeyError, TypeError) as error:
        _parser().exit(2, f"error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
