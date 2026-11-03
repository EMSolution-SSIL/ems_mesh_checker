"""Check mesh integrity, topology and optional geometry quality without modifying it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .checks import MeshCheckError, check_mesh


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="input mesh readable by ems-file-format-converter")
    parser.add_argument("--report", type=Path, help="JSON report (default: <input>_check.json)")
    parser.add_argument("--unit", default="model coordinate unit", help="coordinate unit label; no unit is inferred")
    parser.add_argument(
        "--dimension-spec", type=Path,
        help="optional JSON design limits for model and Property dimensions",
    )
    parser.add_argument(
        "--coordinate-tolerance", type=float, default=0.0,
        help="distance tolerance for coincident-node candidates (default: exact coordinates)",
    )
    parser.add_argument(
        "--free-faces-report", type=Path,
        help="optional full JSON list of unmodified free faces and their original IDs",
    )
    parser.add_argument(
        "--surface-owner", action="append", default=[], metavar="FACE_PROPERTY:VOLUME_PROPERTY",
        help="select the volume side used to check normals on a Property interface",
    )
    parser.add_argument(
        "--surface-normal-direction", choices=("outward", "inward"), default="outward",
        help="expected surface normal relative to the selected volume (default: outward)",
    )
    parser.add_argument(
        "--surface-normal", action="append", default=[], metavar="FACE_PROPERTY:DIRECTION",
        help="override the expected direction for one surface Property (inward or outward)",
    )
    parser.add_argument(
        "--geometry-backend", choices=("auto", "off", "required"), default="auto",
        help="use an installed private geometry backend when available (default: auto)",
    )
    parser.add_argument(
        "--geometry-relative-tolerance", type=float, default=1e-10, metavar="VALUE",
        help="relative Jacobian warning threshold for the private backend",
    )
    parser.add_argument(
        "--progress-interval", type=int, default=500_000, metavar="N",
        help="report mesh-reader progress every N records; 0 disables it",
    )
    parser.add_argument(
        "--fail-on", choices=("none", "error", "warning"), default="none",
        help="return exit code 1 when findings reach this severity (default: none)",
    )
    args = parser.parse_args(argv)
    if args.progress_interval < 0:
        parser.error("--progress-interval must be non-negative")
    if not args.unit:
        parser.error("--unit must not be empty")
    if args.coordinate_tolerance < 0:
        parser.error("--coordinate-tolerance must be non-negative")
    if not 0 <= args.geometry_relative_tolerance < float("inf"):
        parser.error("--geometry-relative-tolerance must be finite and non-negative")
    surface_owner_properties: dict[int, int] = {}
    for value in args.surface_owner:
        try:
            face_property, volume_property = (int(token) for token in value.split(":"))
        except ValueError:
            parser.error(f"invalid --surface-owner {value!r}; expected FACE_PROPERTY:VOLUME_PROPERTY")
        if face_property in surface_owner_properties and surface_owner_properties[face_property] != volume_property:
            parser.error(f"conflicting owner Property for face Property {face_property}")
        surface_owner_properties[face_property] = volume_property
    surface_normal_directions: dict[int, str] = {}
    for value in args.surface_normal:
        parts = value.split(":")
        if len(parts) != 2 or parts[1] not in {"outward", "inward"}:
            parser.error(f"invalid --surface-normal {value!r}; expected FACE_PROPERTY:inward|outward")
        try:
            face_property = int(parts[0])
        except ValueError:
            parser.error(f"invalid --surface-normal {value!r}; expected FACE_PROPERTY:inward|outward")
        if face_property in surface_normal_directions and surface_normal_directions[face_property] != parts[1]:
            parser.error(f"conflicting normal directions for face Property {face_property}")
        surface_normal_directions[face_property] = parts[1]
    input_path = args.input.resolve()
    if not input_path.is_file():
        parser.error(f"input mesh does not exist: {input_path}")
    report_path = (args.report or input_path.with_name(f"{input_path.stem}_check.json")).resolve()
    if report_path == input_path:
        parser.error("report path must differ from the input mesh")
    if args.free_faces_report is not None and args.free_faces_report.resolve() in {input_path, report_path}:
        parser.error("free-face report path must differ from the input mesh and main report")
    if args.dimension_spec is not None:
        spec_path = args.dimension_spec.resolve()
        if not spec_path.is_file():
            parser.error(f"dimension specification does not exist: {spec_path}")
        if spec_path in {report_path, args.free_faces_report.resolve() if args.free_faces_report else None}:
            parser.error("dimension specification must differ from output reports")

    try:
        from ems_file_format_converter import read_mesh

        dimension_spec = None
        if args.dimension_spec is not None:
            dimension_spec = json.loads(args.dimension_spec.read_text(encoding="utf-8"))
        mesh = read_mesh(
            input_path,
            progress=args.progress_interval > 0,
            progress_interval=max(args.progress_interval, 1),
        )
        report = check_mesh(
            mesh, source=str(input_path), coordinate_unit=args.unit,
            coordinate_tolerance=args.coordinate_tolerance,
            surface_owner_properties=surface_owner_properties,
            surface_normal_direction=args.surface_normal_direction,
            surface_normal_directions=surface_normal_directions,
            geometry_backend=args.geometry_backend,
            geometry_relative_tolerance=args.geometry_relative_tolerance,
            dimension_spec=dimension_spec,
        )
        report.write_json(report_path)
        if args.free_faces_report is not None:
            report.write_free_faces_json(args.free_faces_report)
    except (ImportError, MeshCheckError, OSError, RuntimeError, ValueError) as error:
        parser.exit(2, f"mesh check failed: {error}\n")

    print(report.summary_text())
    print(f"report: {report_path}")
    if args.free_faces_report is not None:
        print(f"free faces: {args.free_faces_report.resolve()}")
    if any(result.status == "FAILED" for result in report.checks.values()):
        return 2
    if args.fail_on == "error" and any(item.severity == "ERROR" for item in report.findings):
        return 1
    if args.fail_on == "warning" and any(item.severity in {"ERROR", "WARNING"} for item in report.findings):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
