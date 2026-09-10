"""Extract representative geometry from an EMSolution-compatible mesh."""

from __future__ import annotations

import argparse
import gc
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from ems_file_format_converter import read_mesh
from ems_mesh_checker import (
    BoundaryExtractor,
    CurveSimplificationConfig,
    FeatureEdgeConfig,
    PlanarRegionConfig,
    RuledSurfaceConfig,
    VolumeTopologyConfig,
    extract_planar_regions,
    extract_property_group_feature_edges,
)
from ems_mesh_checker.export import (
    write_planar_regions_dxf,
    write_planar_regions_femap_neutral,
    write_planar_regions_geo,
)


@dataclass(frozen=True)
class ExportCase:
    name: str
    input_path: Path
    output_path: Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        help="input mesh (.atl, .neu, .unv, .msh, or another meshio format)",
    )
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        help=(
            "output DXF path; GEO and Femap NEU files use the same stem "
            "(default: <input>_feature_edges.dxf)"
        ),
    )
    parser.add_argument(
        "--feature-angle",
        type=float,
        default=30.0,
        help="VTK feature angle in degrees (default: 30)",
    )
    parser.add_argument(
        "--min-curve-length",
        type=float,
        default=0.0,
        help="discard only curves shorter than this explicit threshold (default: 0)",
    )
    parser.add_argument(
        "--coplanar-angle",
        type=float,
        default=1.0,
        help="maximum source-face normal angle for planar merging (default: 1 degree)",
    )
    parser.add_argument(
        "--mesh-size",
        type=float,
        help="Gmsh point mesh size; default is model bounding-box diagonal / 100",
    )
    parser.add_argument(
        "--circle-elements-per-turn",
        type=int,
        metavar="N",
        help=(
            "target circumferential discretization for reconstructed circles; "
            "each arc receives N * arc_angle / 360 elements"
        ),
    )
    parser.add_argument(
        "--gmsh-executable",
        type=Path,
        help="optional gmsh.exe path used to parse or mesh each representative GEO",
    )
    parser.add_argument(
        "--gmsh-mesh-dimension",
        type=int,
        choices=(2, 3),
        help="with --gmsh-executable, also generate a .msh using -2 or -3",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="also render a PyVista PNG preview with red feature edges",
    )
    parser.add_argument(
        "--no-arc-simplification",
        action="store_true",
        help="keep every representative loop as straight polyline segments",
    )
    parser.add_argument(
        "--arc-relative-tolerance",
        type=float,
        default=1.0e-4,
        help="maximum radial error relative to fitted radius (default: 1e-4)",
    )
    parser.add_argument(
        "--max-arc-sample-angle",
        type=float,
        default=30.0,
        help="largest source chord angle accepted as an arc sample (default: 30 degrees)",
    )
    parser.add_argument(
        "--no-ruled-surfaces",
        action="store_true",
        help="do not replace extruded circular side strips with four-curve surfaces",
    )
    parser.add_argument(
        "--no-volumes",
        action="store_true",
        help="do not assemble closed Property surfaces into Surface Loops and Volumes",
    )
    property_selection = parser.add_mutually_exclusive_group()
    property_selection.add_argument(
        "--properties",
        action="append",
        type=_parse_property_group,
        metavar="ID[,ID...]",
        help=(
            "export only these Property IDs; comma-separated values and repeated "
            "options are accepted (default: all Properties)"
        ),
    )
    property_selection.add_argument(
        "--exclude-properties",
        action="append",
        type=_parse_property_group,
        metavar="ID[,ID...]",
        help=(
            "export all except these Property IDs; comma-separated values and repeated "
            "options are accepted"
        ),
    )
    parser.add_argument(
        "--merge-properties",
        action="append",
        type=_parse_property_group,
        metavar="ID,ID[,...]",
        help=(
            "treat these Properties as one region and omit their mutual interfaces; "
            "repeat the option for independent groups"
        ),
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=500_000,
        metavar="N",
        help="report Femap node/element progress every N records; 0 disables it (default: 500000)",
    )
    parser.add_argument(
        "--suppress-quad-tri-interfaces",
        action="store_true",
        help=(
            "omit unambiguous same-Property hexahedron-quad/tetrahedron-two-triangle "
            "coupling surfaces while retaining cross-Property boundaries"
        ),
    )
    args = parser.parse_args()
    if args.gmsh_mesh_dimension is not None and args.gmsh_executable is None:
        parser.error("--gmsh-mesh-dimension requires --gmsh-executable")
    if args.circle_elements_per_turn is not None and args.circle_elements_per_turn < 3:
        parser.error("--circle-elements-per-turn must be at least 3")
    if args.progress_interval < 0:
        parser.error("--progress-interval must be non-negative")
    requested_properties = (
        None
        if not args.properties
        else tuple(sorted({value for group in args.properties for value in group}))
    )
    excluded_properties = (
        None
        if not args.exclude_properties
        else tuple(sorted({value for group in args.exclude_properties for value in group}))
    )
    merge_groups = tuple(args.merge_properties or ())
    try:
        _validate_merge_groups(merge_groups)
    except ValueError as error:
        parser.error(str(error))
    config = FeatureEdgeConfig(
        feature_angle_degrees=args.feature_angle,
        min_curve_length=args.min_curve_length,
    )
    planar_config = PlanarRegionConfig(coplanar_angle_degrees=args.coplanar_angle)
    curve_config = CurveSimplificationConfig(
        enabled=not args.no_arc_simplification,
        radial_relative_tolerance=args.arc_relative_tolerance,
        max_sample_angle_degrees=args.max_arc_sample_angle,
    )
    ruled_surface_config = RuledSurfaceConfig(enabled=not args.no_ruled_surfaces)
    volume_topology_config = VolumeTopologyConfig(enabled=not args.no_volumes)

    input_path = args.input.resolve()
    if not input_path.is_file():
        parser.error(f"input mesh does not exist: {input_path}")
    output_path = args.output
    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_feature_edges.dxf")
    elif output_path.suffix.lower() != ".dxf":
        parser.error("output path must use the .dxf extension")
    case = ExportCase(input_path.stem, input_path, output_path.resolve())

    for case in (case,):
        print(f"{case.name}: reading {case.input_path}", flush=True)
        mesh = read_mesh(
            case.input_path,
            progress=args.progress_interval > 0,
            progress_interval=max(args.progress_interval, 1),
        )
        print(f"{case.name}: extracting mesh boundaries", flush=True)
        boundaries = BoundaryExtractor(
            mesh,
            suppress_quad_tri_interfaces=args.suppress_quad_tri_interfaces,
        ).extract()
        if args.suppress_quad_tri_interfaces:
            print(
                f"{case.name}: suppressed_quad_tri_interfaces="
                f"{len(boundaries.suppressed_quad_tri_interfaces)} "
                f"faces={3 * len(boundaries.suppressed_quad_tri_interfaces)}",
                flush=True,
            )
        mesh_size = (
            args.mesh_size
            if args.mesh_size is not None
            else _automatic_mesh_size(boundaries.mesh.points)
        )
        results = []
        planar_results = []
        preview_plotter = _new_preview_plotter() if args.preview else None
        raw_edge_count = 0
        diagnostic_count = 0
        property_groups = _selected_property_groups(
            _property_ids(boundaries),
            requested_properties=requested_properties,
            excluded_properties=excluded_properties,
            merge_groups=merge_groups,
        )
        print(
            f"{case.name}: extracting {len(property_groups)} Property result set(s): "
            + ", ".join(_property_group_text(group) for group in property_groups),
            flush=True,
        )
        for group_index, property_group in enumerate(property_groups, start=1):
            print(
                f"{case.name}: Property result {group_index}/{len(property_groups)} "
                f"{_property_group_text(property_group)}",
                flush=True,
            )
            feature_result = extract_property_group_feature_edges(
                boundaries,
                property_group,
                config=config,
            )
            degenerate = [
                diagnostic
                for diagnostic in feature_result.diagnostics
                if diagnostic.startswith("degenerate_source_coordinates")
            ]
            if degenerate:
                raise RuntimeError(
                    f"{case.name}: source coordinates are degenerate; refusing to write a misleading DXF. "
                    f"First diagnostic: {degenerate[0]}"
                )
            planar_result = extract_planar_regions(feature_result, config=planar_config)
            if preview_plotter is not None:
                _add_preview_result(preview_plotter, feature_result)
            raw_edge_count += feature_result.edge_count
            diagnostic_count += len(feature_result.diagnostics)
            # DXF/GEO writers only need curves and identity after planar
            # assembly. Drop the potentially large VTK/source meshes before
            # processing the next Property.
            lightweight_feature = replace(
                feature_result,
                source_mesh=None,
                surface=None,
                edge_mesh=None,
            )
            results.append(lightweight_feature)
            planar_results.append(replace(planar_result, feature_edges=lightweight_feature))
            del feature_result, planar_result
            gc.collect()
        write_planar_regions_dxf(
            planar_results,
            case.output_path,
            curve_config=curve_config,
            ruled_surface_config=ruled_surface_config,
            volume_topology_config=volume_topology_config,
        )
        dxf_line_count, dxf_arc_count, dxf_polyline_count = _dxf_entity_counts(
            case.output_path
        )
        femap_neutral_path = write_planar_regions_femap_neutral(
            planar_results,
            case.output_path.with_suffix(".neu"),
            curve_config=curve_config,
            ruled_surface_config=ruled_surface_config,
            volume_topology_config=volume_topology_config,
        )
        geo_path = write_planar_regions_geo(
            planar_results,
            case.output_path.with_suffix(".geo"),
            mesh_size=mesh_size,
            curve_config=curve_config,
            ruled_surface_config=ruled_surface_config,
            volume_topology_config=volume_topology_config,
            circle_elements_per_turn=args.circle_elements_per_turn,
        )
        gmsh_mesh_path = None
        if args.gmsh_executable is not None:
            if args.gmsh_mesh_dimension is None:
                _verify_gmsh_parse(args.gmsh_executable, geo_path)
            else:
                gmsh_mesh_path = _generate_gmsh_mesh(
                    args.gmsh_executable,
                    geo_path,
                    dimension=args.gmsh_mesh_dimension,
                )
        preview_path = None
        if preview_plotter is not None:
            preview_path = _write_preview(preview_plotter, case.output_path.with_suffix(".png"))
        planar_count = sum(len(result.regions) for result in planar_results)
        unsupported_count = sum(len(result.unsupported_regions) for result in planar_results)
        (
            line_count,
            circle_count,
            plane_count,
            ruled_count,
            shell_count,
            volume_count,
            open_shell_count,
        ) = _geo_entity_counts(geo_path)
        print(
            f"{case.name}: {case.output_path}\n"
            f"  result_sets={len(results)}, raw_edges={raw_edge_count}, "
            f"curves={sum(len(result.curves) for result in results)}, diagnostics={diagnostic_count}\n"
            f"  representative_dxf={case.output_path}, lines={dxf_line_count}, "
            f"arcs={dxf_arc_count}, polylines={dxf_polyline_count}\n"
            f"  representative_femap_neutral={femap_neutral_path} "
            f"(point/line/arc geometry only)\n"
            f"  representative_geo={geo_path}, planar_regions={planar_count}, "
            f"unsupported_regions={unsupported_count}, mesh_size={mesh_size:.6g}, "
            f"lines={line_count}, circles={circle_count}, "
            f"plane_surfaces={plane_count}, ruled_surfaces={ruled_count}, "
            f"surface_loops={shell_count}, volumes={volume_count}, "
            f"open_shells={open_shell_count}"
            + (f"\n  gmsh_mesh={gmsh_mesh_path}" if gmsh_mesh_path is not None else "")
            + (f"\n  preview={preview_path}" if preview_path is not None else "")
        , flush=True)


def _property_ids(boundaries) -> tuple[int, ...]:
    property_ids = {
        int(face.owner.property_id)
        for face in boundaries.external_faces
        if face.owner.property_id is not None
    }
    for property_a, property_b in boundaries.property_interfaces:
        property_ids.update((property_a, property_b))
    return tuple(sorted(property_ids))


def _parse_property_group(value: str) -> tuple[int, ...]:
    try:
        property_ids = tuple(sorted({int(token.strip()) for token in value.split(",") if token.strip()}))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"Property IDs must be comma-separated integers: {value!r}"
        ) from error
    if not property_ids:
        raise argparse.ArgumentTypeError("at least one Property ID is required")
    if any(property_id < 0 for property_id in property_ids):
        raise argparse.ArgumentTypeError("Property IDs must be non-negative")
    return property_ids


def _validate_merge_groups(merge_groups: tuple[tuple[int, ...], ...]) -> None:
    claimed: set[int] = set()
    for group in merge_groups:
        if len(group) < 2:
            raise ValueError("--merge-properties requires at least two Property IDs")
        overlap = claimed.intersection(group)
        if overlap:
            raise ValueError(
                "--merge-properties groups must not overlap; repeated IDs: "
                f"{sorted(overlap)}"
            )
        claimed.update(group)


def _selected_property_groups(
    available_properties: tuple[int, ...],
    *,
    requested_properties: tuple[int, ...] | None,
    excluded_properties: tuple[int, ...] | None = None,
    merge_groups: tuple[tuple[int, ...], ...] = (),
) -> tuple[tuple[int, ...], ...]:
    available = set(available_properties)
    if requested_properties is None:
        excluded = set(excluded_properties or ())
        missing_excluded = sorted(excluded - available)
        if missing_excluded:
            raise ValueError(
                "excluded Property IDs are not present in the mesh: "
                f"{missing_excluded}"
            )
        selected = available - excluded
    else:
        selected = set(requested_properties)
        missing_requested = sorted(selected - available)
        if missing_requested:
            raise ValueError(
                "requested Property IDs are not present in the mesh: "
                f"{missing_requested}"
            )
    if not selected:
        raise ValueError("no Property IDs remain after applying the Property selection")
    merge_members = {property_id for group in merge_groups for property_id in group}
    missing_merge = sorted(merge_members - available)
    if missing_merge:
        raise ValueError(f"merged Property IDs are not present in the mesh: {missing_merge}")
    unselected_merge = sorted(merge_members - selected)
    if unselected_merge:
        raise ValueError(
            "merged Property IDs must remain selected by --properties/"
            "--exclude-properties: "
            f"{unselected_merge}"
        )
    groups = [group for group in merge_groups if set(group).issubset(selected)]
    groups.extend((property_id,) for property_id in sorted(selected - merge_members))
    return tuple(sorted(groups, key=lambda group: (group[0], len(group), group)))


def _property_group_text(group: tuple[int, ...]) -> str:
    if len(group) == 1:
        return f"P{group[0]}"
    return "GROUP[" + ",".join(f"P{property_id}" for property_id in group) + "]"


def _automatic_mesh_size(points) -> float:
    coordinates = np.asarray(points, dtype=float)
    diagonal = float(np.linalg.norm(np.ptp(coordinates, axis=0)))
    if diagonal <= 0.0:
        raise ValueError("cannot derive mesh size from degenerate model bounds")
    return diagonal / 100.0


def _verify_gmsh_parse(executable: Path, geo_path: Path) -> None:
    if not executable.is_file():
        raise FileNotFoundError(f"Gmsh executable does not exist: {executable}")
    completed = subprocess.run(
        [str(executable), str(geo_path), "-parse_and_exit", "-nopopup"],
        text=True,
        capture_output=True,
        check=False,
    )
    combined = f"{completed.stdout}\n{completed.stderr}"
    if completed.returncode != 0 or "Error" in combined:
        raise RuntimeError(f"Gmsh could not parse {geo_path}:\n{combined}")


def _generate_gmsh_mesh(executable: Path, geo_path: Path, *, dimension: int) -> Path:
    if not executable.is_file():
        raise FileNotFoundError(f"Gmsh executable does not exist: {executable}")
    mesh_path = geo_path.with_suffix(".msh")
    completed = subprocess.run(
        [
            str(executable),
            str(geo_path),
            f"-{dimension}",
            "-format",
            "msh4",
            "-o",
            str(mesh_path),
            "-nopopup",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    combined = f"{completed.stdout}\n{completed.stderr}"
    if completed.returncode != 0 or "Error" in combined:
        raise RuntimeError(f"Gmsh could not mesh {geo_path}:\n{combined}")
    if not mesh_path.is_file() or mesh_path.stat().st_size == 0:
        raise RuntimeError(f"Gmsh reported success but did not create {mesh_path}")
    return mesh_path


def _geo_entity_counts(path: Path) -> tuple[int, int, int, int, int, int, int]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return (
        sum(line.startswith("Line(") for line in lines),
        sum(line.startswith("Circle(") for line in lines),
        sum(line.startswith("Plane Surface(") for line in lines),
        sum(line.startswith("Surface(") for line in lines),
        sum(line.startswith("Surface Loop(") for line in lines),
        sum(line.startswith("Volume(") for line in lines),
        sum(line.startswith("// OPEN_SHELL") for line in lines),
    )


def _dxf_entity_counts(path: Path) -> tuple[int, int, int]:
    import ezdxf

    entities = list(ezdxf.readfile(path).modelspace())
    return (
        sum(entity.dxftype() == "LINE" for entity in entities),
        sum(entity.dxftype() == "ARC" for entity in entities),
        sum(entity.dxftype() in {"POLYLINE", "LWPOLYLINE"} for entity in entities),
    )


def _new_preview_plotter():
    import pyvista as pv

    plotter = pv.Plotter(off_screen=True, window_size=(1600, 1200))
    plotter.set_background("white")
    return plotter


def _add_preview_result(plotter, feature_result) -> None:
    plotter.add_mesh(
        feature_result.surface,
        color="#b8bec8",
        opacity=0.28,
        show_edges=False,
    )
    if feature_result.edge_mesh.n_cells:
        plotter.add_mesh(
            feature_result.edge_mesh,
            color="red",
            line_width=4,
            render_lines_as_tubes=True,
        )


def _write_preview(plotter, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    plotter.view_isometric()
    plotter.reset_camera()
    plotter.show(screenshot=str(path), auto_close=True)
    return path


if __name__ == "__main__":
    main()
