"""Evaluate strictly tagged editable GEO sketches through Gmsh, not text parsing."""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any

import numpy as np

from .editable_dxf import _parameters_from_lines, _segments
from .geometry_follow import MorphingError


def _gmsh_path(executable: str | Path | None) -> str:
    candidate = str(executable) if executable is not None else os.environ.get("GMSH_EXECUTABLE") or shutil.which("gmsh")
    if not candidate or not Path(candidate).is_file():
        raise MorphingError("Gmsh executable is required to evaluate edited GEO geometry")
    return candidate


def write_editable_geo(manifest: dict[str, Any], path: str | Path) -> Path:
    """Write numbered Points/Lines or quarter-circle arcs; no mesh surfaces."""

    target = Path(path)
    if target.exists():
        raise MorphingError(f"refusing to overwrite {target}")
    lines = ["// EMS mesh checker: preserve Point and curve numbers while editing coordinates."]
    def point(identifier: int, coordinates: tuple[float, float, float]) -> None:
        x, y, z = coordinates
        lines.append(f"Point({identifier}) = {{{x:.17g}, {y:.17g}, {z:.17g}, 1}};")
    if manifest.get("shape") == "coaxial_cylinder_3d":
        normal = np.asarray(manifest["axis_direction"], dtype=float)
        if not np.allclose(normal, (0., 0., 1.), atol=float(manifest["tolerance"]), rtol=0):
            raise MorphingError("editable cylinder GEO currently requires a +Z axis")
        center = tuple(map(float, manifest["axis_point"]))
        point(1, center)
        for offset, label in ((2, "inner_radius"), (6, "outer_radius")):
            radius = float(manifest[label])
            for index, direction in enumerate(((1., 0.), (0., 1.), (-1., 0.), (0., -1.))):
                point(offset + index, (center[0] + direction[0] * radius,
                                       center[1] + direction[1] * radius, center[2]))
            for index in range(4):
                start, end = offset + index, offset + (index + 1) % 4
                lines.append(f"Circle({offset - 1 + index}) = {{{start}, 1, {end}}};")
    else:
        segments = _segments(manifest)
        vertices = [start for start, _ in segments.values()]
        for index, (x, y) in enumerate(vertices, start=1):
            point(index, (float(x), float(y), 0.))
        for index, role in enumerate(segments, start=1):
            end = index % len(segments) + 1
            lines.append(f"Line({index}) = {{{index}, {end}}}; // {role}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def _evaluated_entities(
    path: str | Path, executable: str | Path | None,
) -> tuple[dict[int, np.ndarray], dict[int, set[int]], dict[int, tuple[str, tuple[int, ...]]]]:
    gmsh = _gmsh_path(executable)
    with tempfile.TemporaryDirectory(prefix="ems-morph-geo-") as folder:
        output = Path(folder) / "evaluated.msh"
        try:
            process = subprocess.run(
                [gmsh, "-0", str(Path(path).resolve()), "-format", "msh4", "-o", str(output), "-v", "2"],
                capture_output=True, text=True, timeout=60, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise MorphingError(f"Gmsh could not evaluate the edited GEO: {error}") from error
        if process.returncode != 0 or not output.is_file():
            raise MorphingError(f"Gmsh rejected the edited GEO: {(process.stderr or process.stdout)[-500:]}")
        text = output.read_text(encoding="utf-8")
        unrolled = Path(folder) / "evaluated.geo_unrolled"
        try:
            process = subprocess.run(
                [gmsh, "-0", str(Path(path).resolve()), "-format", "geo_unrolled",
                 "-o", str(unrolled), "-v", "2"],
                capture_output=True, text=True, timeout=60, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise MorphingError(f"Gmsh could not expose evaluated curve types: {error}") from error
        if process.returncode != 0 or not unrolled.is_file():
            raise MorphingError("Gmsh could not expose evaluated curve types")
        normalized = unrolled.read_text(encoding="utf-8")
    if "$Entities\n" not in text or "$EndEntities" not in text:
        raise MorphingError("Gmsh output has no evaluated Entities section")
    section = text.split("$Entities\n", 1)[1].split("$EndEntities", 1)[0].strip().splitlines()
    try:
        counts = [int(value) for value in section[0].split()]
        if len(counts) != 4 or counts[2:] != [0, 0] or len(section) != 1 + counts[0] + counts[1]:
            raise ValueError("unexpected point/curve/surface/volume entity counts")
        points: dict[int, np.ndarray] = {}
        for record in section[1:1 + counts[0]]:
            fields = record.split()
            identifier = int(fields[0])
            points[identifier] = np.asarray([float(value) for value in fields[1:4]])
        curves: dict[int, set[int]] = {}
        for record in section[1 + counts[0]:]:
            fields = record.split()
            identifier = int(fields[0])
            offset = 7
            physical_count = int(fields[offset])
            offset += 1 + physical_count
            bounding_count = int(fields[offset])
            bounds = {abs(int(value)) for value in fields[offset + 1:offset + 1 + bounding_count]}
            curves[identifier] = bounds
    except (ValueError, IndexError) as error:
        raise MorphingError(f"Gmsh Entities could not be interpreted: {error}") from error
    specifications: dict[int, tuple[str, tuple[int, ...]]] = {}
    for kind, identifier, node_text in re.findall(
        r"^\s*(Line|Circle)\((\d+)\)\s*=\s*\{([^}]*)\}\s*;",
        normalized, flags=re.MULTILINE,
    ):
        tag = int(identifier)
        if tag in specifications:
            raise MorphingError(f"Gmsh curve {tag} is defined more than once")
        try:
            specifications[tag] = (kind, tuple(int(value.strip()) for value in node_text.split(",")))
        except ValueError as error:
            raise MorphingError(f"Gmsh curve {tag} has unexpected point references") from error
    if set(specifications) != set(curves):
        raise MorphingError("Gmsh evaluated an unsupported or missing curve type")
    return points, curves, specifications


def read_edited_geo(
    manifest: dict[str, Any], path: str | Path, *, gmsh_executable: str | Path | None = None,
) -> dict[str, Any]:
    """Map Gmsh-evaluated, numbered entities back to supported case parameters."""

    points, curves, specifications = _evaluated_entities(path, gmsh_executable)
    tol = float(manifest["tolerance"])
    if manifest.get("shape") == "coaxial_cylinder_3d":
        if not np.allclose(manifest["axis_direction"], (0., 0., 1.), atol=tol, rtol=0):
            raise MorphingError("editable cylinder GEO currently requires a +Z axis")
        if set(points) != set(range(1, 10)) or set(curves) != set(range(1, 9)):
            raise MorphingError("cylinder GEO point/curve IDs were added, removed, or renumbered")
        for offset, curve_start in ((2, 1), (6, 5)):
            for index in range(4):
                expected = {offset + index, offset + (index + 1) % 4}
                if curves[curve_start + index] != expected:
                    raise MorphingError("cylinder GEO arc endpoint correspondence changed")
                kind, references = specifications[curve_start + index]
                if kind != "Circle" or len(references) != 3 or references[1] != 1 or {references[0], references[2]} != expected:
                    raise MorphingError("cylinder GEO arc type or center correspondence changed")
        center = np.asarray(manifest["axis_point"], dtype=float)
        if np.linalg.norm(points[1] - center) > tol:
            raise MorphingError("cylinder GEO axis center changed")
        radii = []
        for offset in (2, 6):
            measures = []
            for index, ray in enumerate(((1., 0.), (0., 1.), (-1., 0.), (0., -1.))):
                delta = points[offset + index] - center
                if abs(delta[2]) > tol or abs(ray[0] * delta[1] - ray[1] * delta[0]) > tol or np.dot(delta[:2], ray) <= 0:
                    raise MorphingError("cylinder GEO arc endpoint moved off its original radial ray")
                measures.append(float(np.linalg.norm(delta[:2])))
            if max(measures) - min(measures) > tol:
                raise MorphingError("cylinder GEO arc endpoints no longer share one radius")
            radii.append(float(np.mean(measures)))
        if radii[0] <= 0 or radii[1] <= radii[0]:
            raise MorphingError("edited radii must satisfy 0 < inner < outer")
        return {"inner_radius": radii[0], "outer_radius": radii[1]}

    segments = _segments(manifest)
    count = len(segments)
    if set(points) != set(range(1, count + 1)) or set(curves) != set(range(1, count + 1)):
        raise MorphingError("GEO point/curve IDs were added, removed, or renumbered")
    lines: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for index, role in enumerate(segments, start=1):
        end = index % count + 1
        if curves[index] != {index, end}:
            raise MorphingError(f"GEO curve {index} endpoint correspondence changed")
        kind, references = specifications[index]
        if kind != "Line" or len(references) != 2 or set(references) != {index, end}:
            raise MorphingError(f"GEO curve {index} is no longer the required Line")
        if max(abs(points[index][2]), abs(points[end][2])) > tol:
            raise MorphingError("2-D GEO points must remain in the XY plane")
        lines[role] = points[index], points[end]
    return _parameters_from_lines(manifest, lines)
