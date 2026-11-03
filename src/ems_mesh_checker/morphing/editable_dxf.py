"""Strict DXF sketches for the parameterized geometry-following shapes.

The DXF is a view/edit surface, not the correspondence authority. A sketch is
accepted only when its named primitives still define the same shape topology.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .geometry_follow import MorphingError


def _ezdxf():
    try:
        import ezdxf
    except ImportError as error:
        raise MorphingError("install ezdxf to use editable DXF sketches") from error
    return ezdxf


def _segments(manifest: dict[str, Any]) -> dict[str, tuple[tuple[float, float], tuple[float, float]]]:
    minimum = manifest["minimum"]
    maximum = manifest["maximum"]
    x0, y0 = map(float, minimum[:2])
    x1, y1 = map(float, maximum[:2])
    shape = manifest["shape"]
    if shape == "rectangle_2d":
        vertices = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
        names = ("BOTTOM", "RIGHT", "TOP", "LEFT")
    elif shape in {"l_shape_2d", "l_prism_3d"}:
        notch_x, notch_y = map(float, manifest["notch"])
        vertices = ((x0, y0), (x1, y0), (x1, notch_y),
                    (notch_x, notch_y), (notch_x, y1), (x0, y1))
        names = ("BOTTOM", "RIGHT", "INNER_H", "INNER_V", "TOP", "LEFT")
    else:
        raise MorphingError("editable line DXF supports rectangle and L prism only")
    return {name: (vertices[index], vertices[(index + 1) % len(vertices)])
            for index, name in enumerate(names)}


def write_editable_dxf(manifest: dict[str, Any], path: str | Path) -> Path:
    """Write only named LINE/CIRCLE entities; no surface or volume entities."""

    ezdxf = _ezdxf()
    target = Path(path)
    if target.exists():
        raise MorphingError(f"refusing to overwrite {target}")
    document = ezdxf.new("R2010")
    space = document.modelspace()
    if manifest.get("shape") == "coaxial_cylinder_3d":
        center = tuple(manifest["axis_point"])
        normal = tuple(manifest["axis_direction"])
        if not np.allclose(normal, (0., 0., 1.), atol=float(manifest["tolerance"]), rtol=0):
            raise MorphingError("editable cylinder DXF currently requires a +Z axis")
        for role, radius in (("INNER", manifest["inner_radius"]), ("OUTER", manifest["outer_radius"])):
            layer = f"EMS_MORPH_{role}"
            document.layers.new(layer)
            space.add_circle(center, float(radius), dxfattribs={"layer": layer, "extrusion": normal})
    else:
        for role, (start, end) in _segments(manifest).items():
            layer = f"EMS_MORPH_{role}"
            document.layers.new(layer)
            space.add_line((*start, 0.0), (*end, 0.0), dxfattribs={"layer": layer})
    target.parent.mkdir(parents=True, exist_ok=True)
    document.saveas(target)
    return target


def read_edited_dxf(manifest: dict[str, Any], path: str | Path) -> dict[str, Any]:
    """Return a parameter case or reject added, removed, or ambiguous entities."""

    ezdxf = _ezdxf()
    document = ezdxf.readfile(path)
    entities = list(document.modelspace())
    tol = float(manifest["tolerance"])
    if manifest.get("shape") == "coaxial_cylinder_3d":
        if not np.allclose(manifest["axis_direction"], (0., 0., 1.), atol=tol, rtol=0):
            raise MorphingError("editable cylinder DXF currently requires a +Z axis")
        expected = {"INNER", "OUTER"}
        if len(entities) != 2 or any(item.dxftype() != "CIRCLE" for item in entities):
            raise MorphingError("cylinder DXF must contain exactly two CIRCLE entities")
        circles = {item.dxf.layer.removeprefix("EMS_MORPH_"): item for item in entities}
        if set(circles) != expected:
            raise MorphingError("cylinder DXF inner/outer layers are missing or duplicated")
        center = np.asarray(manifest["axis_point"], dtype=float)
        normal = np.asarray(manifest["axis_direction"], dtype=float)
        for item in circles.values():
            if np.linalg.norm(np.asarray(item.dxf.center, dtype=float) - center) > tol:
                raise MorphingError("moving the cylinder axis is not supported")
            if np.linalg.norm(np.asarray(item.dxf.extrusion, dtype=float) - normal) > tol:
                raise MorphingError("changing the cylinder axis direction is not supported")
        inner, outer = (float(circles[name].dxf.radius) for name in ("INNER", "OUTER"))
        if inner <= 0 or outer <= inner:
            raise MorphingError("edited radii must satisfy 0 < inner < outer")
        return {"inner_radius": inner, "outer_radius": outer}

    reference = _segments(manifest)
    if len(entities) != len(reference) or any(item.dxftype() != "LINE" for item in entities):
        raise MorphingError("DXF must retain exactly the original named LINE entities")
    lines: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for entity in entities:
        role = entity.dxf.layer.removeprefix("EMS_MORPH_")
        if role not in reference or role in lines or entity.dxf.layer != f"EMS_MORPH_{role}":
            raise MorphingError("DXF line layer is unknown or duplicated")
        first, second = (np.asarray(getattr(entity.dxf, field), dtype=float) for field in ("start", "end"))
        if max(abs(first[2]), abs(second[2])) > tol:
            raise MorphingError("2-D DXF line nodes must remain in the XY plane")
        lines[role] = first, second
    return _parameters_from_lines(manifest, lines)


def _parameters_from_lines(
    manifest: dict[str, Any], lines: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    tol = float(manifest["tolerance"])
    vertices = np.asarray([point[:2] for segment in lines.values() for point in segment])
    minimum, maximum = vertices.min(axis=0), vertices.max(axis=0)
    if np.any(maximum - minimum <= tol):
        raise MorphingError("edited geometry has a collapsed outline")
    if manifest["shape"] == "rectangle_2d":
        x0, y0 = minimum
        x1, y1 = maximum
        proposed = {"BOTTOM": ((x0, y0), (x1, y0)), "RIGHT": ((x1, y0), (x1, y1)),
                    "TOP": ((x1, y1), (x0, y1)), "LEFT": ((x0, y1), (x0, y0))}
        old_min, old_max = np.asarray(manifest["minimum"]), np.asarray(manifest["maximum"])
        other = 1 if manifest["thickness_axis"] == "x" else 0
        if abs((maximum - minimum)[other] - (old_max - old_min)[other]) > tol:
            raise MorphingError("editing the non-thickness rectangle dimension is not supported")
        parameters = {
            "translation": (minimum - old_min).tolist(),
            "thickness": float((maximum - minimum)[1 - other]),
            "anchor": "min",
        }
    else:
        inner_h = lines["INNER_H"]
        inner_v = lines["INNER_V"]
        shared = [point[:2] for point in inner_h if any(np.linalg.norm(point[:2] - other[:2]) <= tol for other in inner_v)]
        if len(shared) != 1:
            raise MorphingError("L geometry has no unique concave corner")
        notch_x, notch_y = shared[0]
        x0, y0 = minimum
        x1, y1 = maximum
        if not x0 + tol < notch_x < x1 - tol or not y0 + tol < notch_y < y1 - tol:
            raise MorphingError("L geometry concave corner is outside the arms")
        proposed = {"BOTTOM": ((x0, y0), (x1, y0)),
                    "RIGHT": ((x1, y0), (x1, notch_y)),
                    "INNER_H": ((x1, notch_y), (notch_x, notch_y)),
                    "INNER_V": ((notch_x, notch_y), (notch_x, y1)),
                    "TOP": ((notch_x, y1), (x0, y1)),
                    "LEFT": ((x0, y1), (x0, y0))}
        parameters = {
            "translation": (minimum - np.asarray(manifest["minimum"])).tolist(),
            "length_x": float(x1 - x0), "length_y": float(y1 - y0),
            "arm_width_x": float(notch_x - x0), "arm_width_y": float(notch_y - y0),
        }
    for role, expected_segment in proposed.items():
        actual = lines[role]
        forward = all(np.linalg.norm(actual[i][:2] - expected_segment[i]) <= tol for i in (0, 1))
        backward = all(np.linalg.norm(actual[i][:2] - expected_segment[1 - i]) <= tol for i in (0, 1))
        if not forward and not backward:
            raise MorphingError(f"segment {role} has changed topology or is not axis-aligned")
    return parameters
