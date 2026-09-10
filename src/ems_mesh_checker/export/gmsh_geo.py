"""Convert the simple LINE/3DFACE DXF subset emitted by this project to GEO."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def dxf_to_geo(dxf_path: str | Path, geo_path: str | Path, *, mesh_size: float = 1.0) -> Path:
    """Convert project-generated DXF lines and faces into a Gmsh GEO script.

    Gmsh does not directly import DXF files.  The conversion preserves LINE entities
    as GEO curves and 3DFACE entities as individual planar GEO surfaces.  It is an
    interchange and validation step, not a replacement for a compact parametric CAD
    model; the latter should be authored separately from the resulting GEO topology.
    """

    import ezdxf

    output = Path(geo_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    document = ezdxf.readfile(dxf_path)
    writer = _GeoWriter(mesh_size)
    for entity in document.modelspace():
        entity_type = entity.dxftype()
        if entity_type == "LINE":
            writer.add_line(entity.dxf.start, entity.dxf.end)
        elif entity_type == "3DFACE":
            vertices = [entity.dxf.vtx0, entity.dxf.vtx1, entity.dxf.vtx2, entity.dxf.vtx3]
            if _point_key(vertices[3]) == _point_key(vertices[2]):
                vertices.pop()
            writer.add_face(vertices)
    output.write_text(writer.render(), encoding="utf-8", newline="\n")
    return output


class _GeoWriter:
    def __init__(self, mesh_size: float) -> None:
        self.mesh_size = mesh_size
        self.points: dict[tuple[float, float, float], int] = {}
        self.lines: dict[tuple[int, int], tuple[int, int, int]] = {}
        self.commands: list[str] = ["// Generated from EMS Mesh Checker DXF", "Geometry.AutoCoherence = 0;"]
        self.next_point = 1
        self.next_line = 1
        self.next_loop = 1
        self.next_surface = 1

    def add_line(self, start: Any, end: Any) -> int:
        start_tag = self._point(start)
        end_tag = self._point(end)
        return self._line(start_tag, end_tag)

    def add_face(self, vertices: list[Any]) -> None:
        point_tags = [self._point(vertex) for vertex in vertices]
        if len(point_tags) not in (3, 4) or len(set(point_tags)) != len(point_tags):
            return
        signed_lines = [
            self._line(point_tags[index], point_tags[(index + 1) % len(point_tags)])
            for index in range(len(point_tags))
        ]
        loop_tag = self.next_loop
        self.next_loop += 1
        surface_tag = self.next_surface
        self.next_surface += 1
        self.commands.append(f"Curve Loop({loop_tag}) = {{{', '.join(map(str, signed_lines))}}};")
        self.commands.append(f"Plane Surface({surface_tag}) = {{{loop_tag}}};")

    def _point(self, point: Any) -> int:
        key = _point_key(point)
        tag = self.points.get(key)
        if tag is not None:
            return tag
        tag = self.next_point
        self.next_point += 1
        self.points[key] = tag
        x, y, z = key
        self.commands.append(f"Point({tag}) = {{{x:.17g}, {y:.17g}, {z:.17g}, {self.mesh_size:.17g}}};")
        return tag

    def _line(self, start_tag: int, end_tag: int) -> int:
        key = tuple(sorted((start_tag, end_tag)))
        line = self.lines.get(key)
        if line is not None:
            tag, stored_start, stored_end = line
            return tag if (stored_start, stored_end) == (start_tag, end_tag) else -tag
        tag = self.next_line
        self.next_line += 1
        self.lines[key] = (tag, start_tag, end_tag)
        self.commands.append(f"Line({tag}) = {{{start_tag}, {end_tag}}};")
        return tag

    def render(self) -> str:
        self.commands.append("")
        return "\n".join(self.commands)


def _point_key(point: Any) -> tuple[float, float, float]:
    values = tuple(float(value) for value in point)
    return (values[0], values[1], values[2] if len(values) > 2 else 0.0)
