"""Stable, JSON-serializable results for read-only mesh checks."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


class MeshCheckError(ValueError):
    """The input cannot be inspected safely as a mesh."""


@dataclass(frozen=True)
class CheckFinding:
    code: str
    severity: str
    message: str
    node_ids: tuple[int, ...] = ()
    element_id: int | None = None
    related_element_ids: tuple[int, ...] = ()
    property_id: int | None = None
    local_node_index: int | None = None
    cell_block_index: int | None = None
    local_element_index: int | None = None
    coordinates: tuple[float, ...] | None = None
    measured_value: float | None = None


@dataclass(frozen=True)
class CheckResult:
    status: str
    reason: str | None = None
    finding_count: int = 0


@dataclass
class CheckReport:
    source: str | None
    num_points: int
    num_cells: int
    cell_types: dict[str, int]
    coordinate_unit: str = "model coordinate unit"
    findings: list[CheckFinding] = field(default_factory=list)
    checks: dict[str, CheckResult] = field(default_factory=dict)
    dimensions: dict[str, Any] = field(default_factory=dict)
    topology: dict[str, Any] = field(default_factory=dict)
    geometry_quality: dict[str, Any] = field(default_factory=dict)
    raw_free_faces: list[dict[str, Any]] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "source": self.source,
            "summary": {
                "num_points": self.num_points,
                "num_cells": self.num_cells,
                "cell_types": dict(sorted(self.cell_types.items())),
                "coordinate_unit": self.coordinate_unit,
                "finding_counts": {
                    severity: sum(item.severity == severity for item in self.findings)
                    for severity in ("ERROR", "WARNING", "INFO")
                },
            },
            "checks": {name: asdict(result) for name, result in sorted(self.checks.items())},
            "dimensions": self.dimensions,
            "topology": self.topology,
            "geometry_quality": self.geometry_quality,
            "findings": [asdict(item) for item in self.findings],
        }

    def write_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return target

    def write_free_faces_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {"schema_version": 1, "source": self.source, "faces": self.raw_free_faces},
                ensure_ascii=False, indent=2, allow_nan=False,
            ) + "\n",
            encoding="utf-8",
        )
        return target

    def summary_text(self) -> str:
        counts = self.to_dict()["summary"]["finding_counts"]
        lines = [
            f"mesh: {self.source or '<in-memory>'}",
            f"points={self.num_points} cells={self.num_cells} "
            f"errors={counts['ERROR']} warnings={counts['WARNING']} info={counts['INFO']}",
        ]
        model = self.dimensions.get("model")
        if model is not None:
            lines.append(
                f"bounds: min={model['minimum']} max={model['maximum']} size={model['size']} "
                f"({self.coordinate_unit})"
            )
        if self.topology:
            lines.append(
                f"free_faces={self.topology['free_face_count']} "
                f"patches={len(self.topology['free_face_patches'])} "
                f"coincident_unshared_face_groups={self.topology['coincident_unshared_face_group_count']}"
            )
        if self.geometry_quality:
            quality = self.geometry_quality
            lines.append(
                f"geometry_backend={quality['backend']} {quality['backend_version']} "
                f"checked={quality['checked_elements']} "
                f"unsupported={sum(quality['unsupported_elements'].values())}"
            )
        lines.extend(f"{name}: {item.status}" for name, item in sorted(self.checks.items()))
        return "\n".join(lines)
