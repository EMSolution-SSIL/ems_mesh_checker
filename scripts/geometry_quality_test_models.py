"""Small, synthetic linear-element meshes with analytically known measures.

The output meshes are local test artifacts, not distribution sample data.
Run ``python scripts/geometry_quality_test_models.py OUTPUT_DIRECTORY`` to
write one VTU file per case for inspection in a mesh viewer.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import meshio
import numpy as np


@dataclass(frozen=True)
class GeometryCase:
    name: str
    cell_type: str
    points: np.ndarray
    expected_measure: float | None
    expected_code: str | None

    def mesh(self) -> meshio.Mesh:
        return meshio.Mesh(
            points=self.points,
            cells=[(self.cell_type, np.arange(len(self.points)).reshape(1, -1))],
            point_data={"id": np.arange(101, 101 + len(self.points))},
            cell_data={"element_id": [np.array([201])], "property_id": [np.array([3])]},
        )


def geometry_cases() -> tuple[GeometryCase, ...]:
    """Return normal, near-flat, flat and inverted single-element test meshes."""
    references = {
        "triangle": ([[0, 0, 0], [2, 0, 0], [0, 3, 0]], 3.0),
        "quad": ([[0, 0, 0], [2, 0, 0], [2, 3, 0], [0, 3, 0]], 6.0),
        "tetra": ([[0, 0, 0], [2, 0, 0], [0, 3, 0], [0, 0, 4]], 4.0),
        "hexahedron": (
            [[0, 0, 0], [2, 0, 0], [2, 3, 0], [0, 3, 0],
             [0, 0, 4], [2, 0, 4], [2, 3, 4], [0, 3, 4]], 24.0,
        ),
        "wedge": (
            [[0, 0, 0], [2, 0, 0], [0, 3, 0],
             [0, 0, 4], [2, 0, 4], [0, 3, 4]], 12.0,
        ),
        "pyramid": (
            [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0], [0, 0, 3]], 4.0,
        ),
    }
    inverted_order = {
        "tetra": [0, 2, 1, 3],
        "hexahedron": [0, 3, 2, 1, 4, 7, 6, 5],
        "wedge": [0, 2, 1, 3, 5, 4],
        "pyramid": [0, 3, 2, 1, 4],
    }
    cases: list[GeometryCase] = []
    for cell_type, (coordinates, measure) in references.items():
        points = np.asarray(coordinates, dtype=float)
        cases.append(GeometryCase(f"{cell_type}_normal", cell_type, points, measure, None))
        thin = points.copy()
        thin[:, 1 if cell_type in ("triangle", "quad") else 2] *= 1e-8
        cases.append(GeometryCase(
            f"{cell_type}_thin", cell_type, thin, measure * 1e-8,
            "NEAR_DEGENERATE_ELEMENT",
        ))
        flat = points.copy()
        flat[:, 1 if cell_type in ("triangle", "quad") else 2] = 0
        cases.append(GeometryCase(
            f"{cell_type}_flat", cell_type, flat, 0.0, "DEGENERATE_ELEMENT",
        ))
        if cell_type in inverted_order:
            cases.append(GeometryCase(
                f"{cell_type}_inverted", cell_type,
                points[inverted_order[cell_type]], measure, "INVERTED_VOLUME_ELEMENT",
            ))
    cases.append(GeometryCase(
        "quad_folded", "quad",
        np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float),
        None, "FOLDED_SURFACE_ELEMENT",
    ))
    cases.append(GeometryCase(
        "hexahedron_twisted", "hexahedron",
        np.asarray([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [0.2, 0.2, -0.5], [0, 1, 1],
        ], dtype=float), None, "MIXED_JACOBIAN_SIGNS",
    ))
    return tuple(cases)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    for case in geometry_cases():
        meshio.write(args.output_directory / f"{case.name}.vtu", case.mesh())
    print(f"Wrote {len(geometry_cases())} synthetic VTU meshes to {args.output_directory}")


if __name__ == "__main__":
    main()
