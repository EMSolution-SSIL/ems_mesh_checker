"""Optional known-geometry tests for a separately built private native backend."""

from __future__ import annotations

from pathlib import Path

import meshio
import numpy as np
import pytest

from ems_mesh_checker.checks import check_mesh
from scripts.geometry_quality_test_models import geometry_cases


@pytest.mark.parametrize("case", geometry_cases(), ids=lambda case: case.name)
def test_synthetic_model_measure_and_classification(case) -> None:
    """Analytical dimensions and deliberate defects test the native kernel end to end."""
    backend = pytest.importorskip("ems_mesh_checker_native")
    metrics = backend.analyze_block(case.cell_type, case.points[np.newaxis, :, :])
    if case.expected_measure is not None:
        assert metrics["measure"][0] == pytest.approx(
            case.expected_measure, rel=1e-10, abs=1e-14,
        )
    report = check_mesh(
        case.mesh(), geometry_backend=backend, geometry_relative_tolerance=1e-6,
    )
    geometry_codes = {
        "NEAR_DEGENERATE_ELEMENT", "DEGENERATE_ELEMENT",
        "INVERTED_VOLUME_ELEMENT", "MIXED_JACOBIAN_SIGNS", "FOLDED_SURFACE_ELEMENT",
    }
    geometry_findings = [item for item in report.findings if item.code in geometry_codes]
    assert [item.code for item in geometry_findings] == (
        [] if case.expected_code is None else [case.expected_code]
    )
    assert report.checks["area_volume_degeneracy"].status == (
        "PASS" if case.expected_code is None else "FINDINGS"
    )
    assert report.geometry_quality["checked_elements"] == 1
    if geometry_findings:
        assert geometry_findings[0].element_id == 201
        assert geometry_findings[0].property_id == 3


@pytest.mark.parametrize("case", [
    case for case in geometry_cases() if case.name.endswith("_normal")
], ids=lambda case: case.name)
@pytest.mark.parametrize("factor", [1e-3, 1e3])
def test_analytical_measure_scales_with_length_unit(case, factor) -> None:
    backend = pytest.importorskip("ems_mesh_checker_native")
    baseline = backend.analyze_block(case.cell_type, case.points[np.newaxis, :, :])
    changed = backend.analyze_block(
        case.cell_type, (case.points * factor + [7 * factor, -11 * factor, 5 * factor])[np.newaxis, :, :],
    )
    dimension = 2 if case.cell_type in ("triangle", "quad") else 3
    assert changed["measure"][0] == pytest.approx(
        baseline["measure"][0] * factor ** dimension, rel=1e-10,
    )


@pytest.mark.parametrize("case", [
    case for case in geometry_cases() if case.name.endswith("_thin")
], ids=lambda case: case.name)
@pytest.mark.parametrize("factor", [1e-3, 1e3])
def test_near_degenerate_classification_is_unit_independent(case, factor) -> None:
    backend = pytest.importorskip("ems_mesh_checker_native")
    scaled_mesh = case.mesh()
    scaled_mesh.points *= factor
    report = check_mesh(
        scaled_mesh, geometry_backend=backend, geometry_relative_tolerance=1e-6,
    )
    assert any(item.code == "NEAR_DEGENERATE_ELEMENT" for item in report.findings)


@pytest.mark.parametrize("case", geometry_cases(), ids=lambda case: case.name)
def test_exported_synthetic_vtu_preserves_quality_result(case, tmp_path) -> None:
    backend = pytest.importorskip("ems_mesh_checker_native")
    path = tmp_path / f"{case.name}.vtu"
    meshio.write(path, case.mesh())
    reloaded = meshio.read(path)
    if case.expected_measure is not None:
        coordinates = reloaded.points[reloaded.cells[0].data]
        metrics = backend.analyze_block(case.cell_type, coordinates)
        assert metrics["measure"][0] == pytest.approx(
            case.expected_measure, rel=1e-10, abs=1e-14,
        )
    report = check_mesh(
        reloaded, geometry_backend=backend, geometry_relative_tolerance=1e-6,
    )
    findings = [
        item for item in report.findings
        if item.code in {
            "NEAR_DEGENERATE_ELEMENT", "DEGENERATE_ELEMENT",
            "INVERTED_VOLUME_ELEMENT", "MIXED_JACOBIAN_SIGNS", "FOLDED_SURFACE_ELEMENT",
        }
    ]
    assert [item.code for item in findings] == (
        [] if case.expected_code is None else [case.expected_code]
    )
    assert report.geometry_quality["checked_elements"] == 1
    if findings:
        assert (findings[0].element_id, findings[0].property_id) == (201, 3)


@pytest.mark.parametrize("relative_path", [
    "data/3D/IronCoil/post_geom_SDEFCOIL.neu",
    "data/3D/QuadTri/post_geom.neu",
])
def test_native_geometry_has_no_findings_in_reference_meshes(relative_path) -> None:
    backend = pytest.importorskip("ems_mesh_checker_native")
    converter = pytest.importorskip("ems_file_format_converter")
    path = Path(__file__).resolve().parents[1] / relative_path
    if not path.is_file():
        pytest.skip("reference mesh is not available locally")
    report = check_mesh(converter.read_mesh(path), geometry_backend=backend)
    assert report.checks["area_volume_degeneracy"].status == "PASS"
    assert report.geometry_quality["checked_elements"] == report.num_cells


@pytest.mark.parametrize(
    ("cell_type", "coordinates", "expected_measure"),
    [
        ("triangle", [[0, 0, 0], [1, 0, 0], [0, 1, 0]], 0.5),
        ("quad", [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], 1.0),
        ("tetra", [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], 1 / 6),
        ("hexahedron", [
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ], 1.0),
        ("wedge", [
            [0, 0, 0], [1, 0, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [0, 1, 1],
        ], 0.5),
        ("pyramid", [
            [-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0], [0, 0, 1],
        ], 4 / 3),
    ],
)
def test_native_metrics_for_reference_elements(cell_type, coordinates, expected_measure) -> None:
    backend = pytest.importorskip("ems_mesh_checker_native")
    metrics = backend.analyze_block(cell_type, np.array([coordinates], dtype=float))
    assert metrics["measure"][0] == pytest.approx(expected_measure, rel=1e-12)
    assert metrics["min_jacobian"][0] > 0
    assert metrics["scale"][0] > 0


def test_native_backend_detects_inverted_and_flat_tetrahedra() -> None:
    backend = pytest.importorskip("ems_mesh_checker_native")
    points = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [3, 0, 0], [4, 0, 0], [3, 1, 0], [3.2, 0.2, 0],
    ], dtype=float)
    mesh = meshio.Mesh(
        points=points,
        cells=[("tetra", np.array([[0, 2, 1, 3], [4, 5, 6, 7]]))],
        point_data={"id": np.arange(101, 109)},
        cell_data={"element_id": [np.array([201, 202])],
                   "property_id": [np.array([3, 3])]},
    )
    report = check_mesh(mesh, geometry_backend=backend)
    assert [(item.code, item.element_id) for item in report.findings] == [
        ("INVERTED_VOLUME_ELEMENT", 201),
        ("DEGENERATE_ELEMENT", 202),
    ]
    assert report.checks["area_volume_degeneracy"].status == "FINDINGS"


def test_native_backend_detects_folded_quad() -> None:
    backend = pytest.importorskip("ems_mesh_checker_native")
    mesh = meshio.Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float),
        cells=[("quad", np.array([[0, 1, 2, 3]]))],
        point_data={"id": np.arange(101, 105)},
        cell_data={"element_id": [np.array([201])], "property_id": [np.array([3])]},
    )
    report = check_mesh(mesh, geometry_backend=backend)
    assert any(item.code == "FOLDED_SURFACE_ELEMENT" for item in report.findings)


def test_native_backend_detects_mixed_jacobian_signs_in_hex() -> None:
    backend = pytest.importorskip("ems_mesh_checker_native")
    mesh = meshio.Mesh(
        points=np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [0.2, 0.2, -0.5], [0, 1, 1],
        ], dtype=float),
        cells=[("hexahedron", np.arange(8).reshape(1, 8))],
        point_data={"id": np.arange(101, 109)},
        cell_data={"element_id": [np.array([201])], "property_id": [np.array([3])]},
    )
    report = check_mesh(mesh, geometry_backend=backend)
    assert any(item.code == "MIXED_JACOBIAN_SIGNS" for item in report.findings)
