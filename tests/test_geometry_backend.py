from __future__ import annotations

from types import SimpleNamespace

import meshio
import numpy as np

from ems_mesh_checker.checks import check_mesh


def _four_tetra() -> meshio.Mesh:
    unit = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
    ])
    points = np.vstack([unit + [4.0 * index, 0.0, 0.0] for index in range(4)])
    cells = np.arange(16).reshape(4, 4)
    return meshio.Mesh(
        points=points,
        cells=[("tetra", cells)],
        point_data={"id": np.arange(101, 117)},
        cell_data={"element_id": [np.arange(201, 205)],
                   "property_id": [np.array([3, 3, 4, 4])]},
    )


def test_private_backend_metrics_are_classified_with_original_ids() -> None:
    backend = SimpleNamespace(
        __name__="fake_geometry_backend", __version__="test-1",
        SUPPORTED_CELL_TYPES=("tetra",),
        analyze_block=lambda cell_type, coordinates: {
            "measure": np.array([1 / 6, 1 / 6, 1 / 6, 1e-12]),
            "min_jacobian": np.array([1.0, -1.0, -1.0, 1e-12]),
            "max_jacobian": np.array([1.0, -1.0, 1.0, 1e-12]),
            "scale": np.ones(4),
        },
    )
    report = check_mesh(_four_tetra(), geometry_backend=backend)

    assert report.checks["area_volume_degeneracy"].status == "FINDINGS"
    assert [(item.code, item.element_id, item.property_id) for item in report.findings] == [
        ("INVERTED_VOLUME_ELEMENT", 202, 3),
        ("MIXED_JACOBIAN_SIGNS", 203, 4),
        ("NEAR_DEGENERATE_ELEMENT", 204, 4),
    ]
    assert report.to_dict()["geometry_quality"]["checked_elements"] == 4
    assert report.geometry_quality["backend_version"] == "test-1"


def test_missing_private_backend_is_not_reported_as_pass(monkeypatch) -> None:
    def missing(name: str):
        raise ModuleNotFoundError("No module named 'ems_mesh_checker_native'", name=name)

    monkeypatch.setattr("ems_mesh_checker.checks.geometry.importlib.import_module", missing)
    automatic = check_mesh(_four_tetra())
    required = check_mesh(_four_tetra(), geometry_backend="required")
    assert automatic.checks["area_volume_degeneracy"].status == "NOT_RUN"
    assert required.checks["area_volume_degeneracy"].status == "FAILED"
    assert any(item.code == "GEOMETRY_BACKEND_FAILED" for item in required.findings)


def test_malformed_private_backend_result_is_failed_not_pass() -> None:
    backend = SimpleNamespace(analyze_block=lambda *_: {"measure": np.ones(4)})
    report = check_mesh(_four_tetra(), geometry_backend=backend)
    assert report.checks["area_volume_degeneracy"].status == "FAILED"
    assert report.geometry_quality == {}


def test_backend_unsupported_types_are_counted_not_passed() -> None:
    backend = SimpleNamespace(
        SUPPORTED_CELL_TYPES=("triangle",),
        analyze_block=lambda *_: (_ for _ in ()).throw(AssertionError("must not run")),
    )
    report = check_mesh(_four_tetra(), geometry_backend=backend)
    assert report.checks["area_volume_degeneracy"].status == "UNSUPPORTED"
    assert report.geometry_quality["unsupported_elements"] == {"tetra": 4}
    assert report.geometry_quality["checked_elements"] == 0
