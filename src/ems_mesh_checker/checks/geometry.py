"""Optional, solver-independent bridge to a private geometry-quality backend.

The backend accepts a cell type and an (elements, nodes, 3) float64 array and
returns four per-element arrays: measure, min_jacobian, max_jacobian, and scale.
No solver source or binary is part of this public package.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

from ems_mesh_checker.topology.element_topology import get_element_topology

from .report import CheckFinding, CheckReport, CheckResult


_INVALID_CODES = {
    "INVALID_CONNECTIVITY", "INVALID_NODE_REFERENCE", "INVALID_CELL_METADATA",
    "REPEATED_ELEMENT_NODE", "DUPLICATE_ELEMENT", "NONFINITE_COORDINATE",
}
_METRICS = ("measure", "min_jacobian", "max_jacobian", "scale")
_CHUNK_SIZE = 50_000


def _metadata(mesh: Any, name: str, block_index: int) -> np.ndarray | None:
    values = getattr(mesh, "cell_data", {}).get(name)
    if values is None or len(values) != len(mesh.cells):
        return None
    array = np.asarray(values[block_index])
    if array.ndim != 1 or len(array) != len(mesh.cells[block_index].data):
        return None
    return array


def _backend_name(backend: Any) -> str:
    return str(getattr(backend, "__name__", type(backend).__name__))


def _fail(report: CheckReport, message: str) -> None:
    report.findings.append(CheckFinding("GEOMETRY_BACKEND_FAILED", "ERROR", message))
    report.checks["area_volume_degeneracy"] = CheckResult("FAILED", message, 1)


def add_geometry_checks(
    mesh: Any,
    report: CheckReport,
    *,
    backend: str | Any = "auto",
    relative_tolerance: float = 1e-10,
) -> None:
    """Interpret signed Jacobians without treating a missing backend as PASS."""

    if not np.isfinite(relative_tolerance) or relative_tolerance < 0:
        raise ValueError("geometry relative_tolerance must be finite and non-negative")
    if isinstance(backend, str):
        if backend not in {"auto", "off", "required"}:
            raise ValueError("geometry backend must be auto, off, or required")
        if backend == "off":
            report.checks["area_volume_degeneracy"] = CheckResult("NOT_RUN", "geometry backend disabled")
            return
        try:
            backend_module = importlib.import_module("ems_mesh_checker_native")
        except ModuleNotFoundError as error:
            if error.name == "ems_mesh_checker_native" and backend == "auto":
                report.checks["area_volume_degeneracy"] = CheckResult(
                    "NOT_RUN", "private geometry backend is not installed",
                )
                return
            _fail(report, f"private geometry backend could not be imported: {error}")
            return
        except (ImportError, OSError) as error:
            _fail(report, f"private geometry backend could not be imported: {error}")
            return
    else:
        backend_module = backend
    if not callable(getattr(backend_module, "analyze_block", None)):
        _fail(report, "geometry backend has no analyze_block function")
        return
    if any(item.code in _INVALID_CODES for item in report.findings):
        report.checks["area_volume_degeneracy"] = CheckResult(
            "NOT_RUN", "invalid mesh input prevents a reliable geometry check",
        )
        return

    points = np.asarray(mesh.points, dtype=float)
    if points.shape[1] == 2:
        points = np.column_stack((points, np.zeros(len(points))))
    supported_types = getattr(backend_module, "SUPPORTED_CELL_TYPES", None)
    supported_types = None if supported_types is None else set(supported_types)
    findings: list[CheckFinding] = []
    checked_count = 0
    unsupported: dict[str, int] = {}
    minimum_scaled_jacobian = float("inf")
    try:
        for block_index, block in enumerate(mesh.cells):
            cell_type = str(block.type)
            topology = get_element_topology(cell_type)
            if topology is None or topology.dimension not in (2, 3) or not len(block.data):
                continue
            if supported_types is not None and cell_type not in supported_types:
                unsupported[cell_type] = unsupported.get(cell_type, 0) + len(block.data)
                continue
            ids = _metadata(mesh, "element_id", block_index)
            properties = _metadata(mesh, "property_id", block_index)
            for start in range(0, len(block.data), _CHUNK_SIZE):
                connectivity = np.asarray(block.data[start:start + _CHUNK_SIZE], dtype=np.intp)
                coordinates = np.ascontiguousarray(points[connectivity], dtype=np.float64)
                try:
                    raw = backend_module.analyze_block(cell_type, coordinates)
                except NotImplementedError:
                    unsupported[cell_type] = unsupported.get(cell_type, 0) + len(block.data) - start
                    break
                if not isinstance(raw, dict) or any(name not in raw for name in _METRICS):
                    raise ValueError(f"{cell_type}: backend result lacks required metrics")
                metrics = {name: np.asarray(raw[name], dtype=float) for name in _METRICS}
                if any(values.shape != (len(connectivity),) or not np.isfinite(values).all()
                       for values in metrics.values()):
                    raise ValueError(f"{cell_type}: backend metrics must be finite per-element arrays")
                for offset in range(len(connectivity)):
                    measure = float(metrics["measure"][offset])
                    lower = float(metrics["min_jacobian"][offset])
                    upper = float(metrics["max_jacobian"][offset])
                    scale = float(metrics["scale"][offset])
                    if measure < 0 or scale < 0 or lower > upper:
                        raise ValueError(f"{cell_type}: backend returned inconsistent geometry metrics")
                    denominator = scale ** topology.dimension if scale > 0 else 0.0
                    scaled = (
                        0.0 if lower <= 0 <= upper or denominator == 0 else
                        min(abs(lower), abs(upper)) / denominator
                    )
                    minimum_scaled_jacobian = min(minimum_scaled_jacobian, scaled)
                    zero_limit = 64 * np.finfo(float).eps * denominator
                    if topology.dimension == 3 and lower < -zero_limit and upper > zero_limit:
                        code, severity, message = "MIXED_JACOBIAN_SIGNS", "ERROR", "Jacobian changes sign inside the volume element"
                    elif topology.dimension == 2 and lower < -zero_limit and upper > zero_limit:
                        code, severity, message = "FOLDED_SURFACE_ELEMENT", "ERROR", "surface Jacobian changes orientation inside the face"
                    elif topology.dimension == 3 and upper < -zero_limit:
                        code, severity, message = "INVERTED_VOLUME_ELEMENT", "ERROR", "volume element has a negative signed Jacobian"
                    elif scale == 0 or measure <= zero_limit or min(abs(lower), abs(upper)) <= zero_limit:
                        code, severity, message = "DEGENERATE_ELEMENT", "ERROR", "element has zero or numerically unresolved measure/Jacobian"
                    elif scaled <= relative_tolerance:
                        code, severity, message = "NEAR_DEGENERATE_ELEMENT", "WARNING", "element Jacobian is small relative to its size"
                    else:
                        continue
                    local_index = start + offset
                    findings.append(CheckFinding(
                        code, severity, message,
                        element_id=None if ids is None else int(ids[local_index]),
                        property_id=None if properties is None else int(properties[local_index]),
                        cell_block_index=block_index,
                        local_element_index=local_index,
                        measured_value=scaled,
                    ))
                checked_count += len(connectivity)
    except (ValueError, TypeError, RuntimeError, OverflowError) as error:
        _fail(report, f"geometry backend result could not be used: {error}")
        return

    version = str(getattr(backend_module, "__version__", "unknown"))
    report.geometry_quality = {
        "backend": _backend_name(backend_module),
        "backend_version": version,
        "relative_jacobian_tolerance": relative_tolerance,
        "checked_elements": checked_count,
        "unsupported_elements": dict(sorted(unsupported.items())),
        "minimum_scaled_jacobian": None if checked_count == 0 else minimum_scaled_jacobian,
    }
    report.findings.extend(findings)
    status = "FINDINGS" if findings else "UNSUPPORTED" if unsupported else "PASS" if checked_count else "NOT_RUN"
    reason = f"{_backend_name(backend_module)} {version}: checked {checked_count} elements"
    if unsupported:
        reason += f"; unsupported: {unsupported}"
    report.checks["area_volume_degeneracy"] = CheckResult(status, reason, len(findings))
