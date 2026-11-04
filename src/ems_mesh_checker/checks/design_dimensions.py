"""Optional, explicit design limits for axis-aligned mesh dimensions."""

from __future__ import annotations

from typing import Any

import numpy as np

from .report import CheckFinding, CheckReport, CheckResult, MeshCheckError


_FIELDS = ("minimum", "maximum", "size")
_AXES = "xyz"


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise MeshCheckError(f"{name} must be a finite number")
    number = float(value)
    if not np.isfinite(number):
        raise MeshCheckError(f"{name} must be a finite number")
    return number


def _vector(value: Any, dimension: int, name: str) -> np.ndarray:
    if not isinstance(value, (list, tuple)) or len(value) != dimension:
        raise MeshCheckError(f"{name} must have {dimension} numeric coordinates")
    return np.array([_number(item, name) for item in value], dtype=float)


def _limits(value: Any, dimension: int, name: str, absolute: float, relative: float) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(value, dict):
        if not value or set(value) - {"minimum", "maximum"}:
            raise MeshCheckError(f"{name} range must contain minimum and/or maximum")
        lower = _vector(value["minimum"], dimension, f"{name}.minimum") if "minimum" in value else np.full(dimension, -np.inf)
        upper = _vector(value["maximum"], dimension, f"{name}.maximum") if "maximum" in value else np.full(dimension, np.inf)
        if np.any(lower > upper):
            raise MeshCheckError(f"{name} minimum exceeds maximum")
    else:
        lower = _vector(value, dimension, name)
        upper = lower.copy()
    finite_lower = np.isfinite(lower)
    finite_upper = np.isfinite(upper)
    lower[finite_lower] -= absolute + relative * np.abs(lower[finite_lower])
    upper[finite_upper] += absolute + relative * np.abs(upper[finite_upper])
    return lower, upper


def add_design_dimension_checks(report: CheckReport, spec: dict[str, Any] | None) -> None:
    """Compare measured model/Property bounds only when design limits are supplied."""
    if spec is None:
        report.checks["design_dimensions"] = CheckResult("NOT_RUN", "no design dimension specification")
        return
    if not isinstance(spec, dict) or set(spec) - {"absolute_tolerance", "relative_tolerance", "model", "properties"}:
        raise MeshCheckError("dimension_spec must be an object with model/properties and optional tolerances")
    if "model" not in spec and "properties" not in spec:
        raise MeshCheckError("dimension_spec must contain model and/or properties")
    absolute = _number(spec.get("absolute_tolerance", 0), "absolute_tolerance")
    relative = _number(spec.get("relative_tolerance", 0), "relative_tolerance")
    if absolute < 0 or relative < 0:
        raise MeshCheckError("dimension tolerances must be non-negative")
    model_bounds = report.dimensions.get("model")
    incomplete_input = {
        "NONFINITE_COORDINATE", "INVALID_CONNECTIVITY", "INVALID_NODE_REFERENCE",
        "INVALID_CELL_METADATA",
    }
    if model_bounds is None or any(item.code in incomplete_input for item in report.findings):
        report.checks["design_dimensions"] = CheckResult("NOT_RUN", "model dimensions are incomplete")
        return
    if report.checks["supported_elements"].status == "UNSUPPORTED":
        report.checks["design_dimensions"] = CheckResult("NOT_RUN", "unsupported elements prevent complete dimensions")
        return
    dimension = len(model_bounds["size"])
    targets: list[tuple[str, int | None, dict[str, Any], dict[str, Any] | None]] = []
    if "model" in spec:
        targets.append(("model", None, spec["model"], model_bounds))
    property_specs = spec.get("properties", {})
    if not isinstance(property_specs, dict):
        raise MeshCheckError("dimension_spec.properties must be an object keyed by Property ID")
    actual_properties = {item["property_id"]: item["bounds"] for item in report.dimensions["properties"]}
    seen_properties: set[int] = set()
    for key, expected in property_specs.items():
        try:
            property_id = int(key)
        except (TypeError, ValueError) as error:
            raise MeshCheckError(f"invalid Property ID in dimension_spec: {key!r}") from error
        if isinstance(key, bool) or str(property_id) != str(key) or property_id in seen_properties:
            raise MeshCheckError(f"invalid or repeated Property ID in dimension_spec: {key!r}")
        seen_properties.add(property_id)
        targets.append((f"Property {property_id}", property_id, expected, actual_properties.get(property_id)))
    if not targets:
        raise MeshCheckError("dimension_spec must contain at least one model or Property limit")

    findings: list[CheckFinding] = []
    comparisons: list[dict[str, Any]] = []
    for scope, property_id, expected, bounds in targets:
        if not isinstance(expected, dict) or not expected or set(expected) - set(_FIELDS):
            raise MeshCheckError(f"{scope} design dimensions must contain minimum, maximum, and/or size")
        if bounds is None:
            findings.append(CheckFinding(
                "DESIGN_PROPERTY_MISSING", "ERROR", f"{scope} is absent from the mesh",
                property_id=property_id,
            ))
            continue
        for field in _FIELDS:
            if field not in expected:
                continue
            lower, upper = _limits(expected[field], dimension, f"{scope}.{field}", absolute, relative)
            actual = bounds[field]
            for axis, measured in enumerate(actual):
                within = lower[axis] <= measured <= upper[axis]
                comparisons.append({
                    "scope": scope, "property_id": property_id,
                    "field": field, "axis": _AXES[axis], "measured": measured,
                    "allowed_minimum": None if not np.isfinite(lower[axis]) else float(lower[axis]),
                    "allowed_maximum": None if not np.isfinite(upper[axis]) else float(upper[axis]),
                    "within_limits": bool(within),
                })
                if within:
                    continue
                node_ids = ()
                if field == "size":
                    node_ids = tuple(value for value in (
                        bounds["minimum_node_ids"][axis], bounds["maximum_node_ids"][axis],
                    ) if value is not None)
                else:
                    node_id = bounds[f"{field}_node_ids"][axis]
                    if node_id is not None:
                        node_ids = (node_id,)
                findings.append(CheckFinding(
                    "DIMENSION_OUT_OF_RANGE", "ERROR",
                    f"{scope} {field}.{_AXES[axis]}={measured:g} is outside configured design limits",
                    node_ids=node_ids, property_id=property_id,
                    measured_value=measured,
                ))
    report.dimensions["design_comparison"] = {
        "absolute_tolerance": absolute, "relative_tolerance": relative,
        "comparisons": comparisons,
    }
    report.findings.extend(findings)
    report.checks["design_dimensions"] = CheckResult(
        "FINDINGS" if findings else "PASS", finding_count=len(findings),
    )
