# EMS Mesh Checker agent guidance

## Scope

This repository provides topology checks and representative Property geometry
extraction for EMSolution finite-element meshes. Keep changes focused on the
package under `src/ems_mesh_checker` and its tests.

## Source and dependency boundaries

- `ems-file-format-converter` is a separate project and is required at version
  0.6.0 or newer. Do not copy its parsers into this repository.
- PyVista/VTK performs feature-edge extraction; the package then applies its
  own Property-aware topology checks and geometry reconstruction.
- The current public scope is linear elements. Do not claim higher-order
  element support unless it is implemented and tested.

## Mesh and geometry invariants

- Preserve Property boundaries by default. Options that remove feature-edge
  loops, small open components, or same-Property quad/triangle interfaces must
  remain opt-in.
- `--suppress-quad-tri-interfaces` may suppress only an unambiguous
  same-Property interface made from one quadrilateral and two triangles; it
  must not remove a cross-Property boundary.
- `--remove-small-loops` is heuristic. It may remove isolated small open
  surface components as well as small closed feature loops, while preserving
  closed or multi-surface components and 2-D closed regions.
- Keep diagnostics auditable when a heuristic removes geometry.

## Data policy

Only the reviewed public samples in `data/3D/IronCoil` and `data/3D/QuadTri`
may be committed. AGC, J, and other customer or internal model data must stay
local. Generated DXF, GEO, Femap Neutral, preview, and mesh files should not
be committed.

## Validation

Run the relevant tests after changes; for a normal package change run:

```powershell
python -m pytest -q
python -m build
python -m twine check dist/*
```

For representative GEO changes, parse a generated file with Gmsh when it is
available. Add regression tests for every new filtering or topology rule, and
test both the default (preserve geometry) and opt-in behavior.

Do not publish to PyPI, push to GitHub, or add model data unless the user asks
for that external action explicitly.
