---
name: ems-mesh-checker
description: Extract and validate representative Property-boundary geometry from EMSolution meshes, including Property selection, feature-edge cleanup, DXF/GEO/Femap export, and Gmsh checks.
---

# EMS Mesh Checker

Use this skill when a user asks how to inspect an EMSolution mesh, extract a
Property outline or interface, create representative CAD/mesher geometry, or
diagnose unwanted feature-edge surfaces. The desired result is representative
external and Property-boundary geometry, not a copy of every finite-element
surface.

## Prerequisites

The package requires Python 3.10 or newer and installs
`ems-file-format-converter>=0.6.0`, PyVista/VTK, meshio, ezdxf, and networkx.
For local development, the converter may instead be installed editable from
its sibling repository. Gmsh is optional but should be used to validate GEO
when available.

## Workflow

1. Identify the input format (`.atl`, `.neu`, `.unv`, `.msh`, or another
   meshio-supported format), dimensionality, and the output path. The CLI
   writes DXF at the requested path and writes GEO and Femap Neutral files with
   the same stem.
2. Decide which Properties should participate:
   - `--properties 10,11` extracts only the listed IDs. Repeat the option or
     use comma-separated IDs as needed.
   - `--exclude-properties 1` extracts everything except the listed IDs; this
     is usually easier when air is one of many Properties.
   - `--merge-properties 10,11` treats the listed adjacent Properties as one
     region and omits their mutual interface. Repeat for independent groups.
   Do not infer Property IDs from names; inspect the mesh or diagnostics first.
3. Extract the Property exterior and interface faces. In 3-D, use
   `--suppress-quad-tri-interfaces` only when a same-Property hexahedron/tet
   coupling face is mesh-conforming: one quadrilateral and two triangles share
   the same nodes. Cross-Property boundaries remain eligible for extraction.
4. Apply feature-edge cleanup only when the user accepts heuristic filtering:

   ```powershell
   ems-mesh-checker model.neu result.dxf `
     --remove-small-loops `
     --max-loop-edges 10
   ```

   This removes closed feature-edge loops and isolated open surface components
   whose boundary has at most the configured number of edges. It does not
   remove a small component merely because it is triangular: closed and
   multi-surface components are preserved, as are 2-D closed regions. Keep the
   diagnostic counts in the user-facing result.
5. Use circular reconstruction by default. Set
   `--circle-elements-per-turn N` when the downstream mesher needs a target
   circumferential discretization. Use `--no-arc-simplification` only when
   straight source segments are explicitly required. A fitted arc is a
   representative geometry, not proof that every source node was exactly
   circular.
6. Prefer a Gmsh parse check for generated GEO:

   ```powershell
   gmsh result_feature_edges.geo -0 -nopopup
   ```

   Add `--gmsh-executable path\to\gmsh.exe` to let the CLI perform this check;
   add `--gmsh-mesh-dimension 2` or `3` to generate a validation mesh. DXF is
   curve-oriented and does not contain GEO `Surface`/`Volume` entities.

## Interpreting results

- A Property may consist of multiple disconnected regions; each region can
  produce its own boundary. This is valid for electromagnetic analysis.
- Remaining `unsupported_regions`, `open_shells`, or cleanup diagnostics are
  evidence to inspect, not automatic failures. Do not flatten a non-planar,
  branched, or self-inconsistent component into a planar surface.
- If a boundary disappears unexpectedly, rerun without cleanup options, then
  compare `--properties`, `--exclude-properties`, `--merge-properties`, and
  `--suppress-quad-tri-interfaces` independently.
- Higher-order elements are outside the current supported scope.

## Safety and reproducibility

Keep cleanup options disabled unless requested or justified by visible mesh
noise. Record the exact command, Property filters, tolerances, and generated
file paths. Never add private customer/internal mesh files to a public commit;
use the reviewed IronCoil and QuadTri samples for reproducible examples.
