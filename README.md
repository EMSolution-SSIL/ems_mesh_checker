# EMS Mesh Checker

`ems_mesh_checker` is a Python proof-of-concept for checking finite-element mesh
topology and reconstructing representative Property geometry from EMSolution
meshes. It uses PyVista/VTK for feature-edge extraction, then applies its own
topology and curve reconstruction to export reusable CAD/mesher input.

## Features

- Extract external faces and interfaces between Property IDs.
- Select only specified Properties with `--properties`.
- Exclude specified Properties, such as an air region, with
  `--exclude-properties`.
- Treat multiple Properties as one region and omit their mutual interfaces with
  `--merge-properties`.
- Optionally suppress same-Property hexahedron/tetrahedron coupling interfaces
  made from one quadrilateral and two triangles.
- Reconstruct representative straight lines and circular arcs from feature
  edges.
- Export the same reconstructed curves to DXF, Gmsh GEO, and Femap Neutral
  geometry files.
- Report node/element loading progress for large Femap Neutral files.

The current PoC targets linear elements. Higher-order element support is not a
current release goal.

## Installation

Python 3.10 or newer is required. The normal installation model is to install
both projects as Python packages; the two repositories do not need to remain in
the same parent directory.

Install from PyPI. `ems-file-format-converter` 0.6.0 or newer is installed
automatically as a dependency:

```powershell
python -m pip install ems-mesh-checker
```

For development of both projects, clone and install both repositories:

```powershell
git clone https://github.com/EMSolution-SSIL/ems_file_format_converter.git
git clone https://github.com/EMSolution-SSIL/ems_mesh_checker.git

python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .\ems_file_format_converter
.\.venv\Scripts\python.exe -m pip install -e ".\ems_mesh_checker[dev]"
```

The editable installs may point to repositories in any directories. Placing
them next to each other is only a convenient convention.

## Command-line usage

The command accepts an arbitrary input mesh. It always writes three files with
the same stem: representative `.dxf`, `.geo`, and Femap geometry `.neu`.

```powershell
ems-mesh-checker INPUT_MESH [OUTPUT.dxf]
```

IronCoil public sample:

```powershell
ems-mesh-checker .\data\3D\IronCoil\post_geom.atl .\generated\IronCoil.dxf `
  --circle-elements-per-turn 48
```

QuadTri public sample, retaining cross-Property boundaries while suppressing
same-Property hex/tet coupling faces:

```powershell
ems-mesh-checker .\data\3D\QuadTri\post_geom.neu .\generated\QuadTri.dxf `
  --suppress-quad-tri-interfaces
```

Export only Properties 2 and 5:

```powershell
ems-mesh-checker model.neu geometry.dxf --properties 2,5
```

Exclude air Property 1 and merge Properties 4 and 5 into one extracted region:

```powershell
ems-mesh-checker model.neu geometry.dxf `
  --exclude-properties 1 `
  --merge-properties 4,5
```

For repeated independent groups, repeat `--merge-properties`. Run
`ems-mesh-checker --help` for tolerance, preview, Gmsh validation, and other
options.

PyVista can occasionally return small triangular or polygonal feature-edge
noise. The opt-in cleanup below removes closed edge loops and isolated open
surface components whose boundary contains at most 10 edges; adjust
`--max-loop-edges` for the model when needed.

```powershell
ems-mesh-checker model.neu geometry.dxf `
  --remove-small-loops `
  --max-loop-edges 10
```

This cleanup is disabled by default because a small loop or open surface can
also be a real geometric feature. Closed and multi-surface components are kept,
as are 2-D closed regions. The CLI reports both removed edge loops and removed
open surface components in its diagnostics.

## Output formats

- DXF contains point/line/arc-oriented representative geometry for CAD and
  mesher interoperability.
- Gmsh GEO contains points, lines, circles, planar/ruled surfaces, surface loops,
  and volumes where reconstruction succeeds.
- Femap Neutral output contains reconstructed point, line, and arc geometry.

Feature-edge extraction is heuristic. Complex or noisy meshes may contain open
shells or unsupported planar regions; review the CLI diagnostics and generated
geometry before using it for production meshing.

## Python API

The lower-level API is available from `ems_mesh_checker`:

```python
from ems_file_format_converter import read_mesh
from ems_mesh_checker import BoundaryExtractor, FeatureEdgeConfig
from ems_mesh_checker import extract_property_group_feature_edges

mesh = read_mesh("model.neu", progress=True)
boundaries = BoundaryExtractor(mesh).extract()
features = extract_property_group_feature_edges(
    boundaries,
    (2, 5),
    config=FeatureEdgeConfig(
        feature_angle_degrees=30.0,
        remove_small_loops=True,
        max_loop_edges=10,
    ),
)
```

## Public sample data

Only the following reviewed sample models are included:

- `data/3D/IronCoil/post_geom.atl`
- `data/3D/QuadTri/post_geom.neu`

Other development and customer models are intentionally excluded. The
`.gitignore` treats all additional files below `data/` as private by default.

## Tests

```powershell
python -m pytest -q
```

## License

MIT License. Copyright (c) 2026 Hiroyuki Kaimori. See `LICENSE`.

## 日本語版

日本語版は [`README_ja.md`](README_ja.md) を参照してください。
