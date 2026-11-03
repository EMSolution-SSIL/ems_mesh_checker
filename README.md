# EMS Mesh Checker

Topology checks and property-boundary extraction for `meshio.Mesh` FEM models.

Version 0.2.0 requires Python 3.11 or newer. Install from this checkout with
`python -m pip install -e .`; the standard install includes CadQuery, PyVista,
ezdxf, and the file-format converter used by the representative-geometry CLI.
Python 3.10 users can use the earlier 0.1.x release.

## Installation and tests

From this repository, install the package and development tools, then run tests:

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
```

The standard install obtains `ems-file-format-converter` as a dependency. To
develop both sibling projects together, install the converter checkout with
`python -m pip install -e ..\ems_file_format_converter` first.

`ems_mesh_checker.io_contract.inspect_mesh_contract()` validates point IDs,
cell-block metadata alignment, and connectivity bounds in meshes supplied by
the converter.

## Mesh checks (Python-only stages)

The separate checker CLI reports input/ID problems, Property dimensions and
face-connected components, unmodified free faces, coincident-but-unshared face
candidates, and explicit surface-element orientation. It does not modify the
mesh. Without a separate native geometry backend, its JSON report marks
area/volume degeneration as `NOT_RUN`; a successful topology-only run does
**not** imply that the volume elements are valid.

With `ems-file-format-converter` importable, run for example:

```powershell
python -m ems_mesh_checker.check_cli data\3D\IronCoil\post_geom.atl `
  --report generated\checks\IronCoil.json `
  --free-faces-report generated\checks\IronCoil_free_faces.json
```

Area, volume and signed-Jacobian checks can run through a separately supplied
`ems_mesh_checker_native` extension. With no extension installed, the default
`--geometry-backend auto` reports `NOT_RUN`; `--geometry-backend required`
reports a failure if it is unavailable, and `off` disables it. The optional
backend reports its name, version, checked-element count, and unsupported
types in JSON. Use `--geometry-relative-tolerance` to adjust the dimensionless
near-degeneracy warning threshold. The public wheel/sdist does not contain
the private extension or solver source.

Design dimensions are checked only when `--dimension-spec design.json` is
provided. For example, this 3-D specification accepts model size within
`0.01` of the target and a range for Property 7's X size:

```json
{
  "absolute_tolerance": 0.01,
  "model": {"size": [10, 20, 30]},
  "properties": {"7": {"size": {"minimum": [2, 3, 4], "maximum": [4, 5, 6]}}}
}
```

`minimum`, `maximum`, and `size` can be set for the model or each Property.
Each accepts an exact coordinate list or a `{ "minimum": [...], "maximum": [...] }`
range. An optional `relative_tolerance` is also supported. The JSON report
records each measured coordinate, its allowed limits, and the node IDs for
violations. Without a specification, `design_dimensions` is `NOT_RUN`.

Use `--coordinate-tolerance <distance>` only when near-coincident nodes should
be considered; the default requires exact coordinates. On an explicit surface
element at a Property interface, specify which volume side defines "outward"
with `--surface-owner FACE_PROPERTY:VOLUME_PROPERTY`. `--fail-on error` or
`--fail-on warning` enables a nonzero exit status for CI; otherwise findings
are reported without failing the command. Free faces are **candidates**, not
automatic evidence of unintended holes, because legitimate outer surfaces and
cavities also remain after face cancellation.
For 3-D meshes, a closed free-face shell strictly inside another shell's
bounding box is reported as an informational review candidate, not as a
confirmed hole. Its patch index and bounds are in the report; each raw free
face carries its patch index for inspection.

The expected surface-normal direction is `outward` by default. For inward-facing
surface elements, pass `--surface-normal-direction inward`; mixed models can
override individual face Properties with `--surface-normal FACE_PROPERTY:inward`
or `FACE_PROPERTY:outward`. An interface face still needs `--surface-owner` to
identify the reference volume. Coincident surface elements in *different*
Properties are reported as informational overlays and do not stop normal or
free-face checks; duplicates within the same Property remain errors.

## DXF export

The standard install includes ezdxf.
Boundary edges are exported as DXF `LINE` entities and triangular or quadrilateral
boundary faces are exported as `3DFACE` entities.

```python
from ems_mesh_checker import BoundaryExtractor
from ems_mesh_checker.export import write_boundary_result_dxf

result = BoundaryExtractor(mesh).extract()
write_boundary_result_dxf(result, "boundaries.dxf")
```

External boundaries are written to the `EXTERNAL` layer; each Property interface is
written to `INTERFACE_P<low>_P<high>`. `write_dxf(result.get_interface(1, 2), path)`
exports an individual interface surface.

## Representative feature curves

The standard install includes PyVista/VTK-backed feature-edge extraction. The
existing topology layer first chooses the exterior or a Property-interface patch;
PyVista then classifies its geometric edges. The returned line segments are joined
into branch-aware polylines and closed loops, while the original selected surface
is retained for verification.

```python
from ems_mesh_checker import (
    BoundaryExtractor,
    FeatureEdgeConfig,
    extract_exterior_feature_edges,
)
from ems_mesh_checker.export import write_feature_edges_dxf

boundaries = BoundaryExtractor(mesh).extract()
feature_edges = extract_exterior_feature_edges(
    boundaries,
    config=FeatureEdgeConfig(feature_angle_degrees=30.0),
)
write_feature_edges_dxf(feature_edges, "representative_feature_edges.dxf")
```

This output contains `POLYLINE` entities, not one `3DFACE` per finite element.
`extract_interface_feature_edges(boundaries, property_a, property_b)` returns one
result for each disconnected interface patch. `min_curve_length` is disabled by
default so small physical features are never silently removed.

For AGC fixtures, use the workspace-local converter without installing it:

```powershell
.\.venv\Scripts\python.exe scripts\export_agc_feature_edges.py
```

The script validates that its source coordinates have sufficient geometric rank and
refuses to write misleading geometry if they are degenerate. It combines exterior
and interface faces per Property, merges source faces only when their normals are
coplanar, and writes the accepted representative curves both as DXF `LINE` / `ARC`
entities and as validated Gmsh geometry. Circular runs fitted independently from
rounded mesh nodes are clustered onto one canonical center/radius before export.
Overlapping angular coverage is emitted once in curve-only DXF/Femap output, while
GEO reuses the same canonical support and retains only subdivisions required by its
surface topology. Curves belonging to multiple Properties use the
`SHARED_PROPERTIES` layer. Unless
`--mesh-size` is specified, the GEO point size is set to one hundredth of the input
mesh bounding-box diagonal. This keeps the same command practical for both
millimetre-scale Femap coordinates and metre-scale EMSolution coordinates.

```python
from ems_mesh_checker import (
    CurveSimplificationConfig,
    PlanarRegionConfig,
    extract_planar_regions,
    extract_property_feature_edges,
    write_planar_regions_dxf,
    write_planar_regions_geo,
)

features = extract_property_feature_edges(boundaries, property_id=10)
regions = extract_planar_regions(
    features,
    config=PlanarRegionConfig(coplanar_angle_degrees=1.0),
)
write_planar_regions_geo(
    regions,
    "property_10.geo",
    mesh_size=0.01,
    curve_config=CurveSimplificationConfig(radial_relative_tolerance=1.0e-4),
    circle_elements_per_turn=48,
)
write_planar_regions_dxf(
    regions,
    "property_10.dxf",
    curve_config=CurveSimplificationConfig(radial_relative_tolerance=1.0e-4),
)
```

PyVista edges are not trusted blindly. By default, a PyVista edge between coplanar
source faces is ignored, while the source-normal check still separates a real corner
that PyVista missed. Non-planar, open, branched, or self-inconsistent components are
reported in `unsupported_regions` and are never flattened into a `Plane Surface`.

Use `--gmsh-executable` to parse every generated GEO. Add
`--gmsh-mesh-dimension 3` to generate a volume mesh, and use `--only-large-3d`
to run only the production-size AGC 3-D fixture:

```powershell
.\.venv\Scripts\python.exe scripts\export_agc_feature_edges.py `
  --only-large-3d `
  --circle-elements-per-turn 48 `
  --gmsh-executable C:\EMSolution\gmsh-4.13.1-Windows64\gmsh.exe `
  --gmsh-mesh-dimension 3
```

### STEP solids

The standard `ems-mesh-checker` command can additionally export one STEP file
per selected Property or merged Property group. The input coordinates' length
unit must be supplied explicitly. The existing DXF, GEO, and Femap curve files
are written alongside the STEP files.

```powershell
ems-mesh-checker data\3D\IronCoil\post_geom.atl generated\ironcoil.dxf `
  --properties 3,4 --step --step-unit MM
```

The example writes `generated\ironcoil_P3.step` and
`generated\ironcoil_P4.step`. `--exclude-properties` and
`--merge-properties` apply to STEP as they do to the other formats. A STEP
file is written only for a closed, valid CAD solid. Unreconstructed surfaces
and open shells raise an error. Plane and recognized circular extrusion
surfaces are stored as CAD surfaces, rather than exporting each mesh facet.
The STEP writer reopens each file and checks the resulting solids before
placing it at the requested path. A separate CAD application should still be
used to inspect critical models before downstream use.

Add `--step-combined` to also write `<output-stem>_ALL.step`. It contains each
selected Property as a named, independent Solid in one STEP assembly; the
individual Property STEP files are still written. Solids are not Boolean
fused, so Property identities and their interface geometry remain available
for inspection.

```powershell
ems-mesh-checker data\3D\IronCoil\post_geom.atl generated\ironcoil_all.dxf `
  --step --step-unit MM --step-combined
```

This writes `generated\ironcoil_all_ALL.step` with Properties 1, 3, and 4,
plus `generated\ironcoil_all_P1.step`, `_P3.step`, and `_P4.step`. For the
QuadTri sample, add `--suppress-quad-tri-interfaces` when the same-Property
hexahedron/tetrahedron coupling should be hidden. The example `MM` unit is an
explicit interpretation of the input coordinates; select the actual model
unit for production data. Gmsh can open the resulting STEP as CAD geometry.

The IronCoil ATLAS model used by the
[pyemsi feature-edge example](https://emsolution-ssil.github.io/pyemsi/docs/api/Plotter/set_feature_edges/)
can be run as an independent real-model regression. The PyVista edge categories
use the same defaults as `DataSetFilters.extract_feature_edges()`; MeshChecker
additionally assembles each Property's external and interface faces before extraction.

```powershell
.\.venv\Scripts\python.exe scripts\export_representative_geometry.py `
  --only-ironcoil `
  --preview `
  --gmsh-executable C:\EMSolution\gmsh-4.13.1-Windows64\gmsh.exe
```

The larger `data/J` Femap fixtures can be selected independently. Start with the
smaller Shield model; the platform input is approximately 1.36 GB and can require
substantial processing time and memory.

```powershell
.\.venv\Scripts\python.exe scripts\export_representative_geometry.py `
  --only-j-shield `
  --circle-elements-per-turn 48

.\.venv\Scripts\python.exe scripts\export_representative_geometry.py `
  --only-j-platform `
  --circle-elements-per-turn 48
```

Femap Neutral input is read as a stream. For large files, node and element counts
plus elapsed time are reported every 500,000 accepted records. Change the interval
with `--progress-interval N`, or use `--progress-interval 0` to disable reporting.

Use `--properties` to export only selected Property IDs, or
`--exclude-properties` to export every Property except the listed IDs. These two
selection modes are mutually exclusive. Use
`--merge-properties` to treat adjacent Properties as one region: the interface
between members of the group is removed before PyVista/VTK feature-edge extraction,
while interfaces against Properties outside the group are retained. Both GEO, DXF,
and Femap Neutral geometry use the same filtered/merged curve set.

```powershell
# Export only P101, P102, and P205. P101/P102 become one geometric region.
.\.venv\Scripts\python.exe scripts\export_representative_geometry.py `
  --only-j-platform `
  --properties 101,102,205 `
  --merge-properties 101,102 `
  --circle-elements-per-turn 48
```

```powershell
# Export every Property except P900 (for example, an air region).
.\.venv\Scripts\python.exe scripts\export_representative_geometry.py `
  --only-j-platform `
  --exclude-properties 900 `
  --circle-elements-per-turn 48
```

`--merge-properties` can be repeated for independent groups. Groups may not
overlap, and every merged Property must remain selected after applying
`--properties` or `--exclude-properties`. The merged group is labelled, for example,
`PROPERTY_GROUP_P101_P102` in representative outputs.

Hexahedron/tetrahedron coupling meshes can contain one hexahedron quad and two
tetrahedron triangles on the same geometric connection. Because their canonical
node sets differ, the normal face table sees all three as external faces. Enable
`--suppress-quad-tri-interfaces` to omit only unambiguous matches where the two
triangles use exactly the quad's four corner nodes, share a quad diagonal, and all
three faces belong to the same Property. Matches spanning different Properties
are retained as Property boundaries. The default is disabled, so ordinary boundary
diagnostics remain unchanged.

```powershell
.\.venv\Scripts\python.exe scripts\export_representative_geometry.py `
  --only-quad-tri `
  --suppress-quad-tri-interfaces `
  --circle-elements-per-turn 48
```

The bundled `data/3D/QuadTri/post_geom.neu` fixture contains 288 geometric coupling
interfaces: 48 within P1 are suppressible, while 240 between P1 and P10 are
retained. The CLI reports both the number of suppressed interfaces and the number
of omitted source faces. Audit records remain available through
`BoundaryResult.suppressed_quad_tri_interfaces`.

PyVista/VTK can occasionally return small triangular or polygonal loops as
feature-edge noise. Enable the following opt-in graph filter to remove closed
loops containing no more than the selected number of edges:

```powershell
.\.venv\Scripts\python.exe scripts\export_representative_geometry.py `
  --only-ironcoil `
  --remove-small-loops `
  --max-loop-edges 10 `
  --circle-elements-per-turn 48
```

The filter follows pyemsi's small-loop cleanup and also removes isolated open
surface components whose boundary is at or below the same threshold. Closed
and multi-surface components, plus 2-D closed regions, are kept. Diagnostics
report detected edge loops, removed segments, and removed open components. The
filter is disabled by default because small geometry can be intentional; use
the lowest suitable threshold.

`--preview` writes a white-background PNG with translucent Property surfaces and
red feature edges, matching the visual convention used in the pyemsi example.
Representative GEO loops recognize circular runs conservatively and emit Gmsh
`Circle` entities; unmatched geometry remains `Line`. Use
`--arc-relative-tolerance` and `--max-arc-sample-angle` for model-specific tuning,
or `--no-arc-simplification` to reproduce the exact polyline-only output.
When matching translated cap arcs and a complete chain of cylindrical side
quadrilaterals are found, the GEO writer replaces those planar strips with
four-curve Gmsh `Surface` patches. Axial Property seams and unequal split points
on the two cap arcs are propagated to both sides, so a cylinder can remain curved
without violating the built-in kernel's four-boundary limit. Use
`--no-ruled-surfaces` for planar-patch-only output.
If Property seams prevent a complete chain, an individual quadrilateral strip is
also accepted when its two connectors are parallel, all four endpoints match a
known circular family within tolerance, and the recovered arc is no larger than
45 degrees. This removes residual circumferential chords without treating an
arbitrary coarse polygon as a cylinder.

Use `--circle-elements-per-turn N` (or the Python API argument with the same
name) to set a target number of circumferential elements. Each Gmsh `Circle` arc
receives a `Transfinite Curve` node count scaled by its central angle; for example,
`N=48` gives 12 elements on a 90-degree arc. This changes the new mesh
discretization without restoring the original finite-element chords.
The GEO writer also shares coincident points, curves, and Property-interface
surfaces. Coplanar faces are split at neighbor-Property seams, and long curves are
subdivided at shared intermediate points. A connected Property component becomes
a `Surface Loop` and `Volume` only when every curve belongs to exactly two surfaces
and the surface orientations are consistent. Disconnected regions of one Property
become separate Volumes; rejected components are emitted as `OPEN_SHELL` comments.
Use `--no-volumes` to disable this assembly.

Arc templates are collected across all Properties and reused when a cylindrical
side has no local cap face. Shared cap and seam points are inserted into a common
angular/axial grid before surfaces are written. A Property that still does not form
a degree-two shell is reported as `OPEN_SHELL` rather than being forced into an
invalid Volume.
The AGC/IronCoil script writes a curve-only representative DXF containing no
`3DFACE`, surface or volume entities. Accepted circles use true DXF `ARC` entities,
including arcs in arbitrary 3-D planes through the DXF extrusion/OCS definition.
The lower-level `write_feature_edges_dxf()` API remains available when the original
PyVista-derived `POLYLINE` samples are needed for inspection.

The same representative-geometry script also writes a Femap Neutral `.neu` file.
This is an editable wireframe-geometry export, not a finite-element mesh: data
blocks 570 and 571 contain shared Femap Points, Lines and true circular Arcs, and
block 413 preserves Property/shared-curve grouping as Femap layers. Surface block
572 and solid block 573 are intentionally omitted. A native Femap solid in block
573 would require an embedded Parasolid X_T or ACIS SAT body, which is outside the
curve interchange handled by this exporter.
Oriented 180-degree arcs include an explicit through-point, so complementary
semicircles remain distinct instead of collapsing to one curve record.

```python
from ems_mesh_checker import write_planar_regions_femap_neutral

write_planar_regions_femap_neutral(planar_results, "representative_geometry.neu")
```

## Gmsh validation

Gmsh does not directly import DXF geometry; attempting to open a `.dxf` as a Gmsh
input file produces a parser error. Use the bundled AGC script to generate DXF and
then convert its supported `LINE`/`3DFACE` entities into a Gmsh-readable `.geo` file:

```powershell
python scripts/export_agc_dxf.py `
  --include-large-3d `
  --gmsh-executable C:\EMSolution\gmsh-4.13.1-Windows64\gmsh.exe
```

The script verifies the generated `.geo` with Gmsh's `-parse_and_exit` mode. The GEO
is a tessellated boundary representation suitable for import validation; create a
compact parametric GEO model separately when changing the underlying geometry.

## Geometry-following mesh morphing (experimental)

This is a separate, manifest-backed workflow. It does **not** infer original node
correspondence from the representative GEO/DXF exports above. Supported shapes are
an axis-aligned 2-D rectangle, a lower-left L cross-section (2-D or straight 3-D
extrusion), a complete coaxial 3-D annulus, and the restricted positive-octant
3-D workflows below. The selected Property must have one connected component
and be surrounded by explicitly deformable Properties. The external boundary is
fixed except for tangential motion on explicitly recognized symmetry planes.
Element connectivity and original node,
element, and Property IDs are retained. Each case starts from the same base mesh.

`ems_mesh_morpher` is an **optional, separately licensed** dependency for
displacement propagation; no morpher source is bundled here. The
`ems_mesh_checker_native` geometry-quality backend is also required to approve a
case. Without either component, this workflow fails closed and writes no accepted
mesh. Install the optional Python dependencies with
`python -m pip install -e ".[morph]"` and arrange the geometry backend
separately. The `[morph]` extra installs the separately licensed morpher.

Example for a 2-D rectangular shield in Property 2, surrounded by air Property 1:

```powershell
python -m ems_mesh_checker.morph_cli manifest base.neu shield_manifest.json `
  --shape rectangle --property 2 --deformable 1 --axis x
python -m ems_mesh_checker.morph_cli apply base.neu shield_manifest.json `
  cases.json output_cases
```

`cases.json` can contain independent thickness cases:

```json
[
  {"name": "thin", "thickness": 1.8, "anchor": "center"},
  {"name": "base", "thickness": 2.0, "anchor": "center"},
  {"name": "thick", "thickness": 2.2, "anchor": "center"}
]
```

Use actual model units in place of the example dimensions. The manifest records a
mesh fingerprint, original boundary node/element correspondence, Property IDs,
shape parameters, and a consistency digest. A changed base mesh or edited manifest
is rejected. For an L shape use `--shape l_prism --notch X Y`; case fields are
`length_x`, `length_y`, `arm_width_x`, and `arm_width_y`. For a cylinder use
`--shape cylinder --axis-point X Y Z --axis-direction X Y Z`; case fields are
`inner_radius` and `outer_radius`. The unchanged base mesh must be supplied for
every run. A rejected case gets a JSON failure report but no mesh file.
Each survey also writes `summary.json` and `summary.csv` with case outcomes.

For a 1/8-symmetry IronCoil model, `octant_scale` uniformly scales one Property
about the origin while adjacent Properties absorb the displacement. The target
must touch the three zero-coordinate symmetry planes; all of its exterior faces
must lie on them. This does **not** edit a general CAD feature or independently
control the coil's inner and outer surfaces. For example, from this repository:

```powershell
python -m ems_mesh_checker.morph_cli manifest data/3D/IronCoil/post_geom.atl `
  iron_manifest.json --shape octant_scale --property 1 --deformable 4
python -m ems_mesh_checker.morph_cli apply data/3D/IronCoil/post_geom.atl `
  iron_manifest.json iron_cases.json iron_results
```

`iron_cases.json`: `[{"name":"smaller","scale":0.90},{"name":"larger","scale":1.10}]`.
Use Property 3 instead of 1 to scale the Coil, with Property 4 still deformable.
Alternatively, a case can specify `{"name":"x_larger","displacement":[0.005,0,0]}`.
This increases the Iron Property's maximum X coordinate by 0.005 model units;
coordinates between X=0 and that outer extent move proportionally. It is an
axis-extent edit, **not** a rigid translation of every node. X=0, Y=0, and Z=0
remain symmetry planes. Specify either `scale` or `displacement`, not both.
Neighboring nodes on a symmetry face may move within that face; the face-normal
coordinate and the non-symmetry exterior remain fixed. The geometry-quality
gate still rejects a scale that inverts or degenerates any element.

`octant_shell` recognizes a complete positive-octant box shell between two
neighboring Properties. Its `inner_radius` and `outer_radius` are the maximum
coordinate of the corresponding square surfaces, not Euclidean radii. One face
can move while the other face and the far exterior stay fixed. The internal
displacement is an analytic max-coordinate radial map, so it does not incur a
large global Laplace solve. For the local BoxShield test model (not distributed):

```powershell
python -m ems_mesh_checker.morph_cli manifest data/3D/BoxShield/pre_geom_hexa.neu `
  box_manifest.json --shape octant_shell --property 1 `
  --inner-property 2 --outer-property 3 --deformable 2 --deformable 3
python -m ems_mesh_checker.morph_cli apply data/3D/BoxShield/pre_geom_hexa.neu `
  box_manifest.json box_cases.json box_results
```

`box_cases.json` may contain `{"name":"inner_smaller","inner_radius":0.0999}`
and `{"name":"outer_larger","outer_radius":0.1011}` (model-coordinate units).
Those examples move one face by 10% of the original 0.001 shell thickness.
The equivalent displacement cases are `{"name":"inner_smaller","inner_displacement":-0.0001}`
and `{"name":"outer_larger","outer_displacement":0.0001}`. Each displacement
is added to the corresponding original interface coordinate. Do not combine an
absolute radius and a displacement for the same face in one case.
The same commands apply to `pre_geom_tetra.neu` with a separately generated
manifest. Octant workflows currently accept JSON parameters only, not editable
GEO/DXF re-import. Shape recognition, a native geometry-quality backend, and
post-write ID/connectivity verification still gate every accepted mesh.

A narrow DXF/GEO edit path is available for these shapes:

```powershell
python -m ems_mesh_checker.morph_cli export-dxf shield_manifest.json editable.dxf
python -m ems_mesh_checker.morph_cli apply-dxf base.neu shield_manifest.json `
  edited.dxf output_from_dxf
python -m ems_mesh_checker.morph_cli export-geo shield_manifest.json editable.geo
python -m ems_mesh_checker.morph_cli apply-geo base.neu shield_manifest.json `
  edited.geo output_from_geo --gmsh-executable C:\path\to\gmsh.exe
```

The DXF must retain the named `EMS_MORPH_*` LINEs or CIRCLEs and the original
outline topology. The GEO must retain its numbered Points and Lines or quarter
Circle arcs; Gmsh evaluates expressions before their coordinates and endpoint
IDs are checked. Cylinder DXF/GEO currently requires a +Z axis. The L-prism
sketch is its XY cross-section, with extrusion data retained in the manifest.
Extra, missing, angled, or disconnected primitives are rejected. Arbitrary
edits to the older representative GEO/DXF files, changed contact topology,
large deformation, and automatic remeshing remain outside this workflow. Small
synthetic tests have passed; real production meshes and large-model performance
still require validation.

## License

MIT License. Copyright (c) 2026 Hiroyuki Kaimori. See `LICENSE`.

## 日本語版

日本語版は [`README_ja.md`](README_ja.md) を参照してください。
