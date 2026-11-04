# Changelog

## Unreleased

## 0.2.0 - 2026-11-03

- Add Python mesh checks for input consistency, Property dimensions, connectivity,
  free faces, duplicate-node candidates, and surface-element orientation, with
  JSON reports. Element area/volume quality remains an optional native backend.
- Add experimental manifest-backed geometry-following mesh morphing for
  supported rectangle, L-prism, cylinder, and octant cases. It requires the
  separately licensed morpher and a compatible geometry-quality backend.
- Add opt-in CadQuery STEP solid export for representative Property geometry,
  including analytical cylindrical faces, explicit coordinate units, and
  read-back validation.
- Add `--step-combined` to write all selected Properties as named independent
  solids in one STEP assembly alongside the per-Property files.
- Make the representative geometry dependencies part of the standard install
  and raise the minimum Python version to 3.11.
- Document individual and combined STEP output in the English and Japanese
  READMEs and the repository skill.

## 0.1.1 - 2026-09-10

- Add opt-in removal of small closed feature-edge loops and isolated open
  surface components with a configurable maximum edge count.

## 0.1.0 - 2026-09-10

- Initial proof-of-concept release.
- Extract external and property-interface faces from 2-D and 3-D meshes.
- Extract representative feature edges with PyVista/VTK.
- Reconstruct line and circular-arc geometry.
- Export DXF, Gmsh GEO, and Femap Neutral point/curve geometry.
- Select, exclude, or merge Property IDs.
- Optionally suppress same-Property hexahedron/tetrahedron coupling interfaces.
