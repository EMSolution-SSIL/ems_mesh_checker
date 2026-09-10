# Changelog

## Unreleased

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
