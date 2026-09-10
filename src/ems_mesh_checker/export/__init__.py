"""Optional interchange-format exports for extracted boundaries."""

from .dxf import (
    DXFExportError,
    write_boundary_result_dxf,
    write_dxf,
    write_feature_edge_results_dxf,
    write_feature_edges_dxf,
    write_planar_regions_dxf,
)
from .gmsh_geo import dxf_to_geo
from .femap_neutral import (
    FemapNeutralGeometryExportError,
    write_planar_regions_femap_neutral,
)
from .representative_geo import RepresentativeGeometryExportError, write_planar_regions_geo
from .volume_topology import (
    ClosedSurfaceShell,
    OpenSurfaceComponent,
    SurfaceTopologyRecord,
    VolumeTopologyConfig,
    VolumeTopologyResult,
    build_volume_topology,
)

__all__ = [
    "DXFExportError",
    "FemapNeutralGeometryExportError",
    "ClosedSurfaceShell",
    "OpenSurfaceComponent",
    "RepresentativeGeometryExportError",
    "SurfaceTopologyRecord",
    "VolumeTopologyConfig",
    "VolumeTopologyResult",
    "build_volume_topology",
    "dxf_to_geo",
    "write_boundary_result_dxf",
    "write_dxf",
    "write_feature_edge_results_dxf",
    "write_feature_edges_dxf",
    "write_planar_regions_dxf",
    "write_planar_regions_femap_neutral",
    "write_planar_regions_geo",
]
