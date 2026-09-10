"""Representative-geometry helpers backed by PyVista/VTK."""

from .feature_edges import (
    FeatureEdgeConfig,
    FeatureEdgeCurve,
    FeatureEdgeDependencyError,
    FeatureEdgeResult,
    extract_exterior_feature_edges,
    extract_all_property_feature_edges,
    extract_feature_edges,
    extract_interface_feature_edges,
    extract_property_group_feature_edges,
    extract_property_feature_edges,
    meshio_surface_to_polydata,
)
from .curve_simplification import (
    CircularArcPrimitive,
    CurvePrimitive,
    CurveSimplificationConfig,
    LinePrimitive,
    SimplifiedCurveLoop,
    simplify_planar_loop,
)
from .planar_regions import (
    PlanarLoop,
    PlanarRegion,
    PlanarRegionConfig,
    PlanarRegionResult,
    UnsupportedRegion,
    extract_planar_regions,
)
from .ruled_surfaces import (
    RuledSurfaceConfig,
    RuledSurfacePatch,
    RuledSurfaceResult,
    reconstruct_ruled_surfaces,
)

__all__ = [
    "CircularArcPrimitive",
    "CurvePrimitive",
    "CurveSimplificationConfig",
    "FeatureEdgeConfig",
    "FeatureEdgeCurve",
    "FeatureEdgeDependencyError",
    "FeatureEdgeResult",
    "LinePrimitive",
    "SimplifiedCurveLoop",
    "extract_exterior_feature_edges",
    "extract_all_property_feature_edges",
    "extract_feature_edges",
    "extract_interface_feature_edges",
    "extract_property_group_feature_edges",
    "extract_property_feature_edges",
    "meshio_surface_to_polydata",
    "simplify_planar_loop",
    "PlanarLoop",
    "PlanarRegion",
    "PlanarRegionConfig",
    "PlanarRegionResult",
    "RuledSurfaceConfig",
    "RuledSurfacePatch",
    "RuledSurfaceResult",
    "UnsupportedRegion",
    "extract_planar_regions",
    "reconstruct_ruled_surfaces",
]
