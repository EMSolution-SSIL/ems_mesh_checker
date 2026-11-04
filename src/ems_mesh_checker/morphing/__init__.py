"""Parameter-driven geometry-following mesh morphing (optional integration)."""

from .geometry_follow import (
    MorphingError,
    apply_rectangle_case,
    create_rectangle_manifest,
    mesh_fingerprint,
)
from .parametric import (
    apply_cylinder_case, apply_l_prism_case,
    create_cylinder_manifest, create_l_prism_manifest,
)
from .octant import (
    apply_octant_scale_case, apply_octant_shell_case,
    create_octant_scale_manifest, create_octant_shell_manifest,
)
from .editable_dxf import read_edited_dxf, write_editable_dxf
from .editable_geo import read_edited_geo, write_editable_geo

__all__ = [
    "MorphingError",
    "apply_rectangle_case",
    "create_rectangle_manifest",
    "mesh_fingerprint",
    "apply_cylinder_case",
    "apply_l_prism_case",
    "create_cylinder_manifest",
    "create_l_prism_manifest",
    "apply_octant_scale_case",
    "apply_octant_shell_case",
    "create_octant_scale_manifest",
    "create_octant_shell_manifest",
    "read_edited_dxf",
    "write_editable_dxf",
    "read_edited_geo",
    "write_editable_geo",
]
