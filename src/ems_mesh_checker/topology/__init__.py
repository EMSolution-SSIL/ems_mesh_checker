"""Topology primitives shared by boundary extraction and mesh checks."""

from .element_topology import ElementTopology, get_element_topology
from .face import FaceOwner, FaceRecord
from .face_table import FaceTable, TopologyError

__all__ = [
    "ElementTopology",
    "FaceOwner",
    "FaceRecord",
    "FaceTable",
    "TopologyError",
    "get_element_topology",
]
