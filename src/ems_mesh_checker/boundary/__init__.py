"""Property-boundary extraction from a topology face table."""

from .extractor import BoundaryExtractionError, BoundaryExtractor
from .result import BoundaryFace, BoundaryResult, QuadTriInterface

__all__ = [
    "BoundaryExtractionError",
    "BoundaryExtractor",
    "BoundaryFace",
    "BoundaryResult",
    "QuadTriInterface",
]
