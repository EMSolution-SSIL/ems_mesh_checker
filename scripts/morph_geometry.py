"""Repository checkout entry point for geometry-following mesh morphing."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ems_mesh_checker.morph_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
