#!/usr/bin/env python3
"""CLI wrapper for pose subspace pipeline."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from vector_manifold.rotation.pose_pipeline import main


if __name__ == "__main__":  # pragma: no cover
    main()
