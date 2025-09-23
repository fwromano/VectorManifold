#!/usr/bin/env python3
"""CLI wrapper to build and run semantic dictionary inference."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from vector_manifold.semantic.describer import main


if __name__ == "__main__":  # pragma: no cover
    main()
