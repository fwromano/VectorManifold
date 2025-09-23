#!/usr/bin/env python3
"""CLI wrapper for YOLO-based cropping."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from vector_manifold.data.crop_frames import main


if __name__ == "__main__":  # pragma: no cover
    main()
