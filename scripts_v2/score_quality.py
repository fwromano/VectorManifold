#!/usr/bin/env python3
"""Compute per-frame quality features for calibration."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from vector_manifold.v2.features import load_image_paths, load_masks
from vector_manifold.v2.quality import compute_quality_dataframe, save_quality_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score per-frame quality metrics")
    parser.add_argument("--frames", type=Path, required=True, help="Directory with frame images")
    parser.add_argument("--run", type=Path, required=True, help="Run directory for outputs")
    parser.add_argument("--mask-dir", type=Path, help="Optional directory with binary masks")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_paths = load_image_paths(args.frames)
    masks = load_masks(args.mask_dir, image_paths) if args.mask_dir else None

    df = compute_quality_dataframe(image_paths, masks=masks)
    quality_path = args.run / "quality_v2.csv"
    save_quality_csv(df, quality_path)
    print(f"Saved quality metrics to {quality_path}")


if __name__ == "__main__":
    main()
