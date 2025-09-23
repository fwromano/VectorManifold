"""Utilities to drop near-identical frames before downstream processing."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove near-duplicate frames based on pixel similarity.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing raw frames")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write deduplicated frames")
    parser.add_argument("--threshold", type=float, default=0.01,
                        help="Minimum normalized mean absolute difference to keep a frame")
    parser.add_argument("--resize", type=int, default=128,
                        help="Side length used for downsampled comparison patches")
    return parser.parse_args()


def load_frame_list(input_dir: Path) -> list[Path]:
    frames = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not frames:
        raise FileNotFoundError(f"No frames found under {input_dir}")
    return frames


def frame_descriptor(image: Image.Image, resize: int) -> np.ndarray:
    patch = image.resize((resize, resize), Image.BILINEAR).convert("L")
    return np.asarray(patch, dtype=np.uint8)


def normalized_mad(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return float(diff.mean() / 255.0)


def deduplicate_frames(input_dir: Path, output_dir: Path, threshold: float, resize: int) -> dict[str, int | float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    kept = 0
    skipped = 0
    max_gap = 0.0
    min_gap = float("inf")

    previous_descriptor: np.ndarray | None = None
    frames = load_frame_list(input_dir)
    for frame_path in frames:
        with Image.open(frame_path) as img:
            rgb = img.convert("RGB")
            descriptor = frame_descriptor(rgb, resize)

        keep = False
        gap = 1.0
        if previous_descriptor is None:
            keep = True
            gap = 1.0
        else:
            gap = normalized_mad(descriptor, previous_descriptor)
            min_gap = min(min_gap, gap)
            if gap >= threshold:
                keep = True

        if keep:
            kept += 1
            previous_descriptor = descriptor
            destination = output_dir / f"frame_{kept:05d}{frame_path.suffix.lower()}"
            rgb.save(destination)
            max_gap = max(max_gap, gap)
        else:
            skipped += 1
            max_gap = max(max_gap, gap)
            min_gap = min(min_gap, gap)

    return {
        "input_frames": len(frames),
        "kept_frames": kept,
        "skipped_frames": skipped,
        "max_gap": max_gap,
        "min_gap": 0.0 if min_gap == float("inf") else min_gap,
    }


def main() -> None:
    args = parse_args()
    stats = deduplicate_frames(args.input_dir, args.output_dir, args.threshold, args.resize)
    print(f"Kept {stats['kept_frames']} / {stats['input_frames']} frames (threshold={args.threshold})")
    print(
        "Skipped {skipped} duplicates; min gap {min_gap:.6f}, max gap {max_gap:.4f}".format(
            skipped=stats["skipped_frames"],
            min_gap=stats["min_gap"],
            max_gap=stats["max_gap"],
        )
    )


if __name__ == "__main__":
    main()
