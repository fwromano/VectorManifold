#!/usr/bin/env python3
"""Select evenly spaced angles from processed sim2 crops."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import shutil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pick representative frames every N degrees.")
    parser.add_argument("--source-root", type=Path, default=Path("data/processed/sim2_cropped"),
                        help="Root directory containing per-video folders of cropped frames.")
    parser.add_argument("--dest-root", type=Path, default=Path("data/processed/sim2_cropped_step"))
    parser.add_argument("--step", type=int, default=45, help="Angle step in degrees.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    parser.add_argument("--tolerance", type=float, default=7.5,
                        help="Maximum allowed angular difference when selecting frames.")
    return parser.parse_args()


def load_angles(folder: Path) -> List[Tuple[int, Path]]:
    entries: List[Tuple[int, Path]] = []
    for image_path in sorted(folder.glob("*.png")):
        name = image_path.stem
        if "_" not in name:
            continue
        angle_str = name.rsplit("_", 1)[-1]
        try:
            angle_val = int(angle_str)
        except ValueError:
            continue
        entries.append((angle_val, image_path))
    return entries


def angular_distance(a: float, b: float) -> float:
    return abs(((a - b + 180.0) % 360.0) - 180.0)


def pick_frames(angles: List[Tuple[int, Path]], step: int, tolerance: float) -> List[Tuple[int, int, Optional[Path], float]]:
    if not angles:
        return []

    picks: List[Tuple[int, int, Path, float]] = []
    normalized: List[Tuple[float, int, Path]] = []
    for actual, path in angles:
        norm = float(actual % 360)
        normalized.append((norm, actual, path))

    for target in range(0, 360, step):
        best_angle: Optional[int] = None
        best_path: Optional[Path] = None
        best_delta = float("inf")

        for norm, actual, path in normalized:
            delta = angular_distance(norm, target)
            if delta < best_delta - 1e-6 or (abs(delta - best_delta) <= 1e-6 and (best_angle is None or actual < best_angle)):
                best_delta = delta
                best_angle = actual
                best_path = path

        if best_path is None:
            picks.append((target, -1, None, float("inf")))
        else:
            picks.append((target, best_angle if best_angle is not None else -1, best_path, best_delta))

    return picks


def main() -> None:
    args = parse_args()
    args.dest_root.mkdir(parents=True, exist_ok=True)

    for folder in sorted(args.source_root.iterdir()):
        if not folder.is_dir() or folder.name.startswith("."):
            continue
        angles = load_angles(folder)
        picks = pick_frames(angles, args.step, args.tolerance)
        dest_dir = args.dest_root / folder.name
        dest_dir.mkdir(parents=True, exist_ok=True)

        for target, actual, path, delta in picks:
            out_name = f"{folder.name}_{target:03d}.png"
            out_path = dest_dir / out_name

            if path is None or delta == float("inf"):
                print(f"[WARN] {folder.name}: no frame within tolerance for target {target}°.")
                if out_path.exists() and args.overwrite:
                    out_path.unlink()
                continue

            if delta > args.tolerance:
                print(f"[WARN] {folder.name}: target {target}° uses {actual}° (Δ={delta:.1f}°).")

            if out_path.exists() and not args.overwrite:
                continue

            shutil.copy2(path, out_path)

        selected = sum(1 for _, _, p, d in picks if p is not None and d != float("inf"))
        print(f"{folder.name}: selected {selected} frames.")


if __name__ == "__main__":
    main()
