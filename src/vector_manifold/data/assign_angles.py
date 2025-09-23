"""Assign approximate viewing angles to car rotation frames and update metadata."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

# Key frames provided by the user (1-indexed frame numbers -> angle in degrees).
ANCHORS: Sequence[Tuple[int, float]] = (
    (17, 0.0),    # front
    (34, 90.0),   # right side
    (51, 180.0),  # back
    (63, 270.0),  # left side
    (78, 0.0),    # front again
    (103, 270.0), # left side (reverse sweep)
    (114, 180.0), # back
    (130, 90.0),  # right side
    (160, 0.0),   # front
)

DEFAULT_RUNS = (
    Path("artifacts/runs/car_rotation3_fastvit_cropped"),
)


def shortest_delta(start: float, end: float) -> float:
    options = [end - start, end - start + 360.0, end - start - 360.0]
    return min(options, key=abs)


def compute_angles(num_frames: int) -> tuple[np.ndarray, np.ndarray]:
    anchors = sorted(ANCHORS, key=lambda x: x[0])
    anchors = [(max(1, min(num_frames, frame)), angle) for frame, angle in anchors]

    frame_idxs = [f - 1 for f, _ in anchors]
    unwrapped_vals: List[float] = []

    prev_mod = anchors[0][1]
    prev_unwrapped = prev_mod
    unwrapped_vals.append(prev_unwrapped)

    for (_, angle_mod), frame_idx in zip(anchors[1:], frame_idxs[1:]):
        delta = shortest_delta(prev_mod, angle_mod)
        prev_unwrapped += delta
        unwrapped_vals.append(prev_unwrapped)
        prev_mod = angle_mod

    frames_array = np.asarray(frame_idxs, dtype=float)
    values_array = np.asarray(unwrapped_vals, dtype=float)
    full_idx = np.arange(num_frames, dtype=float)

    interpolated = np.interp(full_idx, frames_array, values_array)

    if frames_array[0] > 0:
        slope = (values_array[1] - values_array[0]) / (frames_array[1] - frames_array[0])
        for i in range(int(frames_array[0])):
            interpolated[i] = values_array[0] + slope * (i - frames_array[0])

    if frames_array[-1] < num_frames - 1:
        slope = (values_array[-1] - values_array[-2]) / (frames_array[-1] - frames_array[-2])
        for i in range(int(frames_array[-1]) + 1, num_frames):
            interpolated[i] = values_array[-1] + slope * (i - frames_array[-1])

    wrapped = np.mod(interpolated, 360.0)
    return wrapped, interpolated


def parse_frame_index(name: str) -> int:
    digits = "".join(ch for ch in name if ch.isdigit())
    return int(digits) if digits else 0


def update_run(run_dir: Path, angles_per_frame: np.ndarray) -> None:
    metadata_path = run_dir / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.csv not found in {run_dir}")

    with metadata_path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))

    updated_rows = []
    angles_for_rows: List[float] = []
    for row in rows:
        frame_idx = parse_frame_index(row["uid"])
        if frame_idx <= 0 or frame_idx > len(angles_per_frame):
            angle_val = np.nan
        else:
            angle_val = angles_per_frame[frame_idx - 1]
        row["angle_deg"] = f"{angle_val:.6f}" if not np.isnan(angle_val) else ""
        updated_rows.append(row)
        angles_for_rows.append(angle_val)

    with metadata_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["uid", "source", "angle_deg"])
        writer.writeheader()
        writer.writerows(updated_rows)

    angles_array = np.asarray(angles_for_rows, dtype=float)
    np.save(run_dir / "angles.npy", angles_array)
    print(f"Updated {run_dir} with angle annotations.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Assign viewing angles based on anchor frames.")
    parser.add_argument("--run", action="append", type=Path, help="Output run directory (repeatable)")
    parser.add_argument("--frames", type=int, default=180, help="Total number of frames")
    args = parser.parse_args()

    runs = args.run if args.run else DEFAULT_RUNS
    angles_wrapped, angles_unwrapped = compute_angles(args.frames)

    for run_dir in runs:
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        update_run(run_dir, angles_wrapped)
        np.save(run_dir / "angles_unwrapped.npy", angles_unwrapped)


if __name__ == "__main__":
    main()
