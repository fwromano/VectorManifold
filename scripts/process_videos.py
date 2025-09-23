#!/usr/bin/env python3
"""End-to-end pipeline: extract, deduplicate, and crop vehicle frames for multiple videos."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Ensure Ultralytics writes configs inside the workspace
WORKSPACE_CACHE = Path(".cache")
WORKSPACE_CACHE.mkdir(exist_ok=True)
for subdir in ("xdg", "ultralytics"):
    (WORKSPACE_CACHE / subdir).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CONFIG_HOME", str((WORKSPACE_CACHE / "xdg").resolve()))
os.environ.setdefault("ULTRALYTICS_SETTINGS_DIR", str((WORKSPACE_CACHE / "ultralytics").resolve()))

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ultralytics import YOLO  # noqa: E402

from vector_manifold.data.crop_frames import process_frames  # noqa: E402
from vector_manifold.data.deduplicate_frames import deduplicate_frames  # noqa: E402


@dataclass
class PipelineStats:
    video_name: str
    extracted: int
    kept: int
    skipped_duplicates: int
    cropped: int
    min_gap: float
    max_gap: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process raw videos into deduplicated, cropped frame sets.")
    parser.add_argument("--videos", nargs="*", default=None,
                        help="Video basenames to process (without extension). Defaults to all .mp4 under data/raw/video")
    parser.add_argument("--fps", type=float, default=10, help="Target FPS for frame extraction")
    parser.add_argument(
        "--scale",
        type=int,
        default=512,
        help="Target long-side resolution for frames (<=0 keeps native resolution)",
    )
    parser.add_argument("--threshold", type=float, default=0.01,
                        help="Normalized MAD threshold for deduplication")
    parser.add_argument("--resize", type=int, default=128,
                        help="Resolution for deduplication comparison patches")
    parser.add_argument("--confidence", type=float, default=0.35, help="YOLO confidence threshold")
    parser.add_argument("--margin", type=int, default=5, help="Pixel margin to add around YOLO mask bounds")
    parser.add_argument("--model", default="assets/models/yolov8n-seg.pt", help="Path to a YOLO segmentation checkpoint")
    parser.add_argument("--classes", type=int, nargs="+", default=[2, 7],
                        help="COCO class IDs to retain (default: car and truck)")
    parser.add_argument("--keep-temp", action="store_true", help="Retain intermediate raw frames for inspection")
    return parser.parse_args()


def discover_videos(explicit: Iterable[str] | None) -> list[Path]:
    video_root = Path("data/raw/video")
    if explicit:
        videos = [video_root / f"{name}.mp4" for name in explicit]
    else:
        videos = sorted(video_root.glob("*.mp4"))
    missing = [str(path) for path in videos if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing videos: {', '.join(missing)}")
    return videos


def safe_rmtree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def count_frames(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(1 for p in directory.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})


def extract_frames(video: Path, output_dir: Path, fps: float, scale: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = output_dir / "frame_%05d.png"
    if scale and scale > 0:
        vf = f"fps={fps},scale={scale}:-2"
    else:
        vf = f"fps={fps}"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video),
        "-vf",
        vf,
        str(pattern),
    ]
    subprocess.run(cmd, check=True)


def process_video(video_path: Path, args: argparse.Namespace, model: YOLO) -> PipelineStats:
    name = video_path.stem
    tmp_dir = Path("data/processed/frames") / f"{name}_raw"
    dedup_dir = Path("data/processed/frames") / name
    crop_dir = Path("data/processed/frames") / f"{name}_cropped"

    print(f"\n=== Processing {name} ===")

    safe_rmtree(tmp_dir)
    safe_rmtree(dedup_dir)
    safe_rmtree(crop_dir)

    extract_frames(video_path, tmp_dir, args.fps, args.scale)
    extracted = count_frames(tmp_dir)
    print(f"Extracted {extracted} frames from {video_path}")

    stats = deduplicate_frames(tmp_dir, dedup_dir, args.threshold, args.resize)
    kept = stats["kept_frames"]
    skipped = stats["skipped_frames"]
    min_gap = stats["min_gap"]
    max_gap = stats["max_gap"]
    print(
        f"Deduplicated to {kept} frames (removed {skipped}); min gap {min_gap:.6f}, max gap {max_gap:.4f}"
    )

    process_frames(
        args.model,
        dedup_dir,
        crop_dir,
        args.confidence,
        args.margin,
        model=model,
        target_classes=set(args.classes),
    )
    cropped = count_frames(crop_dir)
    print(f"Cropped {cropped} frames")

    if not args.keep_temp:
        safe_rmtree(tmp_dir)

    return PipelineStats(
        video_name=name,
        extracted=extracted,
        kept=kept,
        skipped_duplicates=skipped,
        cropped=cropped,
        min_gap=min_gap,
        max_gap=max_gap,
    )


def main() -> None:
    args = parse_args()
    videos = discover_videos(args.videos)
    if not videos:
        print("No videos to process.")
        return

    # Warm up YOLO once for all videos
    model = YOLO(args.model)

    summaries: list[PipelineStats] = []
    for video_path in videos:
        summaries.append(process_video(video_path, args, model))

    print("\n=== Summary ===")
    header = f"{'Video':<15}{'Extracted':>12}{'Unique':>12}{'Removed':>12}{'Cropped':>12}{'MinGap':>10}"
    print(header)
    print("-" * len(header))
    for summary in summaries:
        print(
            f"{summary.video_name:<15}{summary.extracted:>12}{summary.kept:>12}{summary.skipped_duplicates:>12}"
            f"{summary.cropped:>12}{summary.min_gap:>10.6f}"
        )


if __name__ == "__main__":
    main()
