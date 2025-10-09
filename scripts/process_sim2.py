#!/usr/bin/env python3
"""Process sim2 videos: crop vehicles with YOLOv8 and save one frame per angle."""

from __future__ import annotations

import sys

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure Ultralytics writes configs inside the workspace
WORKSPACE_CACHE = Path(".cache")
WORKSPACE_CACHE.mkdir(exist_ok=True)
for subdir in ("xdg", "ultralytics"):
    (WORKSPACE_CACHE / subdir).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CONFIG_HOME", str((WORKSPACE_CACHE / "xdg").resolve()))
os.environ.setdefault("ULTRALYTICS_SETTINGS_DIR", str((WORKSPACE_CACHE / "ultralytics").resolve()))

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from vector_manifold.data.crop_frames import crop_with_mask


@dataclass(frozen=True)
class AngleExtractionConfig:
    roi_x: int = 0
    roi_y: int = 0
    roi_width: int = 220
    roi_height: int = 80
    digits_start_x: int = 100
    min_width: int = 5
    min_height: int = 10
    min_area: int = 40
    threshold: int = 200
    template_scale: float = 1.0
    template_thickness: int = 2
    template_size: int = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process rotating vehicle videos and save one frame per angle.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw/video/sim2"),
                        help="Directory containing input .mp4 files.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/sim2_cropped"),
                        help="Directory to write cropped frames.")
    parser.add_argument("--model", type=Path, default=Path("assets/models/yolov8n-seg.pt"),
                        help="Path to YOLOv8 segmentation checkpoint.")
    parser.add_argument("--confidence", type=float, default=0.35, help="YOLO confidence threshold.")
    parser.add_argument("--margin", type=int, default=5, help="Pixel margin around YOLO mask crop.")
    parser.add_argument("--classes", type=int, nargs="+", default=[2, 7],
                        help="COCO class IDs to retain (default: car=2, truck=7).")
    parser.add_argument("--max-angle", type=int, default=360,
                        help="Stop processing a video after this many unique angles have been saved.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip angles that already exist on disk.")
    return parser.parse_args()


def discover_videos(input_dir: Path) -> List[Path]:
    videos = sorted([p for p in input_dir.glob("*.mp4") if p.is_file()])
    if not videos:
        raise FileNotFoundError(f"No .mp4 files found under {input_dir}")
    return videos


def pad_to_square(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    size = max(h, w)
    top = (size - h) // 2
    bottom = size - h - top
    left = (size - w) // 2
    right = size - w - left
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)


def create_digit_templates(config: AngleExtractionConfig) -> Dict[str, np.ndarray]:
    font = cv2.FONT_HERSHEY_SIMPLEX
    templates: Dict[str, np.ndarray] = {}
    for digit in "0123456789":
        canvas = np.zeros((60, 40), dtype=np.uint8)
        cv2.putText(canvas, digit, (2, 45), font, config.template_scale, 255, config.template_thickness, cv2.LINE_AA)
        ys, xs = np.where(canvas > 0)
        if not len(xs):
            raise RuntimeError("Failed to render digit template.")
        crop = canvas[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
        crop = pad_to_square(crop)
        crop = cv2.resize(crop, (config.template_size, config.template_size), interpolation=cv2.INTER_AREA)
        _, crop = cv2.threshold(crop, 128, 255, cv2.THRESH_BINARY)
        templates[digit] = crop
    return templates


def classify_digit(char_img: np.ndarray, templates: Dict[str, np.ndarray]) -> Optional[str]:
    best_digit: Optional[str] = None
    best_score = float("inf")
    for digit, template in templates.items():
        score = int(np.sum(cv2.absdiff(char_img, template)))
        if score < best_score:
            best_score = score
            best_digit = digit
    return best_digit


def extract_angle(frame: np.ndarray, config: AngleExtractionConfig,
                  templates: Dict[str, np.ndarray]) -> Optional[int]:
    x0 = config.roi_x
    y0 = config.roi_y
    x1 = min(frame.shape[1], x0 + config.roi_width)
    y1 = min(frame.shape[0], y0 + config.roi_height)
    if x0 >= x1 or y0 >= y1:
        return None

    roi = frame[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, config.threshold, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: List[Tuple[int, int, int, int]] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if x + w <= config.digits_start_x:
            continue
        if w < config.min_width or h < config.min_height or w * h < config.min_area:
            continue
        candidates.append((x, y, w, h))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    digits: List[str] = []
    for x, y, w, h in candidates:
        char_img = binary[y:y + h, x:x + w]
        char_img = pad_to_square(char_img)
        char_img = cv2.resize(char_img, (config.template_size, config.template_size), interpolation=cv2.INTER_AREA)
        _, char_img = cv2.threshold(char_img, 128, 255, cv2.THRESH_BINARY)
        digit = classify_digit(char_img, templates)
        if digit is None:
            return None
        digits.append(digit)

    try:
        return int("".join(digits))
    except ValueError:
        return None


def crop_vehicle(frame: np.ndarray, model: YOLO, confidence: float, margin: int,
                 target_classes: set[int]) -> Optional[Image.Image]:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result_list = model.predict(source=rgb, conf=confidence, verbose=False, classes=list(target_classes))
    if not result_list:
        return None
    result = result_list[0]
    if result.masks is None or result.boxes is None:
        return None

    best_mask = None
    best_area = 0.0
    for mask, cls in zip(result.masks.data, result.boxes.cls):
        if int(cls.item()) not in target_classes:
            continue
        mask_np = mask.cpu().numpy()
        area = float(mask_np.sum())
        if area > best_area:
            best_area = area
            best_mask = mask_np

    if best_mask is None:
        return None

    image = Image.fromarray(rgb)
    mask_img = Image.fromarray((best_mask > 0.5).astype(np.uint8) * 255)
    mask_resized = mask_img.resize(image.size, resample=Image.NEAREST)
    mask_bool = np.array(mask_resized, dtype=bool)

    try:
        cropped = crop_with_mask(image, mask_bool, margin)
    except ValueError:
        return None

    return cropped


def ensure_output_dir(base: Path, video_name: str) -> Path:
    out_dir = base / video_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def existing_angles(output_dir: Path) -> set[int]:
    angles = set()
    for path in output_dir.glob("*.png"):
        name = path.stem
        if "_" not in name:
            continue
        _, angle_str = name.rsplit("_", 1)
        try:
            angles.add(int(angle_str))
        except ValueError:
            continue
    return angles


def process_video(video_path: Path, output_dir: Path, model: YOLO, args: argparse.Namespace,
                  angle_cfg: AngleExtractionConfig, templates: Dict[str, np.ndarray],
                  target_classes: set[int]) -> dict[str, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Skipping {video_path.name}: unable to open video stream.")
        return {"frames": 0, "saved": 0, "angles": 0}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    seen_angles: set[int] = existing_angles(output_dir) if args.skip_existing else set()
    saved = len(seen_angles)
    processed_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed_frames += 1

        angle = extract_angle(frame, angle_cfg, templates)
        if angle is None or angle in seen_angles:
            continue

        cropped = crop_vehicle(frame, model, args.confidence, args.margin, target_classes)
        if cropped is None:
            continue

        filename = f"{video_path.stem}_{angle:03d}.png"
        cropped.save(output_dir / filename)
        seen_angles.add(angle)
        saved += 1

        if args.max_angle > 0 and saved >= args.max_angle:
            break

    cap.release()
    status = {
        "frames": processed_frames if processed_frames else total_frames,
        "saved": saved,
        "angles": len(seen_angles),
    }
    print(f"{video_path.name}: processed {status['frames']} frames, saved {status['saved']} unique angles.")
    return status


def main() -> None:
    args = parse_args()
    videos = discover_videos(args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    angle_cfg = AngleExtractionConfig()
    templates = create_digit_templates(angle_cfg)
    model = YOLO(str(args.model))
    target_classes = set(args.classes)
    stats: dict[str, dict[str, int]] = {}

    for video in videos:
        video_out = ensure_output_dir(args.output_dir, video.stem)
        stats[video.stem] = process_video(video, video_out, model, args, angle_cfg, templates, target_classes)

    print("\nSummary:")
    for name, stat in stats.items():
        print(f"  {name}: frames={stat['frames']}, unique_angles={stat['angles']}, saved={stat['saved']}")


if __name__ == "__main__":
    main()
