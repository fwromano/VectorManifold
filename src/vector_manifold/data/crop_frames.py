"""Use YOLOv8 segmentation to isolate cars and save cropped versions of frames."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO


COCO_CAR_CLASS_ID = 2  # 0-indexed (person=0, bicycle=1, car=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply YOLOv8 segmentation to crop car regions.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Path to frames directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where to write cropped frames")
    parser.add_argument("--model", default="yolov8n-seg.pt", help="Ultralytics model checkpoint")
    parser.add_argument("--confidence", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--margin", type=int, default=5, help="Extra pixels to add around the crop")
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=[COCO_CAR_CLASS_ID],
        help="COCO class IDs to accept (default: car only)",
    )
    return parser.parse_args()


def load_sorted_images(input_dir: Path) -> list[Path]:
    return sorted(p for p in input_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})


def crop_with_mask(image: Image.Image, mask_bool: np.ndarray, margin: int) -> Image.Image:
    if not mask_bool.any():
        raise ValueError("Empty mask")
    img_w, img_h = image.size

    coords = np.argwhere(mask_bool)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    x1 = x_min - margin
    x2 = x_max + margin
    y1 = y_min - margin
    y2 = y_max + margin

    x1 = max(int(np.floor(x1)), 0)
    y1 = max(int(np.floor(y1)), 0)
    x2 = min(int(np.ceil(x2)), img_w - 1)
    y2 = min(int(np.ceil(y2)), img_h - 1)

    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid crop bounds")

    return image.crop((x1, y1, x2 + 1, y2 + 1))


def process_frames(model_path: str, input_dir: Path, output_dir: Path, conf: float, margin: int,
                   model: YOLO | None = None, target_classes: set[int] | None = None) -> None:
    owns_model = False
    if model is None:
        model = YOLO(model_path)
        owns_model = True
    if target_classes is None:
        target_classes = {COCO_CAR_CLASS_ID}
    else:
        target_classes = set(target_classes)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = load_sorted_images(input_dir)
    for idx, frame_path in enumerate(frame_paths, start=1):
        image = Image.open(frame_path).convert("RGB")
        result_list = model.predict(source=str(frame_path), conf=conf, verbose=False)
        if not result_list:
            continue
        result = result_list[0]
        if result.masks is None or result.boxes is None:
            continue

        # collect car masks and pick the largest area
        best_mask = None
        best_area = 0.0
        for mask, box, cls in zip(result.masks.data, result.boxes.xyxy, result.boxes.cls):
            if int(cls.item()) not in target_classes:
                continue
            mask_np = mask.cpu().numpy()
            area = float(mask_np.sum())
            if area > best_area:
                best_area = area
                best_mask = mask_np
        if best_mask is None:
            continue

        try:
            mask_img = Image.fromarray((best_mask > 0.5).astype(np.uint8) * 255)
            resized_mask = mask_img.resize(image.size, resample=Image.NEAREST)
            mask_bool = np.array(resized_mask, dtype=bool)
            cropped = crop_with_mask(image, mask_bool, margin)
        except ValueError:
            continue

        cropped.save(output_dir / frame_path.name)
        if idx % 25 == 0:
            print(f"Processed {idx} frames", flush=True)

    if owns_model:
        del model


def main() -> None:
    args = parse_args()
    process_frames(
        args.model,
        args.input_dir,
        args.output_dir,
        args.confidence,
        args.margin,
        target_classes=set(args.classes),
    )


if __name__ == "__main__":
    main()
