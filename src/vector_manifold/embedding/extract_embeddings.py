"""FastViT embedding explorer.

This script helps investigate how image transformations (e.g., rotation) change the
embeddings produced by a FastViT model from TIMM. It can operate on an existing
folder of images or generate rotated variants of a single image on the fly.

Example usage (generate rotations):
    python scripts/fastvit_embedding_explorer.py \
        --base-image data/car.jpg \
        --angles "0,30,60,90,120,150,180" \
        --output-dir artifacts/runs/car_rotations

Example usage (directory of images):
    python scripts/fastvit_embedding_explorer.py \
        --image-dir data/cars \
        --output-dir artifacts/runs/car_dataset
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import timm
from timm.data import create_transform, resolve_data_config


ImageRecord = Tuple[str, Optional[float], Image.Image, str]


@dataclass
class Sample:
    """Container for a single image embedding and metadata."""

    uid: str
    source: str
    angle: Optional[float]
    embedding: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explore FastViT embeddings.")
    parser.add_argument(
        "--model",
        default="fastvit_sa12",
        help="FastViT model name from timm (default: fastvit_sa12)"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (default: auto-detect)"
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        help="Directory of images to embed."
    )
    parser.add_argument(
        "--base-image",
        type=Path,
        help="Single image to generate rotations from (requires --angles or --angle-step)."
    )
    parser.add_argument(
        "--angles",
        type=str,
        help="Comma-separated rotation angles in degrees for --base-image."
    )
    parser.add_argument(
        "--angle-step",
        type=float,
        help="Generate rotations from 0 to <360 with this step, e.g., 30."
    )
    parser.add_argument(
        "--expand-rotation",
        action="store_true",
        help="Expand image bounds when rotating (avoids cropping)."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where results (embeddings, plots) will be stored."
    )
    parser.add_argument(
        "--save-rotations",
        action="store_true",
        help="Save generated rotated images into the output directory."
    )
    parser.add_argument(
        "--crop-square",
        action="store_true",
        help=(
            "After rotation, center-crop the largest possible square before feeding the"
            " model. Useful when expand-rotation introduces empty borders."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for embedding extraction (default: 16)."
    )
    args = parser.parse_args()

    if args.image_dir is None and args.base_image is None:
        parser.error("Provide either --image-dir or --base-image (with rotations).")

    if args.base_image is not None:
        if not args.base_image.exists():
            parser.error(f"Base image not found: {args.base_image}")
        if args.angles is None and args.angle_step is None:
            parser.error("Provide --angles or --angle-step when using --base-image.")

    if args.image_dir is not None and not args.image_dir.exists():
        parser.error(f"Image directory not found: {args.image_dir}")

    return args


def get_angles(args: argparse.Namespace) -> List[float]:
    if args.angles:
        return [float(x.strip()) for x in args.angles.split(",") if x.strip()]
    if args.angle_step:
        steps = np.arange(0, 360, args.angle_step, dtype=float)
        return steps.tolist()
    return []


def build_model(model_name: str, device: str) -> tuple[torch.nn.Module, dict]:
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model.eval()
    model.to(device)
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config, is_training=False)
    return model, {"transform": transform, "config": config}


def load_image(path: Path) -> Image.Image:
    image = Image.open(path).convert("RGB")
    return image


def rotate_image(image: Image.Image, angle: float, expand: bool, crop_square: bool) -> Image.Image:
    rotated = image.rotate(angle, resample=Image.BICUBIC, expand=expand)
    if crop_square:
        width, height = rotated.size
        min_side = min(width, height)
        left = (width - min_side) / 2
        top = (height - min_side) / 2
        right = left + min_side
        bottom = top + min_side
        rotated = rotated.crop((left, top, right, bottom))
    return rotated


def batched(iterable: Sequence[ImageRecord], batch_size: int):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def compute_embeddings(
    model: torch.nn.Module,
    transform,
    device: str,
    samples: Sequence[ImageRecord],
    batch_size: int,
) -> List[Sample]:
    results: List[Sample] = []
    for batch in batched(samples, batch_size):
        ids, angles, images, sources = zip(*batch)
        tensors = torch.stack([transform(img) for img in images]).to(device)
        with torch.no_grad():
            embeds = model(tensors).cpu().numpy()
        for uid, angle, embedding, source in zip(ids, angles, embeds, sources):
            results.append(Sample(uid=uid, source=source, angle=angle, embedding=embedding))
    return results


def load_samples_from_dir(image_dir: Path) -> List[ImageRecord]:
    paths = sorted(
        [p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
    )
    if not paths:
        raise RuntimeError(f"No image files found in {image_dir}")
    return [(p.name, None, load_image(p), str(p)) for p in paths]


def generate_rotated_samples(
    base_image_path: Path,
    angles: Iterable[float],
    expand: bool,
    crop_square: bool,
) -> List[ImageRecord]:
    base = load_image(base_image_path)
    samples: List[ImageRecord] = []
    for angle in angles:
        rotated = rotate_image(base, angle, expand=expand, crop_square=crop_square)
        uid = f"{base_image_path.stem}_rot{angle:g}{base_image_path.suffix}"
        source = f"rotation:{angle:g}"
        samples.append((uid, angle, rotated, source))
    return samples


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_metadata(output_dir: Path, samples: List[Sample], model_name: str, config: dict) -> None:
    meta_path = output_dir / "metadata.csv"
    with meta_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["uid", "source", "angle_deg"])
        for sample in samples:
            angle_str = "" if sample.angle is None else f"{sample.angle:.6f}"
            writer.writerow([sample.uid, sample.source, angle_str])

    info = {
        "model": model_name,
        "num_samples": len(samples),
        "embedding_dim": int(samples[0].embedding.size) if samples else 0,
        "config": config,
    }
    with (output_dir / "run_info.json").open("w") as f:
        json.dump(info, f, indent=2)

    embeddings = np.stack([s.embedding for s in samples], axis=0)
    np.save(output_dir / "embeddings.npy", embeddings)


def compute_pairwise_metrics(samples: List[Sample], output_dir: Path) -> dict:
    embeddings = np.stack([s.embedding for s in samples], axis=0)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    cosine = embeddings @ embeddings.T / (norms @ norms.T)
    np.save(output_dir / "cosine_similarity.npy", cosine)

    euclidean = np.sqrt(np.maximum(0.0, np.sum((embeddings[:, None, :] - embeddings[None, :, :]) ** 2, axis=-1)))
    np.save(output_dir / "euclidean_distance.npy", euclidean)

    return {"cosine": cosine, "euclidean": euclidean}


def plot_angle_relationship(samples: List[Sample], cosine: np.ndarray, output_dir: Path) -> None:
    angles = [s.angle for s in samples]
    if any(angle is None for angle in angles):
        return

    angle_pairs = []
    cosine_pairs = []
    n = len(samples)
    for i in range(n):
        for j in range(i + 1, n):
            delta = abs(angles[i] - angles[j])
            delta = min(delta, 360 - delta)  # handle wrap-around
            angle_pairs.append(delta)
            cosine_pairs.append(cosine[i, j])

    plt.figure(figsize=(8, 5))
    plt.scatter(angle_pairs, cosine_pairs, alpha=0.6)
    plt.title("Cosine similarity vs. rotation difference")
    plt.xlabel("Absolute angle difference (deg)")
    plt.ylabel("Cosine similarity")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "cosine_vs_angle.png", dpi=160)
    plt.close()


def maybe_save_rotations(
    generated_samples: Sequence[ImageRecord],
    output_dir: Path,
) -> None:
    rotations_dir = output_dir / "generated_rotations"
    rotations_dir.mkdir(exist_ok=True, parents=True)
    for uid, angle, image, _ in generated_samples:
        if angle is None:
            continue
        suffix = ".png"
        image.save(rotations_dir / f"{uid.replace('/', '_')}.png")


def main() -> None:
    args = parse_args()
    ensure_output_dir(args.output_dir)

    model, data_assets = build_model(args.model, args.device)
    transform = data_assets["transform"]

    samples_source: List[ImageRecord] = []
    generated: List[ImageRecord] = []

    if args.image_dir is not None:
        samples_source.extend(load_samples_from_dir(args.image_dir))

    if args.base_image is not None:
        angles = get_angles(args)
        generated = generate_rotated_samples(
            args.base_image, angles, expand=args.expand_rotation, crop_square=args.crop_square
        )
        samples_source.extend(generated)
        if args.save_rotations:
            maybe_save_rotations(generated, args.output_dir)

    # Deduplicate IDs if necessary while preserving original sources.
    used_ids = {}
    normalized_samples: List[ImageRecord] = []
    for uid, angle, image, source in samples_source:
        count = used_ids.get(uid, 0)
        if count:
            new_uid = f"{uid}_{count}"
        else:
            new_uid = uid
        used_ids[uid] = count + 1
        normalized_samples.append((new_uid, angle, image, source))

    embedding_samples = compute_embeddings(
        model, transform, args.device, normalized_samples, args.batch_size
    )

    if not embedding_samples:
        raise RuntimeError("No embeddings were computed.")

    save_metadata(args.output_dir, embedding_samples, args.model, data_assets["config"])

    metrics = compute_pairwise_metrics(embedding_samples, args.output_dir)
    plot_angle_relationship(embedding_samples, metrics["cosine"], args.output_dir)


if __name__ == "__main__":
    main()
