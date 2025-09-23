"""Compute CLIP image embeddings for a directory of frames."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image

try:
    import open_clip
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError("open_clip is required for CLIP embedding extraction") from exc


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract CLIP embeddings for a directory of images.")
    parser.add_argument("--image-dir", type=Path, required=True, help="Directory of images to embed")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write embeddings and metadata")
    parser.add_argument("--model", default="ViT-L-14", help="open_clip model name (default: ViT-L-14)")
    parser.add_argument("--pretrained", default="openai", help="open_clip pretrained tag (default: openai)")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to run on (default: cuda if available)",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference (default: 32)")
    parser.add_argument("--normalize", action="store_true", help="L2-normalize embeddings before saving")
    parser.add_argument(
        "--angle-step",
        type=float,
        default=None,
        help="If set, assign per-frame angles (degrees) assuming constant step",
    )
    parser.add_argument(
        "--start-angle",
        type=float,
        default=0.0,
        help="Starting angle (degrees) when --angle-step is provided",
    )
    return parser.parse_args()


def list_images(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise RuntimeError(f"No image files found under {image_dir}")
    return images


def batched(items: Iterable[Path], batch_size: int) -> Iterable[list[Path]]:
    batch: list[Path] = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _pairwise_cosine(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    normalized = embeddings / norms
    return normalized @ normalized.T


def _pairwise_euclidean(embeddings: np.ndarray) -> np.ndarray:
    sq_norms = np.sum(embeddings ** 2, axis=1, keepdims=True)
    distances_sq = sq_norms + sq_norms.T - 2.0 * (embeddings @ embeddings.T)
    np.maximum(distances_sq, 0.0, out=distances_sq)
    return np.sqrt(distances_sq, out=distances_sq)


def encode_directory(
    image_paths: list[Path],
    model_name: str,
    pretrained: str,
    device: str,
    batch_size: int,
    normalize: bool,
) -> tuple[np.ndarray, list[str]]:
    model, preprocess, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model.to(device)
    model.eval()

    embeddings: list[np.ndarray] = []
    identifiers: list[str] = []

    with torch.no_grad():
        for batch in batched(image_paths, batch_size):
            images = [preprocess(Image.open(path).convert("RGB")) for path in batch]
            inputs = torch.stack(images).to(device)
            feats = model.encode_image(inputs)
            if normalize:
                feats = torch.nn.functional.normalize(feats, dim=-1)
            embeddings.append(feats.cpu().numpy())
            identifiers.extend(path.name for path in batch)

    return np.concatenate(embeddings, axis=0), identifiers


def write_metadata(
    output_dir: Path,
    identifiers: list[str],
    image_paths: list[Path],
    angles: np.ndarray | None,
) -> None:
    metadata_path = output_dir / "metadata.csv"
    with metadata_path.open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["uid", "source", "index", "angle_deg"]
        writer.writerow(header)
        for idx, (name, path) in enumerate(zip(identifiers, image_paths)):
            angle_val = ""
            if angles is not None:
                angle_val = f"{angles[idx]:.6f}"
            writer.writerow([name, str(path), idx, angle_val])


def write_run_info(output_dir: Path, *, model_name: str, pretrained: str, device: str, batch_size: int, normalize: bool) -> None:
    info_path = output_dir / "run_info.json"
    payload = {
        "model": model_name,
        "pretrained": pretrained,
        "device": device,
        "batch_size": batch_size,
        "normalize": normalize,
    }
    info_path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    args = parse_args()

    image_paths = list_images(args.image_dir)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings, identifiers = encode_directory(
        image_paths,
        args.model,
        args.pretrained,
        args.device,
        args.batch_size,
        args.normalize,
    )

    np.save(output_dir / "embeddings.npy", embeddings)
    if args.angle_step is not None:
        angles = (args.start_angle + np.arange(len(identifiers), dtype=float) * args.angle_step) % 360.0
    else:
        angles = None
    write_metadata(output_dir, identifiers, image_paths, angles)
    if angles is not None:
        np.save(output_dir / "angles.npy", angles)
        np.save(output_dir / "angles_unwrapped.npy", args.start_angle + np.arange(len(identifiers), dtype=float) * args.angle_step)

    cosine = embeddings @ embeddings.T if args.normalize else _pairwise_cosine(embeddings)
    np.save(output_dir / "cosine_similarity.npy", cosine)
    euclidean = _pairwise_euclidean(embeddings)
    np.save(output_dir / "euclidean_distance.npy", euclidean)
    write_run_info(
        output_dir,
        model_name=args.model,
        pretrained=args.pretrained,
        device=args.device,
        batch_size=args.batch_size,
        normalize=args.normalize,
    )

    print(f"Saved embeddings to {output_dir / 'embeddings.npy'} (shape={embeddings.shape})")


if __name__ == "__main__":
    main()
