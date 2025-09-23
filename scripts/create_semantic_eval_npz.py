#!/usr/bin/env python3
"""Build an evaluation NPZ with FastViT + CLIP features and side labels."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

try:
    import torch
    import open_clip
except ImportError as exc:  # pragma: no cover
    raise ImportError("Requires torch and open_clip to compute CLIP image embeddings") from exc


Array = np.ndarray


def load_embeddings(run_dir: Path) -> tuple[Array, Array]:
    Z = np.load(run_dir / "embeddings.npy").astype(np.float32)
    angles_deg = np.load(run_dir / "angles.npy").astype(np.float32)
    Z = l2norm(Z)
    theta = np.deg2rad(angles_deg)
    return Z, theta


def l2norm(Z: Array, eps: float = 1e-8) -> Array:
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    norms = np.where(norms > eps, norms, 1.0)
    return Z / norms


def assign_side_labels(angles_deg: Array) -> Array:
    labels = []
    for ang in angles_deg:
        a = ang % 360.0
        if a < 45 or a >= 315:
            labels.append(0)  # front
        elif 45 <= a < 135:
            labels.append(1)  # right
        elif 135 <= a < 225:
            labels.append(2)  # rear
        else:
            labels.append(3)  # left
    return np.asarray(labels, dtype=np.int64)


def compute_clip_features(image_dir: Path, model_name: str, pretrained: str, device: str, batch_size: int) -> Array:
    model, preprocess, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model.to(device)
    model.eval()
    image_paths = sorted(image_dir.glob("*.png"))
    tensors: List[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            imgs = [preprocess(Image.open(p).convert("RGB")) for p in batch_paths]
            batch = torch.stack(imgs).to(device)
            feats = model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            tensors.append(feats.cpu())
    return torch.cat(tensors, dim=0).numpy().astype(np.float32)


def build_pairs(length: int) -> Array:
    if length < 2:
        return np.empty((0, 2), dtype=np.int64)
    shift = length // 2
    pairs = [(i, (i + shift) % length) for i in range(length)]
    return np.asarray(pairs, dtype=np.int64)


def load_label_csv(path: Path, length: int) -> Array:
    import csv

    labels = np.full(length, -1, dtype=np.int64)
    with path.open() as f:
        reader = csv.DictReader(f)
        if "index" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError(f"Label CSV {path} must contain 'index' and 'label' columns")
        for row in reader:
            idx = int(row["index"])
            if idx < 0 or idx >= length:
                raise ValueError(f"Label index {idx} out of range (0..{length-1})")
            labels[idx] = int(row["label"])
    if np.any(labels < 0):
        missing = np.where(labels < 0)[0][:10]
        raise ValueError(f"Missing labels for indices: {missing}...")
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Create evaluation NPZ for semantic readout experiments")
    parser.add_argument("run", type=Path, help="Embedding run directory (contains embeddings.npy/angles.npy)")
    parser.add_argument("images", type=Path, help="Directory with frame PNGs ordered by index")
    parser.add_argument("--out", type=Path, default=Path("artifacts/eval/car_rotation_semantic.npz"))
    parser.add_argument("--model", default="ViT-L-14", help="CLIP model name for image embeddings")
    parser.add_argument("--pretrained", default="openai", help="CLIP pretrained tag")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help="Additional label CSVs in the form name=path (CSV with columns index,label)",
    )
    args = parser.parse_args()

    run_dir = args.run
    image_dir = args.images
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory {run_dir} not found")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory {image_dir} not found")

    embeddings, theta_rad = load_embeddings(run_dir)
    angles_deg = np.rad2deg(theta_rad)
    side_labels = assign_side_labels(angles_deg)

    clip_features = compute_clip_features(image_dir, args.model, args.pretrained, args.device, args.batch_size)
    if clip_features.shape[0] != embeddings.shape[0]:
        raise ValueError("CLIP features count does not match embeddings count")

    n = embeddings.shape[0]
    idx = np.arange(n)
    train_end = int(0.6 * n)
    dev_end = int(0.8 * n)
    train_idx = idx[:train_end]
    dev_idx = idx[train_end:dev_end]
    test_idx = idx[dev_end:]

    def gather(arr: Array, index: Array) -> Array:
        return arr[index]

    same_pairs_dev = build_pairs(len(dev_idx))

    label_arrays: Dict[str, Array] = {"side": side_labels}

    for entry in args.label:
        if "=" not in entry:
            raise ValueError(f"Label specification must be name=path, got '{entry}'")
        name, path_str = entry.split("=", 1)
        name = name.strip()
        label_path = Path(path_str)
        if not label_path.exists():
            raise FileNotFoundError(f"Label file {label_path} not found")
        labels = load_label_csv(label_path, n)
        label_arrays[name] = labels

    output_path = args.out
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "Z_train": gather(embeddings, train_idx),
        "Z_dev": gather(embeddings, dev_idx),
        "Z_test": gather(embeddings, test_idx),
        "theta_train": gather(theta_rad, train_idx),
        "theta_dev": gather(theta_rad, dev_idx),
        "Z_clip_train": gather(clip_features, train_idx),
        "Z_clip_dev": gather(clip_features, dev_idx),
        "pairs_dev": same_pairs_dev,
    }

    for name, values in label_arrays.items():
        payload[f"{name}_train"] = gather(values, train_idx)
        payload[f"{name}_dev"] = gather(values, dev_idx)
        payload[f"{name}_test"] = gather(values, test_idx)

    np.savez_compressed(output_path, **payload)
    print(f"Saved evaluation dataset to {output_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
