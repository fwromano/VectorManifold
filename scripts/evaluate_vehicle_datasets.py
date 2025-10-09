#!/usr/bin/env python3
"""Evaluate multiple image-text models on vehicle classification datasets."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

try:
    import open_clip
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError("open_clip is required for model evaluation") from exc

try:  # pragma: no cover - optional dependency
    from datasets import load_dataset
except ImportError:
    load_dataset = None  # type: ignore


Array = np.ndarray


@dataclass
class ModelConfig:
    name: str
    model: str
    pretrained: str


@dataclass
class DatasetConfig:
    name: str
    kind: str
    root: Path | None = None
    repo_id: str | None = None
    splits: Sequence[str] | None = None
    limit_per_class: int = 0


class ImageFolderDataset(Dataset):
    def __init__(self, items: Sequence[Tuple[Path, int]], preprocess):
        self._items = list(items)
        self._preprocess = preprocess

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._items)

    def __getitem__(self, index: int):
        path, label = self._items[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        tensor = self._preprocess(image)
        return tensor, label


class HFDataset(Dataset):
    def __init__(self, records: Sequence[Tuple[Image.Image, int]], preprocess):
        self._records = list(records)
        self._preprocess = preprocess

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._records)

    def __getitem__(self, index: int):
        image, label = self._records[index]
        image = image.convert("RGB")
        return self._preprocess(image), label


def list_color_items(root: Path) -> Tuple[List[Tuple[Path, int]], List[str]]:
    labels = sorted(entry.name for entry in root.iterdir() if entry.is_dir())
    items: List[Tuple[Path, int]] = []
    for idx, label in enumerate(labels):
        folder = root / label
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
            for path in folder.glob(ext):
                items.append((path, idx))
    return items, labels


def limit_samples(items: Sequence[Tuple[object, int]], per_class: int) -> List[Tuple[object, int]]:
    if per_class <= 0:
        return list(items)
    counts: Dict[int, int] = {}
    limited: List[Tuple[object, int]] = []
    for record in items:
        label = record[1]
        seen = counts.get(label, 0)
        if seen >= per_class:
            continue
        limited.append(record)
        counts[label] = seen + 1
    return limited


def load_hf_records(repo_id: str, splits: Sequence[str]) -> Tuple[List[Tuple[Image.Image, int]], List[str]]:
    if load_dataset is None:  # pragma: no cover - dependency guard
        raise ImportError("datasets library is required for Hugging Face datasets")
    records: List[Tuple[Image.Image, int]] = []
    label_names: List[str] | None = None
    for split in splits:
        ds = load_dataset(repo_id, split=split)
        for row in ds:
            records.append((row["image"], int(row["label"])))
        if label_names is None:
            label_names = ds.features["label"].names  # type: ignore[index]
    if label_names is None:
        raise ValueError(f"No labels found in dataset {repo_id}")
    return records, list(label_names)


def build_prompts(labels: Sequence[str], template: str) -> List[str]:
    prompts = [template.format(label=label) for label in labels]
    return prompts


def to_tensor_list(dataloader: DataLoader, *, device: torch.device, model) -> Tuple[Array, Array]:
    features: List[torch.Tensor] = []
    labels: List[int] = []
    with torch.no_grad():
        for batch, batch_labels in dataloader:
            batch = batch.to(device)
            embed = model.encode_image(batch)
            embed = embed / embed.norm(dim=-1, keepdim=True)
            features.append(embed.cpu())
            labels.extend(batch_labels.tolist())
    return torch.cat(features, dim=0).numpy(), np.asarray(labels, dtype=int)


def encode_text(model, tokenizer, prompts: Sequence[str], *, device: torch.device) -> Array:
    with torch.no_grad():
        tokens = tokenizer(prompts).to(device)
        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu().numpy()


def evaluate(
    model_cfg: ModelConfig,
    dataset_cfg: DatasetConfig,
    *,
    device: torch.device,
    batch_size: int,
    out_dir: Path,
    num_workers: int,
) -> Dict[str, object]:
    model, _, preprocess = open_clip.create_model_and_transforms(model_cfg.model, pretrained=model_cfg.pretrained)
    tokenizer = open_clip.get_tokenizer(model_cfg.model)
    model.to(device)
    model.eval()

    if dataset_cfg.kind == "local":
        assert dataset_cfg.root is not None
        items, labels = list_color_items(dataset_cfg.root)
        items = limit_samples(items, dataset_cfg.limit_per_class)
        prompt_template = "a photo of a {label} car"
        dataset = ImageFolderDataset(items, preprocess)
        label_indices = {idx: label for idx, label in enumerate(labels)}
    elif dataset_cfg.kind == "hf":
        assert dataset_cfg.repo_id is not None and dataset_cfg.splits is not None
        records, labels = load_hf_records(dataset_cfg.repo_id, dataset_cfg.splits)
        records = limit_samples(records, dataset_cfg.limit_per_class)
        prompt_template = "a photo of a {label}"
        dataset = HFDataset(records, preprocess)
        label_indices = {idx: label for idx, label in enumerate(labels)}
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported dataset kind: {dataset_cfg.kind}")

    prompts = build_prompts([label.replace("_", " ").replace("-", " ") for label in label_indices.values()], prompt_template)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    image_embeddings, labels_array = to_tensor_list(dataloader, device=device, model=model)
    text_embeddings = encode_text(model, tokenizer, prompts, device=device)
    sims = image_embeddings @ text_embeddings.T
    preds = sims.argmax(axis=1)

    accuracy = float((preds == labels_array).mean())
    confusion = np.zeros((len(prompts), len(prompts)), dtype=np.int32)
    for truth, pred in zip(labels_array, preds):
        confusion[truth, pred] += 1

    per_class: Dict[str, float] = {}
    for idx, label in label_indices.items():
        mask = labels_array == idx
        per_class[label] = float((preds[mask] == idx).mean()) if mask.any() else float("nan")

    model_dir = out_dir / dataset_cfg.name / model_cfg.name
    model_dir.mkdir(parents=True, exist_ok=True)
    np.save(model_dir / "confusion.npy", confusion)
    np.save(model_dir / "similarities.npy", sims.astype(np.float32))
    info = {
        "dataset": dataset_cfg.name,
        "model": model_cfg.name,
        "model_arch": model_cfg.model,
        "pretrained": model_cfg.pretrained,
        "accuracy": accuracy,
        "per_class_accuracy": per_class,
        "prompt_template": prompt_template,
        "num_samples": int(len(labels_array)),
    }
    (model_dir / "metrics.json").write_text(json.dumps(info, indent=2))
    return info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--color-root", type=Path, default=Path("data/dataset"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out", type=Path, default=Path("artifacts/reports/vehicle_compare"))
    parser.add_argument("--skip-model", action="append", default=[])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--per-class-limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    models = [
        ModelConfig(name="clip_v1", model="ViT-L-14", pretrained="openai"),
        ModelConfig(name="openclip_v2", model="ViT-H-14", pretrained="laion2b_s32b_b79k"),
        ModelConfig(name="sota_v3", model="ViT-SO400M-14-SigLIP", pretrained="webli"),
    ]
    models = [m for m in models if m.name not in set(args.skip_model)]

    datasets_cfg = [
        DatasetConfig(
            name="color_folders",
            kind="local",
            root=args.color_root,
            limit_per_class=args.per_class_limit,
        ),
        DatasetConfig(
            name="drbimmer_vehicle_classification",
            kind="hf",
            repo_id="DrBimmer/vehicle-classification",
            splits=("train", "validation"),
            limit_per_class=args.per_class_limit,
        ),
    ]

    args.out.mkdir(parents=True, exist_ok=True)
    summaries: List[Dict[str, object]] = []
    for dataset_cfg in datasets_cfg:
        for model_cfg in models:
            info = evaluate(
                model_cfg,
                dataset_cfg,
                device=device,
                batch_size=args.batch_size,
                out_dir=args.out,
                num_workers=args.num_workers,
            )
            summaries.append(info)
            print(f"{dataset_cfg.name} — {model_cfg.name}: accuracy={info['accuracy']:.3f}")

    summary_path = args.out / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2))
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
