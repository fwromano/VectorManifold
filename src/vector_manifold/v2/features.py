from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
from PIL import Image

try:
    import torch
    import open_clip
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError("open_clip and torch are required for v2 feature extraction") from exc


@dataclass
class FeatureExtractionConfig:
    model: str = "ViT-L-14"
    pretrained: str = "openai"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32
    use_fp16: bool = True


def load_image_paths(frames_dir: Path) -> List[Path]:
    if not frames_dir.exists():
        raise FileNotFoundError(f"frames directory {frames_dir} not found")
    paths = sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not paths:
        raise FileNotFoundError(f"no image files found in {frames_dir}")
    return paths


def load_masks(mask_dir: Optional[Path], image_paths: Sequence[Path]) -> Optional[List[np.ndarray]]:
    if mask_dir is None:
        return None
    if not mask_dir.exists():
        raise FileNotFoundError(f"mask directory {mask_dir} not found")
    masks: List[np.ndarray] = []
    for img_path in image_paths:
        mask_path = mask_dir / img_path.name
        if not mask_path.exists():
            raise FileNotFoundError(f"mask {mask_path} missing")
        mask = Image.open(mask_path).convert("L")
        masks.append(np.asarray(mask, dtype=np.float32) / 255.0)
    return masks


def apply_mask(image: Image.Image, mask: Optional[np.ndarray]) -> Image.Image:
    if mask is None:
        return image
    img_np = np.asarray(image, dtype=np.float32)
    if mask.shape != img_np.shape[:2]:
        from PIL import Image as PILImage

        mask_img = PILImage.fromarray((mask * 255).astype(np.uint8))
        mask_img = mask_img.resize((img_np.shape[1], img_np.shape[0]), resample=PILImage.BILINEAR)
        mask = np.asarray(mask_img, dtype=np.float32) / 255.0
    mask = mask[..., None]
    masked = img_np * mask + 255.0 * (1.0 - mask)
    return Image.fromarray(np.clip(masked, 0, 255).astype(np.uint8))


def compute_image_features(
    image_paths: Sequence[Path],
    *,
    masks: Optional[Sequence[np.ndarray]] = None,
    model_name: str = "ViT-L-14",
    pretrained: str = "openai",
    device: Optional[str] = None,
    batch_size: int = 32,
    use_fp16: bool = True,
) -> np.ndarray:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.eval().to(device)
    if use_fp16 and device != "cpu":
        model = model.half()

    features: List[torch.Tensor] = []
    mask_iter = masks if masks is not None else [None] * len(image_paths)
    mask_list = list(mask_iter)

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        batch_masks = mask_list[start : start + batch_size]
        images = []
        for path, mask in zip(batch_paths, batch_masks):
            img = Image.open(path).convert("RGB")
            img = apply_mask(img, mask)
            images.append(preprocess(img))
        batch = torch.stack(images)
        if use_fp16 and device != "cpu":
            batch = batch.half()
        batch = batch.to(device)
        with torch.no_grad():
            feats = model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        features.append(feats.cpu().float())

    return torch.cat(features, dim=0).numpy()


def save_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array.astype(np.float32))


def load_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path).astype(np.float32)


__all__ = [
    "FeatureExtractionConfig",
    "load_image_paths",
    "load_masks",
    "compute_image_features",
    "save_array",
    "load_array",
]
