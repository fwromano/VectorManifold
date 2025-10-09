from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from PIL import Image


def _laplacian_variance(gray: np.ndarray) -> float:
    gray = gray.astype(np.float32)
    # simple 3x3 Laplacian
    center = gray[1:-1, 1:-1] * -4.0
    neighbors = (
        gray[1:-1, :-2]
        + gray[1:-1, 2:]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
    )
    lap = center + neighbors
    return float(np.var(lap))


def _mask_metrics(mask: Optional[np.ndarray]) -> tuple[float, float]:
    if mask is None:
        return 1.0, 1.0
    binary = mask > 0.5
    area = float(np.mean(binary))
    if not binary.any():
        return 0.0, 0.0
    rows = np.any(binary, axis=1)
    cols = np.any(binary, axis=0)
    r_indices = np.where(rows)[0]
    c_indices = np.where(cols)[0]
    height = (r_indices[-1] - r_indices[0] + 1) / binary.shape[0]
    width = (c_indices[-1] - c_indices[0] + 1) / binary.shape[1]
    return area, max(height, width)


def compute_quality_dataframe(
    image_paths: Sequence[Path],
    *,
    masks: Optional[Sequence[np.ndarray]] = None,
    detector_conf: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    if detector_conf is not None and len(detector_conf) != len(image_paths):
        raise ValueError("detector_conf length mismatch")
    mask_list = list(masks) if masks is not None else [None] * len(image_paths)
    records = []
    for idx, (path, mask) in enumerate(zip(image_paths, mask_list)):
        image = Image.open(path).convert("RGB")
        gray = np.asarray(image.convert("L"), dtype=np.float32)
        lap_var = _laplacian_variance(gray)
        area, bbox_ratio = _mask_metrics(mask)
        record = {
            "frame_index": idx,
            "mask_area_norm": area,
            "bbox_span_norm": bbox_ratio,
            "laplacian_var": lap_var,
            "detector_conf": float(detector_conf[idx]) if detector_conf is not None else 1.0,
            "occlusion_ratio": float(1.0 - area) if mask is not None else np.nan,
            "track_stability": np.nan,
        }
        records.append(record)
    return pd.DataFrame.from_records(records)


def save_quality_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


__all__ = ["compute_quality_dataframe", "save_quality_csv"]
