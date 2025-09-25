#!/usr/bin/env python3
"""Decode v2 scores into probability distributions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from vector_manifold.v2.calibration import CalibrationBundle, load_calibration_bundle
from vector_manifold.v2.decode import decode_probabilities
from vector_manifold.v2.promptbank import PromptBank


def _load_quality_features(path: Optional[Path]) -> Optional[np.ndarray]:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    feature_cols = [
        "mask_area_norm",
        "bbox_span_norm",
        "laplacian_var",
        "detector_conf",
        "occlusion_ratio",
        "track_stability",
    ]
    available = [col for col in feature_cols if col in df.columns]
    if not available:
        return None
    return (
        df[available]
        .fillna(df[available].mean())
        .to_numpy(dtype=np.float32)
    )


def _load_calibration(path: Optional[Path]) -> Optional[CalibrationBundle]:
    if path is None:
        return None
    if path.is_dir():
        return load_calibration_bundle(path)
    raise FileNotFoundError(path)


def _save_json(records: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(records, f, indent=2)


def _save_group_probabilities(records: List[Dict], promptbank: PromptBank, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_count = len(records)
    for group, group_config in promptbank.config.groups.items():
        class_names = list(group_config.classes.keys())
        probs = np.zeros((frame_count, len(class_names)), dtype=np.float32)
        for idx, entry in enumerate(records):
            distribution = entry.get(group, {}).get("distribution", [])
            for cls_idx, class_name in enumerate(class_names):
                canonical = group_config.classes[class_name].prompts[0]
                for cand in distribution:
                    if cand.get("prompt") == canonical:
                        probs[idx, cls_idx] = float(cand.get("prob", 0.0))
                        break
        np.save(out_dir / f"{group}_probs.npy", probs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decode cosine scores into calibrated probabilities")
    parser.add_argument("--run", type=Path, required=True, help="Run directory containing scores_raw.npy")
    parser.add_argument("--promptbank", type=Path, required=True, help="Prompt bank npz")
    parser.add_argument("--scores", default="scores_raw.npy", help="Scores filename within the run directory")
    parser.add_argument("--quality", type=Path, help="Optional quality CSV for tau(x)")
    parser.add_argument("--calibration", type=Path, help="Calibration directory (contains calibration_bundle.json)")
    parser.add_argument("--pooling", choices=["max", "mean"], default="max")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature (tau)")
    parser.add_argument(
        "--subtract-negatives",
        action="store_true",
        help="Subtract negative prompt scores during pooling",
    )
    parser.add_argument("--out-json", type=Path, help="Destination JSON for per-frame distributions")
    parser.add_argument("--out-dir", type=Path, help="Directory for per-group probability arrays")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run
    score_path = run_dir / args.scores
    if not score_path.exists():
        raise FileNotFoundError(score_path)

    scores = np.load(score_path).astype(np.float32)
    promptbank = PromptBank.from_npz(args.promptbank)
    calibration = _load_calibration(args.calibration)
    quality_features = _load_quality_features(args.quality)

    records = decode_probabilities(
        scores,
        promptbank,
        calibration=calibration,
        quality_features=quality_features,
        topk=args.topk,
        pooling=args.pooling,
        temperature=args.temperature,
        subtract_negatives=args.subtract_negatives,
    )

    if args.out_json:
        _save_json(records, args.out_json)
    if args.out_dir:
        _save_group_probabilities(records, promptbank, args.out_dir)

    print(f"Decoded probabilities for {len(records)} frames")


if __name__ == "__main__":
    main()
