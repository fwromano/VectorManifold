#!/usr/bin/env python3
"""Compute prompt scores via cosine similarities."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from vector_manifold.v2.features import load_array
from vector_manifold.v2.promptbank import PromptBank
from vector_manifold.v2.scoring import batch_synonym_pool, cosine_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute cosine scores against prompt bank")
    parser.add_argument("--run", type=Path, required=True, help="Run directory containing image_features.npy")
    parser.add_argument("--text-dict", type=Path, required=True, help="Prompt bank npz produced by build_promptbank.py")
    parser.add_argument("--features", default="image_features.npy", help="Feature filename (default image_features.npy)")
    parser.add_argument(
        "--projector",
        type=Path,
        help="Optional rotation projector (.npy) to remove orientation subspace before scoring",
    )
    parser.add_argument("--out", type=Path, help="Optional custom path for raw scores (npy)")
    parser.add_argument("--pooling", choices=["max", "mean"], default="max", help="Synonym pooling mode")
    parser.add_argument(
        "--subtract-negatives",
        action="store_true",
        help="Subtract negative prompt scores when pooling synonyms",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_path = args.run / args.features
    features = load_array(feature_path)
    if args.projector is not None:
        projector = np.load(args.projector).astype(np.float32)
        if projector.shape[0] != projector.shape[1]:
            raise ValueError("projector must be square")
        if projector.shape[0] != features.shape[1]:
            raise ValueError("projector dimensionality does not match feature dimension")
        residual = features - features @ projector
        norms = np.linalg.norm(residual, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-6, None)
        features = residual / norms
    promptbank = PromptBank.from_npz(args.text_dict)

    scores = cosine_scores(features, promptbank)
    out_path = args.out or (args.run / "scores_raw.npy")
    np.save(out_path, scores.astype(np.float32))

    pooled = batch_synonym_pool(
        scores, promptbank, mode=args.pooling, subtract_negatives=args.subtract_negatives
    )
    pooled_path = args.run / "scores_pooled.npz"
    np.savez_compressed(pooled_path, **{group: matrix.astype(np.float32) for group, matrix in pooled.items()})

    print(f"Saved raw scores to {out_path} and pooled scores to {pooled_path}")


if __name__ == "__main__":
    main()
