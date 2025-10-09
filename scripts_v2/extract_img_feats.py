#!/usr/bin/env python3
"""Extract CLIP-family image features for the v2 pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from vector_manifold.v2.features import compute_image_features, load_image_paths, load_masks, save_array


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract dual-encoder image features for v2 flow")
    parser.add_argument("--frames", type=Path, required=True, help="Directory with ordered frame images")
    parser.add_argument("--out", type=Path, required=True, help="Output directory (typically artifacts/runs/<run>)")
    parser.add_argument("--model", default="ViT-L-14", help="open_clip model name (e.g., ViT-L-14)")
    parser.add_argument("--pretrained", default="openai", help="open_clip pretrained tag")
    parser.add_argument("--mask-dir", type=Path, help="Optional directory with binary masks matching frame filenames")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None, help="Computation device override (cpu/cuda)")
    parser.add_argument("--no-fp16", action="store_true", help="Disable fp16 inference even on CUDA")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_paths = load_image_paths(args.frames)
    masks = None
    if args.mask_dir is not None:
        masks = load_masks(args.mask_dir, image_paths)

    features = compute_image_features(
        image_paths,
        masks=masks,
        model_name=args.model,
        pretrained=args.pretrained,
        device=args.device,
        batch_size=args.batch_size,
        use_fp16=not args.no_fp16,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    save_array(args.out / "image_features.npy", features)

    # Record frame ordering for reproducibility
    frame_index_path = args.out / "frame_index_v2.txt"
    frame_index_path.write_text("\n".join(str(path.name) for path in image_paths))

    print(f"Saved {features.shape[0]} features to {args.out / 'image_features.npy'}")


if __name__ == "__main__":
    main()
