#!/usr/bin/env python3
"""Build a prompt bank using a dual-encoder text tower."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np

from vector_manifold.v2.promptbank import PromptBank, merge_prompt_configs

try:
    import torch
    import open_clip
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError("open_clip and torch are required to build the prompt bank") from exc


def encode_texts(model_name: str, pretrained: str, prompts: List[str], device: str) -> np.ndarray:
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.eval().to(device)
    if device != "cpu":
        model = model.half()

    tokens = tokenizer(prompts)
    tokens = tokens.to(device)
    with torch.no_grad():
        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu().float().numpy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build prompt bank for v2 flow")
    parser.add_argument("--prompts", nargs="+", type=Path, required=True, help="JSON prompt files (merged)")
    parser.add_argument("--out", type=Path, required=True, help="Destination .npz file")
    parser.add_argument("--model", default="ViT-L-14", help="open_clip model name")
    parser.add_argument("--pretrained", default="openai", help="open_clip pretrained tag")
    parser.add_argument("--device", default=None, help="Device override")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    config = merge_prompt_configs(args.prompts)

    def _encoder(texts: List[str]) -> np.ndarray:
        return encode_texts(args.model, args.pretrained, texts, device)

    promptbank = PromptBank.build(config, _encoder)
    promptbank.to_npz(args.out)
    print(f"Wrote prompt bank with {len(promptbank.prompt_list)} prompts to {args.out}")


if __name__ == "__main__":
    main()
