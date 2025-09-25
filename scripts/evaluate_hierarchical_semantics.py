#!/usr/bin/env python3
"""Run hierarchical semantic evaluation across CLIP and SigLIP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from vector_manifold.eval.hierarchical_semantics import (  # noqa: E402
    ClipRunImageEncoder,
    SiglipEncoder,
    extract_dictionary_groups,
    load_samples,
    prompt_to_local_index,
    score_groups,
)
from vector_manifold.semantic.describer import TextDictionary  # noqa: E402


GROUPS = ("type", "vehicle", "color", "occupancy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="JSON file with curated samples")
    parser.add_argument(
        "--dictionary", type=Path, default=Path("artifacts/text_dictionary.npz"), help="Precomputed CLIP text dictionary"
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("artifacts/runs"),
        help="Directory containing <run>_clip embedding folders",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/reports/hard_cases"),
        help="Output directory for metrics and plots",
    )
    parser.add_argument("--siglip-model", default="google/siglip-so400m-patch14-384")
    parser.add_argument("--device", default="cpu", help="Torch device for SigLIP")
    parser.add_argument("--image-batch", type=int, default=16, help="SigLIP image batch size")
    parser.add_argument("--text-batch", type=int, default=64, help="SigLIP text batch size")
    parser.add_argument("--temperature", type=float, default=0.1, help="Softmax temperature for probability decode")
    return parser.parse_args()


def ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_filename(name: str) -> str:
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in name)


def build_records(
    *,
    model_name: str,
    samples,
    group_prompts: Dict[str, List[str]],
    scores: Dict[str, np.ndarray],
    probs: Dict[str, np.ndarray],
    prompt_lookup: Dict[str, Dict[str, int]],
) -> tuple[list[dict], pd.DataFrame]:
    records: List[dict] = []
    per_sample: List[dict] = []
    for row, sample in enumerate(samples):
        sample_entry: Dict[str, dict] = {}
        for group in GROUPS:
            prompts = group_prompts[group]
            group_scores = scores[group][row]
            group_probs = probs[group][row]
            top_idx = int(np.argmax(group_scores))
            top_prompt = prompts[top_idx]
            truth_prompt = sample.expected.get(group)
            truth_idx = prompt_lookup[group].get(truth_prompt) if truth_prompt is not None else None
            truth_score = float(group_scores[truth_idx]) if truth_idx is not None else float("nan")
            truth_prob = float(group_probs[truth_idx]) if truth_idx is not None else float("nan")
            top_score = float(group_scores[top_idx])
            top_prob = float(group_probs[top_idx])
            sample_entry[group] = {
                "top_prompt": top_prompt,
                "top_score": top_score,
                "top_prob": top_prob,
                "truth_prompt": truth_prompt,
                "truth_score": truth_score,
                "truth_prob": truth_prob,
            }
            records.append(
                {
                    "model": model_name,
                    "sample_id": sample.id,
                    "source_run": sample.source_run,
                    "group": group,
                    "truth": truth_prompt,
                    "pred": top_prompt,
                    "correct": int(top_prompt == truth_prompt) if truth_prompt is not None else 0,
                    "top_prob": top_prob,
                    "truth_prob": truth_prob,
                    "top_score": top_score,
                    "truth_score": truth_score,
                    "tags": ";".join(sample.tags),
                }
            )
        per_sample.append({"id": sample.id, "source_run": sample.source_run, "predictions": sample_entry})
    return per_sample, pd.DataFrame.from_records(records)


def plot_accuracy(df: pd.DataFrame, out_path: Path) -> None:
    pivot = df.groupby(["model", "group"], as_index=False)["correct"].mean()
    groups = list(GROUPS)
    models = list(pivot["model"].unique())
    width = 0.35
    x = np.arange(len(groups))
    plt.figure(figsize=(8, 4))
    for idx, model in enumerate(models):
        model_scores = [pivot[(pivot["model"] == model) & (pivot["group"] == group)]["correct"].mean() for group in groups]
        plt.bar(x + (idx - (len(models) - 1) / 2) * width, model_scores, width=width, label=model)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Accuracy")
    plt.xticks(x, groups)
    plt.legend()
    plt.title("Hierarchical accuracy by group")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_conditional_color(df: pd.DataFrame, out_path: Path) -> None:
    rows: List[dict] = []
    for model in df["model"].unique():
        type_df = df[(df["group"] == "type") & (df["model"] == model)][["sample_id", "correct"]]
        color_df = df[(df["group"] == "color") & (df["model"] == model)][["sample_id", "correct"]]
        merged = color_df.merge(type_df, on="sample_id", suffixes=("_color", "_type"))
        overall = merged["correct_color"].mean()
        conditioned = merged[merged["correct_type"] == 1]["correct_color"].mean()
        rows.append({"model": model, "metric": "overall", "value": overall})
        rows.append({"model": model, "metric": "given_type", "value": conditioned})
    plot_df = pd.DataFrame(rows)
    metrics = ["overall", "given_type"]
    models = list(plot_df["model"].unique())
    width = 0.35
    x = np.arange(len(metrics))
    plt.figure(figsize=(6, 4))
    for idx, model in enumerate(models):
        vals = [plot_df[(plot_df["model"] == model) & (plot_df["metric"] == metric)]["value"].mean() for metric in metrics]
        plt.bar(x + (idx - (len(models) - 1) / 2) * width, vals, width=width, label=model)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Accuracy")
    plt.xticks(x, ["color", "color | type"])
    plt.legend()
    plt.title("Color accuracy with and without type gating")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_confusion(df: pd.DataFrame, group: str, model: str, out_path: Path) -> None:
    subset = df[(df["group"] == group) & (df["model"] == model)]
    cats = sorted(set(subset["truth"].dropna()) | set(subset["pred"].dropna()))
    matrix = pd.crosstab(subset["truth"], subset["pred"], normalize="index").reindex(index=cats, columns=cats, fill_value=0.0)
    plt.figure(figsize=(6, 5))
    im = plt.imshow(matrix.values, vmin=0.0, vmax=1.0, cmap="viridis")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    ticks = np.arange(len(cats))
    plt.xticks(ticks, cats, rotation=45, ha="right")
    plt.yticks(ticks, cats)
    plt.title(f"{group.title()} confusion — {model}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    ensure_outdir(args.out)

    samples = load_samples(args.dataset)
    if not samples:
        raise ValueError(f"Dataset {args.dataset} contains no samples")

    dictionary = TextDictionary.load(args.dictionary)
    prompt_list, group_prompts, clip_text_embeds, group_indices = extract_dictionary_groups(dictionary, GROUPS)
    prompt_lookup = prompt_to_local_index(group_prompts)

    clip_encoder = ClipRunImageEncoder(args.runs_root)
    clip_embeddings = clip_encoder.encode(samples)
    clip_scores, clip_probs = score_groups(clip_embeddings, clip_text_embeds, group_indices, temperature=args.temperature)
    clip_preds, clip_df = build_records(
        model_name="CLIP ViT-L/14",
        samples=samples,
        group_prompts=group_prompts,
        scores=clip_scores,
        probs=clip_probs,
        prompt_lookup=prompt_lookup,
    )

    siglip = SiglipEncoder(args.siglip_model, device=args.device)
    image_paths = [sample.image_path for sample in samples]
    siglip_embeds = siglip.encode_images(image_paths, batch_size=args.image_batch)
    siglip_text = siglip.encode_texts(prompt_list, batch_size=args.text_batch)
    siglip_scores, siglip_probs = score_groups(siglip_embeds, siglip_text, group_indices, temperature=args.temperature)
    siglip_preds, siglip_df = build_records(
        model_name="SigLIP so400m/14",
        samples=samples,
        group_prompts=group_prompts,
        scores=siglip_scores,
        probs=siglip_probs,
        prompt_lookup=prompt_lookup,
    )

    combined_df = pd.concat([clip_df, siglip_df], ignore_index=True)

    ensure_outdir(args.out / "plots")
    plot_accuracy(combined_df, args.out / "plots" / "accuracy_by_group.png")
    plot_conditional_color(combined_df, args.out / "plots" / "color_conditional.png")
    for model in combined_df["model"].unique():
        plot_confusion(
            combined_df,
            "color",
            model,
            args.out / "plots" / f"color_confusion_{safe_filename(model)}.png",
        )

    combined_df.to_csv(args.out / "hierarchical_metrics.csv", index=False)
    (args.out / "predictions").mkdir(parents=True, exist_ok=True)
    Path(args.out / "predictions" / "clip_predictions.json").write_text(json.dumps(clip_preds, indent=2))
    Path(args.out / "predictions" / "siglip_predictions.json").write_text(json.dumps(siglip_preds, indent=2))

    summary = {
        "samples": len(samples),
        "models": ["CLIP ViT-L/14", "SigLIP so400m/14"],
        "temperature": args.temperature,
        "dataset": str(args.dataset),
        "siglip_model": args.siglip_model,
    }
    Path(args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved evaluation artifacts to {args.out}")


if __name__ == "__main__":  # pragma: no cover
    main()
