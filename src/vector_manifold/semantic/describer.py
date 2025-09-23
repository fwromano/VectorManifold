"""Cached text-dictionary, lightweight probes, and rotation aware descriptions."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import open_clip
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError(
        "open_clip and torch must be installed to build the semantic dictionary"
    ) from exc


Array = np.ndarray


PROMPT_GROUPS: Mapping[str, Sequence[str]] = {
    "color": [
        "a photo of a white vehicle",
        "a photo of a black vehicle",
        "a photo of a gray vehicle",
        "a photo of a red vehicle",
        "a photo of a blue vehicle",
        "a photo of a green vehicle",
        "a photo of an orange vehicle",
        "a photo of a yellow vehicle",
        "a photo of a purple vehicle",
    ],
    "vehicle": [
        "a photo of a car",
        "a photo of a truck",
        "a photo of an suv",
    ],
    "view": [
        "front view of a car",
        "rear view of a car",
        "left side of a car",
        "right side of a car",
        "driver side of a car",
        "passenger side of a car",
        "front three quarter view of a car",
        "rear three quarter view of a car",
        "top view of a car",
    ],
    "side": [
        "front view of a car",
        "right side of a car",
        "rear view of a car",
        "left side of a car",
    ],
    "type": [
        "a photo of a person",
        "a photo of a car",
        "a photo of a truck",
    ],
    "occupancy": [
        "a vehicle with a driver inside",
        "an empty vehicle",
    ],
    "armament": [
        "an unarmed vehicle",
        "an armed vehicle",
    ],
    "parts": [
        "close up of a car wheel",
        "close up of a car grille",
        "close up of a car headlight",
        "close up of a car taillight",
        "close up of a car logo",
        "close up of a car side mirror",
        "close up of a car windshield",
        "close up of a car spoiler",
    ],
}


@dataclass
class TextDictionary:
    prompts: Dict[str, List[str]]
    embeddings: Array  # shape (M, d)
    group_indices: Dict[str, np.ndarray]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            embeddings=self.embeddings.astype(np.float32),
            prompts=np.array([json.dumps(self.prompts)], dtype=np.unicode_),
            group_indices=np.array(
                [json.dumps({k: v.tolist() for k, v in self.group_indices.items()})], dtype=np.unicode_
            ),
        )

    @staticmethod
    def load(path: Path) -> "TextDictionary":
        data = np.load(path, allow_pickle=False)
        prompts = json.loads(data["prompts"].item())
        groups = json.loads(data["group_indices"].item())
        return TextDictionary(
            prompts={k: list(v) for k, v in prompts.items()},
            embeddings=data["embeddings"],
            group_indices={k: np.asarray(v, dtype=int) for k, v in groups.items()},
        )


@dataclass
class Probe:
    name: str
    weights: Array  # shape (d,)
    bias: float

    def score(self, embeddings: Array) -> Array:
        return embeddings @ self.weights + self.bias


def encode_prompts(
    prompts: Mapping[str, Sequence[str]],
    *,
    model_name: str,
    pretrained: str,
    device: str,
) -> TextDictionary:
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    model.to(device)

    all_embeddings: List[Array] = []
    group_indices: Dict[str, List[int]] = {}
    offset = 0

    with torch.no_grad():
        for group, phrasing in prompts.items():
            tokens = tokenizer(phrasing)
            tokens = tokens.to(device)
            text_features = model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            all_embeddings.append(text_features.cpu().numpy())
            idx = list(range(offset, offset + len(phrasing)))
            group_indices[group] = idx
            offset += len(phrasing)

    stacked = np.vstack(all_embeddings)
    return TextDictionary(
        prompts={k: list(v) for k, v in prompts.items()},
        embeddings=stacked,
        group_indices={k: np.asarray(v, dtype=int) for k, v in group_indices.items()},
    )


def train_ridge_probe(embeddings: Array, labels: Array, alpha: float = 0.1) -> Tuple[Array, float]:
    x = embeddings
    y = labels.reshape(-1, 1)
    n, d = x.shape
    gram = x.T @ x + alpha * np.eye(d)
    xy = x.T @ y
    w = np.linalg.solve(gram, xy).flatten()
    bias = float(y.mean() - x.mean(axis=0) @ w)
    return w, bias


def load_rotation_basis(run_dir: Path) -> Array:
    basis_path = run_dir / "rotation_analysis" / "basis_U.npy"
    if basis_path.exists():
        return np.load(basis_path)
    raise FileNotFoundError("rotation basis not found; run rotation_subspace_analysis first")


def derotate(embeddings: Array, basis: Array) -> Array:
    projector = basis @ basis.T
    residual = embeddings - embeddings @ projector
    norms = np.linalg.norm(residual, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return residual / norms


def compute_group_distributions(
    dictionary: TextDictionary,
    embeddings: Array,
    *,
    temperature: float = 1.0,
) -> Dict[str, List[List[Dict[str, float]]]]:
    sims = embeddings @ dictionary.embeddings.T
    output: Dict[str, List[List[Dict[str, float]]]] = {}
    temp = max(temperature, 1e-6)
    for group, indices in dictionary.group_indices.items():
        group_scores = sims[:, indices]
        logits = group_scores / temp
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs_sum = probs.sum(axis=1, keepdims=True)
        probs = probs / np.where(probs_sum == 0.0, 1.0, probs_sum)
        order = np.argsort(group_scores, axis=1)[:, ::-1]
        records: List[List[Dict[str, float]]] = []
        prompts = dictionary.prompts[group]
        for row_scores, row_probs, row_order in zip(group_scores, probs, order):
            records.append([
                {
                    "prompt": prompts[int(idx)],
                    "score": float(row_scores[int(idx)]),
                    "prob": float(row_probs[int(idx)]),
                }
                for idx in row_order
            ])
        output[group] = records
    return output


def run_description(
    run_dir: Path,
    dictionary: TextDictionary,
    *,
    topk: int,
    rotation_basis: Optional[Array],
    probes: Optional[List[Probe]],
    out_path: Path,
    temperature: float,
    summary_path: Optional[Path] = None,
) -> None:
    embeddings = np.load(run_dir / "embeddings.npy")
    embeddings = embeddings.astype(np.float64)
    if rotation_basis is not None:
        embeddings = derotate(embeddings, rotation_basis)
    else:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.where(norms == 0.0, 1.0, norms)

    group_distributions = compute_group_distributions(
        dictionary,
        embeddings,
        temperature=temperature,
    )

    probe_outputs: Dict[str, List[float]] = {}
    if probes:
        for probe in probes:
            probe_outputs[probe.name] = probe.score(embeddings).tolist()

    records = []
    for idx in range(embeddings.shape[0]):
        entry = {
            "index": int(idx),
        }
        for group, distributions in group_distributions.items():
            distribution = distributions[idx]
            top = distribution[: max(1, min(topk, len(distribution)))]
            entry[group] = {
                "topk": top,
                "distribution": distribution,
            }
        for probe_name, values in probe_outputs.items():
            entry[probe_name] = values[idx]
        records.append(entry)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, indent=2))

    if summary_path is not None:
        summary: Dict[str, Dict[str, float]] = {}
        for group, distributions in group_distributions.items():
            prompt_sums: Dict[str, float] = {}
            for dist in distributions:
                for item in dist:
                    prompt_sums[item["prompt"]] = prompt_sums.get(item["prompt"], 0.0) + item["prob"]
            normaliser = float(len(distributions)) if distributions else 1.0
            summary[group] = {
                prompt: value / normaliser for prompt, value in prompt_sums.items()
            }
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2))


def parse_probe_specs(paths: Sequence[Path]) -> List[Probe]:
    probes: List[Probe] = []
    for path in paths:
        data = np.load(path)
        probes.append(
            Probe(
                name=path.stem,
                weights=data["weights"],
                bias=float(data["bias"]),
            )
        )
    return probes


def load_training_labels(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    import csv

    indices: List[int] = []
    labels: List[float] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        if "index" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError("CSV must contain 'index' and 'label' columns")
        for row in reader:
            indices.append(int(row["index"]))
            labels.append(float(row["label"]))
    return np.asarray(indices, dtype=int), np.asarray(labels, dtype=float)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Fast semantic description via cached prompts and probes")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-dict", help="Cache text embeddings")
    build.add_argument("--model", default="ViT-L-14", help="open_clip model name")
    build.add_argument("--pretrained", default="openai", help="Checkpoint ID")
    build.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    build.add_argument("--out", type=Path, default=Path("artifacts/text_dictionary.npz"))

    describe = subparsers.add_parser("describe", help="Describe embeddings using cached dictionary")
    describe.add_argument("run", type=Path, help="Embedding run directory")
    describe.add_argument("--dict", type=Path, default=Path("artifacts/text_dictionary.npz"))
    describe.add_argument("--topk", type=int, default=1)
    describe.add_argument("--no-rotation", action="store_true", help="Skip rotation projection")
    describe.add_argument("--basis", type=Path, help="Explicit rotation basis path")
    describe.add_argument("--probe", type=Path, action="append", help="Probe weights saved via numpy savez (weights,bias)")
    describe.add_argument("--out", type=Path, default=Path("artifacts/semantic_descriptions.json"))
    describe.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature for likelihoods")
    describe.add_argument("--summary-out", type=Path, help="Aggregate likelihoods across frames")

    train = subparsers.add_parser("train-probe", help="Train a tiny ridge probe on labelled embeddings")
    train.add_argument("run", type=Path, help="Embedding run directory")
    train.add_argument("--labels", type=Path, required=True, help="CSV with columns index,label")
    train.add_argument("--alpha", type=float, default=0.1, help="Ridge regularisation strength")
    train.add_argument("--no-rotation", action="store_true", help="Skip rotation removal before training")
    train.add_argument("--basis", type=Path, help="Explicit rotation basis path")
    train.add_argument("--out", type=Path, required=True, help="Where to save probe weights (npz)")

    args = parser.parse_args(argv)

    if args.command == "build-dict":
        dictionary = encode_prompts(
            PROMPT_GROUPS,
            model_name=args.model,
            pretrained=args.pretrained,
            device=args.device,
        )
        dictionary.save(args.out)
        print(f"Saved dictionary to {args.out}")
        return

    if args.command == "describe":
        dictionary = TextDictionary.load(args.dict)
        if args.no_rotation:
            basis = None
        else:
            basis_path = args.basis if args.basis else args.run / "rotation_analysis" / "basis_U.npy"
            basis = np.load(basis_path)
        probes = parse_probe_specs(args.probe) if args.probe else None
        out_path = args.out
        run_description(
            args.run,
            dictionary,
            topk=args.topk,
            rotation_basis=basis,
            probes=probes,
            out_path=out_path,
            temperature=args.temperature,
            summary_path=args.summary_out,
        )
        print(f"Wrote semantic descriptions to {out_path}")
        if args.summary_out:
            print(f"Wrote summary likelihoods to {args.summary_out}")
        return

    if args.command == "train-probe":
        embeddings = np.load(args.run / "embeddings.npy").astype(np.float64)
        if not args.no_rotation:
            basis_path = args.basis if args.basis else args.run / "rotation_analysis" / "basis_U.npy"
            basis = np.load(basis_path)
            embeddings = derotate(embeddings, basis)
        else:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.where(norms == 0.0, 1.0, norms)

        idx, labels = load_training_labels(args.labels)
        if idx.size == 0:
            raise ValueError("No labels provided")
        selected = embeddings[idx]
        weights, bias = train_ridge_probe(selected, labels, alpha=args.alpha)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.out, weights=weights.astype(np.float32), bias=np.float32(bias))
        print(f"Saved probe to {args.out}")
        return


if __name__ == "__main__":  # pragma: no cover
    main()
