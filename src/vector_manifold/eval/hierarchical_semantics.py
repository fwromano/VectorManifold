"""Utilities for hierarchical semantic evaluation across image-text models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
from PIL import Image

from vector_manifold.semantic.describer import TextDictionary

try:  # pragma: no cover - optional dependency
    import torch
    from transformers import SiglipModel, SiglipProcessor
except Exception:  # pragma: no cover - handled at runtime
    torch = None  # type: ignore[assignment]
    SiglipModel = None  # type: ignore[assignment]
    SiglipProcessor = None  # type: ignore[assignment]


Array = np.ndarray


@dataclass(frozen=True)
class Sample:
    """Curated dataset entry for semantic evaluation."""

    id: str
    image_path: Path
    source_run: str
    index: int
    expected: Mapping[str, str]
    tags: Tuple[str, ...]


def load_samples(path: Path) -> List[Sample]:
    """Load curated samples from a JSON file."""

    payload = json.loads(path.read_text())
    raw_samples = payload.get("samples", [])
    samples: List[Sample] = []
    for entry in raw_samples:
        samples.append(
            Sample(
                id=str(entry["id"]),
                image_path=Path(entry["image_path"]),
                source_run=str(entry["source_run"]),
                index=int(entry["index"]),
                expected=dict(entry.get("expected", {})),
                tags=tuple(entry.get("tags", [])),
            )
        )
    return samples


def extract_dictionary_groups(
    dictionary: TextDictionary,
    groups: Sequence[str],
) -> Tuple[List[str], Dict[str, List[str]], Array, Dict[str, Array]]:
    """Return prompts, embeddings, and slice indices for selected groups."""

    prompt_list: List[str] = []
    per_group_prompts: Dict[str, List[str]] = {}
    embeddings: List[Array] = []
    group_indices: Dict[str, Array] = {}
    offset = 0
    for group in groups:
        if group not in dictionary.prompts:
            raise KeyError(f"Group '{group}' not found in text dictionary")
        prompts = list(dictionary.prompts[group])
        idx = dictionary.group_indices[group]
        emb = dictionary.embeddings[idx]
        prompt_list.extend(prompts)
        embeddings.append(emb)
        per_group_prompts[group] = prompts
        group_indices[group] = np.arange(offset, offset + len(prompts), dtype=int)
        offset += len(prompts)
    stacked = np.vstack(embeddings) if embeddings else np.empty((0, 0), dtype=np.float32)
    return prompt_list, per_group_prompts, stacked, group_indices


class ClipRunImageEncoder:
    """Load precomputed CLIP embeddings from saved run directories."""

    def __init__(self, runs_root: Path):
        self._runs_root = runs_root
        self._cache: MutableMapping[str, Array] = {}

    def _load_run(self, run: str) -> Array:
        if run not in self._cache:
            run_dir = self._runs_root / f"{run}_clip"
            embeddings_path = run_dir / "embeddings.npy"
            if not embeddings_path.exists():
                raise FileNotFoundError(f"Missing embeddings for run '{run}' at {embeddings_path}")
            self._cache[run] = np.load(embeddings_path)
        return self._cache[run]

    def encode(self, samples: Sequence[Sample]) -> Array:
        rows: List[Array] = []
        for sample in samples:
            embed = self._load_run(sample.source_run)[sample.index]
            rows.append(embed.astype(np.float32, copy=False))
        return np.stack(rows, axis=0)


class SiglipEncoder:
    """Thin wrapper around Hugging Face SigLIP models for embeddings."""

    def __init__(self, model_name: str, device: str = "cpu"):
        if torch is None or SiglipModel is None or SiglipProcessor is None:  # pragma: no cover - runtime guard
            raise ImportError("transformers with SigLip support is required for SiglipEncoder")
        self._device = torch.device(device)
        self._model = SiglipModel.from_pretrained(model_name)
        self._processor = SiglipProcessor.from_pretrained(model_name)
        self._model.eval()
        self._model.to(self._device)

    def encode_images(self, image_paths: Sequence[Path], batch_size: int = 16) -> Array:
        features: List[Array] = []
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start : start + batch_size]
            images = []
            for path in batch_paths:
                with Image.open(path) as img:
                    images.append(img.convert("RGB"))
            with torch.no_grad():
                inputs = self._processor(images=images, return_tensors="pt").to(self._device)
                feats = self._model.get_image_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            features.append(feats.cpu().numpy().astype(np.float32))
        return np.concatenate(features, axis=0) if features else np.empty((0, 0), dtype=np.float32)

    def encode_texts(self, prompts: Sequence[str], batch_size: int = 64) -> Array:
        features: List[Array] = []
        iterator = iter(prompts)
        batch = list(islice(iterator, batch_size))
        while batch:
            with torch.no_grad():
                inputs = self._processor(text=batch, return_tensors="pt", padding=True).to(self._device)
                feats = self._model.get_text_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            features.append(feats.cpu().numpy().astype(np.float32))
            batch = list(islice(iterator, batch_size))
        return np.concatenate(features, axis=0) if features else np.empty((0, 0), dtype=np.float32)


def score_groups(
    image_embeddings: Array,
    text_embeddings: Array,
    group_indices: Mapping[str, Array],
    *,
    temperature: float,
) -> Tuple[Dict[str, Array], Dict[str, Array]]:
    """Return cosine scores and temperature-scaled probabilities per group."""

    if image_embeddings.size == 0 or text_embeddings.size == 0:
        return {}, {}

    sims = image_embeddings @ text_embeddings.T
    scores: Dict[str, Array] = {}
    probs: Dict[str, Array] = {}
    tau = max(temperature, 1e-6)
    for group, idx in group_indices.items():
        group_scores = sims[:, idx]
        scores[group] = group_scores
        logits = group_scores / tau
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        denom = exp.sum(axis=1, keepdims=True)
        probs[group] = exp / np.where(denom == 0.0, 1.0, denom)
    return scores, probs


def prompt_to_local_index(group_prompts: Mapping[str, Sequence[str]]) -> Dict[str, Dict[str, int]]:
    """Build lookup tables for prompt string -> local index."""

    lookup: Dict[str, Dict[str, int]] = {}
    for group, prompts in group_prompts.items():
        table = {prompt: idx for idx, prompt in enumerate(prompts)}
        lookup[group] = table
    return lookup
